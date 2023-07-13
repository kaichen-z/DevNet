from __future__ import absolute_import, division, print_function
import skimage
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# ---------- Structure Modules in DevNet ---------- 
from .pose_encoder import PoseEncoder
from .pose_decoder import PoseDecoder
from .resnet_encoder import ResnetEncoder # Feature Extractor In FeatDepth
from .depth_decoder import DepthDecoder
#from .depth_decoder2 import DepthDecoder

# ---------- Manager System For DevNet ----------  
from ..registry import MONO
# ---------- Supporting Modules For DevNet ----------   
from .utils import get_embedder 
from .utils import HomographySample
from .utils import get_disparity_list
from .utils import transformation_from_parameters # From prediction Result to Matrix.
from .render_utils import get_xyz_from_plane_disparity # Get 3D Points from Disparity
from .render_utils import get_tgt_xyz_from_plane_disparity 
from .render_utils import predict_density_from_disparity 
from .render_utils import render
from .render_utils import inverse_matrix
from .render_utils import render_tgt_depth
from mono.datasets.utils import compute_errors
from .layers import SSIM
from .layers import Backproject
from .layers import Project

@MONO.register_module
class mono_dev7(nn.Module):
    def __init__(self, options):
        super(mono_dev7, self).__init__()
        self.opt = options
        print(self.opt)
        # ----- Pose Module -----
        self.use_alpha = False
        self.PoseEncoder = PoseEncoder(self.opt.pose_num_layers, self.opt.pose_pretrained_path)
        self.PoseDecoder = PoseDecoder(self.PoseEncoder.num_ch_enc, color = self.opt.use_color_loss)
        # ----- Depth Module -----
        self.embedder, out_dim = get_embedder(self.opt.pos_encoding_multires)
        self.backbone = ResnetEncoder(num_layers = self.opt.resnet_num_layers,
                                    pretrained = self.opt.imagenet_pretrained,
                                    pretrained_path = self.opt.depth_pretrained_path)
        self.decoder = DepthDecoder(num_ch_enc = self.backbone.num_ch_enc,
            embedder = self.embedder, embedder_out_dim = out_dim,
            output_channels = self.opt.num_bins, use_alpha=self.use_alpha)

        #self.decoder = DepthDecoder(ch_enc = self.backbone.num_ch_enc,
        #    num_output_channels = self.opt.num_bins)
        # ----- Resume -----
        if self.opt.pretrained_depth is not None:
            self.resume(self.opt.pretrained_depth)
        # ----- Support Module -----
        self.ssim = SSIM()
        self.backproject = Backproject(self.opt.imgs_per_gpu, self.opt.height, self.opt.width)
        self.project= Project(self.opt.imgs_per_gpu, self.opt.height, self.opt.width)
        self.feat_backproject = Backproject(self.opt.imgs_per_gpu, int(self.opt.height/2), int(self.opt.width/2))
        self.feat_project = Project(self.opt.imgs_per_gpu, int(self.opt.height/2), int(self.opt.width/2))
        self.init_data()

    # ---------- Support Module ----------
    def init_data(self,):
        H, W = self.opt.height, self.opt.width
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.homography_sampler_list = \
            [HomographySample(int(H / 2), int(W / 2), device=device),
             HomographySample(int(H / 4), int(W / 4), device=device),
             HomographySample(int(H / 8), int(W / 8), device=device),
             HomographySample(int(H / 16), int(W / 16), device=device)]
        self.upsample_list = \
            [nn.Upsample(size=(int(H / 2), int(W / 2))),
             nn.Upsample(size=(int(H / 4), int(W / 4))),
             nn.Upsample(size=(int(H / 8), int(W / 8))),
             nn.Upsample(size=(int(H / 16), int(W / 16)))]
    
    def resume(self, weights):
        self.load_state_dict(torch.load(weights)['state_dict'])

    # ---------- Modification ----------
    def forward(self, inputs, iteration = 0, epoch = 0):
        outputs = {}
        if self.training:  
            if self.opt.use_depth_loss:
                outputs.update(self.predict_poses(inputs))
                density_list, disparity_list = self.network_process(inputs, 0) # (List of Featmaps), (B, S)
                for scale in self.opt.scales:
                    outputs[("density", 0, scale)] = density_list[scale]
                    outputs[("depth", 0, scale)] = self.calculate_depth(inputs, scale, density_list[scale], disparity_list)
                with torch.no_grad(): # To reduce memory request.
                    density_list, disparity_list = self.network_process(inputs, -1) # (List of Featmaps), (B, S)
                    for scale in self.opt.scales:
                        outputs[("depth", -1, scale)] = self.calculate_depth(inputs, scale, density_list[scale], disparity_list)
                        if self.opt.use_depth_loss_ts:
                            G_src_tgt = outputs[("cam_T_cam", 0, -1)]
                            G_tgt_src = inverse_matrix(G_src_tgt)
                            outputs[("tgt_depth", -1, scale)], outputs[("tgt_depth_mask", -1, scale)] = self.calculate_tgt_depth(inputs, scale, density_list[scale], disparity_list, G_tgt_src)

                if self.opt.aug_consistency: 
                    with torch.no_grad():
                        density_list, disparity_list = self.network_process_nor(inputs, 0) # (List of Featmaps), (B, S)
                        for scale in self.opt.scales:
                            outputs[("depth_consistency", 0, scale)] = self.calculate_depth(inputs, scale, density_list[scale], disparity_list)

            else:
                frame = 0 
                outputs.update(self.predict_poses(inputs))
                density_list, disparity_list = self.network_process(inputs, frame)
                for scale in self.opt.scales:
                    outputs[("density", 0, scale)] = density_list[scale]
                    outputs[("depth", frame, scale)] = self.calculate_depth(inputs, scale, density_list[scale], disparity_list)
            '''Changing from the First Stage to Second One.'''
            loss_dict = self.compute_losses(inputs, outputs)
            return outputs, loss_dict
        density_src_list, disparity_src = self.network_process(inputs, frame = 0)
        outputs[("depth", 0, 0)] = self.calculate_depth(inputs, 0, density_src_list[0], disparity_src)
        return outputs

    # ---------- Modification ----------
    def network_process(self, inputs, frame):
        img = inputs[('color_aug', frame, 0)]
        B = img.size(0)
        disparity_list = get_disparity_list(self.opt, B, device=img.device) # B, S
        density_list = predict_density_from_disparity(self.disp_predictor, img, disparity_list)
        return density_list, disparity_list

    def network_process_nor(self, inputs, frame):
        img = inputs[('color', frame, 0)]
        B = img.size(0)
        disparity_list = get_disparity_list(self.opt, B, device=img.device) # B, S
        density_list = predict_density_from_disparity(self.disp_predictor, img, disparity_list)
        return density_list, disparity_list

    # ---------- Modification ----------
    def disp_predictor(self, src_imgs_BCHW, disparity_BS):
        conv1_out, block1_out, block2_out, block3_out, block4_out = self.backbone(src_imgs_BCHW)
        outputs = self.decoder([conv1_out, block1_out, block2_out, block3_out, block4_out], disparity_BS)
        output_list = [outputs[("disp", 0)], outputs[("disp", 1)], outputs[("disp", 2)], outputs[("disp", 3)]]
        return output_list

    # ---------- Modification ----------
    def predict_poses(self, inputs):
        outputs = {}
        pose_feats = {f_i: F.interpolate(inputs["color_aug", f_i, 0], [192, 640], mode="bilinear", align_corners=False) for f_i in self.opt.frame_ids}
        for f_i in self.opt.frame_ids[1:]:
            if not f_i == "s":
                if f_i < 0:
                    pose_inputs = [pose_feats[f_i], pose_feats[0]]
                else:
                    pose_inputs = [pose_feats[0], pose_feats[f_i]]
                # from [0] -> [1]
                pose_inputs = self.PoseEncoder(torch.cat(pose_inputs, 1))
                if not self.opt.use_color_loss:
                    axisangle, translation = self.PoseDecoder(pose_inputs)
                elif self.opt.use_color_loss:
                    axisangle, translation, color_a, color_b = self.PoseDecoder(pose_inputs)
                    if f_i < 0:
                        outputs[("color_a", f_i)] = 1/color_a
                        outputs[("color_b", f_i)] = - color_b/color_a
                    elif f_i > 0:
                        outputs[("color_a", f_i)] = color_a
                        outputs[("color_b", f_i)] = color_b
                outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(axisangle[:, 0], \
                    translation[:, 0], invert=(f_i < 0))
        return outputs

    # ---------- Modification ----------
    def calculate_depth(self, inputs, scale, density_list, disparity):
        img = inputs[('color_aug', 0, 0)]
        K_scaled = inputs[('K')][:, :3, :3]/ (2 ** (scale))
        K_scaled[:, 2, 2] = 1
        torch.cuda.synchronize()
        K_scaled_inv = torch.inverse(K_scaled)
        xyz_BS3HW = get_xyz_from_plane_disparity(
            self.homography_sampler_list[scale].meshgrid.to(img.device), \
                disparity.to(img.device), K_scaled_inv.to(img.device))
        depth_syn = render(density_list, xyz_BS3HW, self.use_alpha)
        return depth_syn

    def calculate_tgt_depth(self, inputs, scale, density_src_list, disparity_src, G_tgt_src):
        K_scaled = inputs[('K')][:, :3, :3]/ (2 ** (scale))
        K_scaled[:, 2, 2] = 1
        K_scaled_inv = torch.inverse(K_scaled)
        torch.cuda.synchronize()
        # Apply scale factor
        if self.opt.stereo_scale:
            with torch.no_grad():
                G_tgt_src = torch.clone(G_tgt_src)
                G_tgt_src[:, 0:3, 3] = G_tgt_src[:, 0:3, 3] / self.opt.STEREO_SCALE_FACTOR
        tgt_depth_syn, tgt_mask_syn = self.render_novel_view(density_src_list,
                                                disparity_src, G_tgt_src, 
                                                K_scaled_inv, K_scaled, scale=scale)
        threshold = 8
        tgt_mask = torch.ge(tgt_mask_syn, threshold).to(torch.float32)
        return tgt_depth_syn, tgt_mask

    '''--------- LOSS FUNCTION ---------'''
    def compute_losses(self, inputs, outputs):
        loss_add = {}
        loss_dict = {}
        ''' ---------- scale involved in the COLOR_RECONSTRUCTION and SMOOTH LOSS ---------'''
        for scale in self.opt.scales:
            for frame_id in self.opt.frame_ids[1:]:
                """ initialization """
                target = inputs[("color", 0, 0)]
                """ image_reconstruction_loss """
                outputs = self.generate_images_pred_tgt_src(inputs, outputs, scale)
                if not self.opt.use_color_loss:
                    pred = outputs[("color", frame_id, scale)]                
                elif self.opt.use_color_loss:
                    _, C, H, W = outputs[("color", frame_id, scale)].size()
                    pred = outputs[("color", frame_id, scale)] * outputs[("color_a", frame_id)].unsqueeze(-1).repeat(1, C, H, W) \
                         + outputs[("color_b", frame_id)].unsqueeze(-1).repeat(1, C, H, W)              
                loss_add[('repro_loss', frame_id, scale)] = self.compute_reprojection_loss(pred, target)                
                
                """ automask """
                if scale == 0:
                    pred = inputs[("color", frame_id, 0)]
                    identity_reprojection_loss = self.compute_reprojection_loss(pred, target)
                    identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).cuda() * 1e-5
                    loss_add[('iden_loss', frame_id)] = identity_reprojection_loss
                _, iden_mask = torch.min(torch.cat([loss_add[('iden_loss', frame_id)], \
                    loss_add[('repro_loss', frame_id, scale)]], dim = 1), dim=1)
                loss_add[('iden_mask', frame_id, scale)] = iden_mask.detach().float().unsqueeze(1)

            """ Color_Reconstruction_Loss. """
            reprojection_losses = []
            for frame_id in self.opt.frame_ids[1:]:
                loss_add[('repro_loss', frame_id, scale)] = loss_add[('repro_loss', frame_id, scale)]*loss_add[('iden_mask', frame_id, scale)] + loss_add[('iden_loss', frame_id)]*(1 - loss_add[('iden_mask', frame_id, scale)])
                reprojection_losses.append(loss_add[('repro_loss', frame_id, scale)])
            loss_dict[('repro_loss', scale)], mask = torch.min(torch.cat(reprojection_losses, dim=1), dim=1)
            loss_dict[('repro_loss', scale)] = loss_dict[('repro_loss', scale)].mean()/len(self.opt.scales)

            if self.opt.use_depth_loss:
                loss_add.update(self.generate_occlusion(inputs, outputs, loss_add, scale))
                mask_depth = (1 - mask) * loss_add[('iden_mask', -1, scale)]
                loss_dict[('depth_loss', scale)] = loss_add[('depth_loss', scale)]
                loss_dict[('depth_loss', scale)] = loss_dict[('depth_loss', scale)] * mask_depth
                loss_dict[('depth_loss', scale)] = self.opt.depth_weight * loss_dict[('depth_loss', scale)].mean()

            if self.opt.use_var_loss:
                var_depth = self.opt.var_weight * outputs[("density", 0, scale)].var(1).mean()

            if self.opt.use_depth_loss_ts:
                loss_add.update(self.generate_depth_loss(outputs, loss_add, scale))
                loss_dict[('depth_loss_ts', scale)] = loss_add[("depth_loss_ts", scale)]*mask_depth
                loss_dict[('depth_loss_ts', scale)] = self.opt.depth_weight_ts * loss_dict[('depth_loss_ts', scale)].mean()

            """ smooth loss (Color Space)"""
            if self.opt.use_smooth_loss:
                if self.opt.disp_norm:
                    disp = torch.reciprocal(outputs[("depth", 0, scale)])
                    mean_disp = disp.mean(2, True).mean(3, True)
                    disp = disp / (mean_disp + 1e-7)
                else:
                    disp = torch.reciprocal(outputs[("depth", 0, scale)])
                target = inputs[("color", 0, 0)]
                smooth_loss = self.get_smooth_loss(disp, target)
                loss_dict[('smooth_loss', scale)] = self.opt.smoothness_weight * smooth_loss /\
                        (2 ** scale)/len(self.opt.scales)

            """ smooth loss (Color Space)"""
            if self.opt.aug_consistency:
                loss_dict[('aug_consist_loss', scale)] = self.opt.consist_weight * self.robust_l1(outputs[("depth", 0, scale)], outputs[("depth_consistency", 0, scale)]).mean(1, True)

        """ Color_Parameters_Loss. """
        if self.opt.use_color_loss:
            for frame_id in self.opt.frame_ids[1:]:
                loss_dict[('color_loss', frame_id, 0)] = self.opt.color_weight * ((outputs[('color_a', frame_id)]-1)**2 + outputs[('color_b', frame_id)]**2)
        return loss_dict

    def generate_images_pred_tgt_src(self, inputs, outputs, scale):
        # Bi-Direction
        depth = outputs[("depth", 0, scale)]
        depth = F.interpolate(depth, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
        for _, frame_id in enumerate(self.opt.frame_ids[1:]):
            if frame_id == "s":
                T = inputs["stereo_T"]
            else:
                T = outputs[("cam_T_cam", 0, frame_id)]
            cam_points = self.backproject(depth, torch.inverse(inputs[("K")]))
            pix_coords, _ = self.project(cam_points, inputs[("K")], T) #[b,h,w,2]
            img = inputs[("color", frame_id, 0)]
            outputs[("color", frame_id, scale)] = F.grid_sample(img, pix_coords, padding_mode="border")
        return outputs

    def generate_occlusion(self, inputs, outputs, loss_add, scale):
        depth = outputs[("depth", 0, scale)]
        tgt_depth = F.interpolate(depth, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)

        T = outputs[("cam_T_cam", 0, -1)]
        depth = outputs[("depth", -1, scale)]
        src_depth = F.interpolate(depth, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
        cam_points = self.backproject(tgt_depth, torch.inverse(inputs[("K")]))
        pix_src_tgt, tgt_src_depth1 = self.project(cam_points, inputs[("K")], T) 
        
        tgt_src_depth2 = F.grid_sample(src_depth, pix_src_tgt, mode="nearest", padding_mode="border")
        tgt_src_transform = (tgt_src_depth1 - tgt_src_depth2).abs()
        variable_bar = (tgt_depth - src_depth).abs() 
        
        loss_add[("depth_loss", scale)] = (tgt_src_transform/(tgt_src_depth1 + tgt_src_depth2))
        loss_add[("depth_loss_iden", scale)] = (variable_bar/(tgt_depth + src_depth))
        return loss_add

    def generate_depth_loss(self, outputs, loss_add, scale):
        tgt_depth = F.interpolate(outputs[("depth", 0, scale)], [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
        tgt_src_depth = F.interpolate(outputs[("tgt_depth", -1, scale)], [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
        tgt_src_mask = F.interpolate(outputs[("tgt_depth_mask", -1, scale)], [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
        tgt_src_transform = (tgt_depth - tgt_src_depth).abs()
        loss_add[("depth_loss_ts", scale)] = (tgt_src_transform/(tgt_depth + tgt_src_depth))*tgt_src_mask
        return loss_add

    def compute_reprojection_loss(self, pred, target):
        photometric_loss = self.robust_l1(pred, target).mean(1, True)
        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = (0.85 * ssim_loss + 0.15 * photometric_loss)
        return reprojection_loss

    def robust_l1(self, pred, target):
        eps = 1e-3
        return torch.sqrt(torch.pow(target - pred, 2) + eps ** 2)

    def get_smooth_loss(self, disp, img):
        _, _, h, w = disp.size()
        a1 = 0.5
        a2 = 0.5
        img = F.interpolate(img, (h, w), mode='area')
        disp_dx, disp_dy = self.gradient(disp)
        img_dx, img_dy = self.gradient(img)
        disp_dxx, disp_dxy = self.gradient(disp_dx)
        disp_dyx, disp_dyy = self.gradient(disp_dy)
        img_dxx, img_dxy = self.gradient(img_dx)
        img_dyx, img_dyy = self.gradient(img_dy)
        smooth1 = torch.mean(disp_dx.abs() * torch.exp(-a1 * img_dx.abs().mean(1, True))) + \
                  torch.mean(disp_dy.abs() * torch.exp(-a1 * img_dy.abs().mean(1, True)))
        smooth2 = torch.mean(disp_dxx.abs() * torch.exp(-a2 * img_dxx.abs().mean(1, True))) + \
                  torch.mean(disp_dxy.abs() * torch.exp(-a2 * img_dxy.abs().mean(1, True))) + \
                  torch.mean(disp_dyx.abs() * torch.exp(-a2 * img_dyx.abs().mean(1, True))) + \
                  torch.mean(disp_dyy.abs() * torch.exp(-a2 * img_dyy.abs().mean(1, True)))
        return smooth1+smooth2

    def gradient(self, D):
        D_dy = D[:, :, 1:] - D[:, :, :-1]
        D_dx = D[:, :, :, 1:] - D[:, :, :, :-1]
        return D_dx, D_dy

    def render_novel_view(self, mpi_all_sigma_src,
                            disparity_all_src, G_tgt_src,
                            K_src_inv, K_tgt, scale=0):
        xyz_src_BS3HW = get_xyz_from_plane_disparity(
            self.homography_sampler_list[scale].meshgrid, disparity_all_src, K_src_inv)
        xyz_tgt_BS3HW = get_tgt_xyz_from_plane_disparity(
            xyz_src_BS3HW.to(mpi_all_sigma_src.device), G_tgt_src.to(mpi_all_sigma_src.device))
        # Bx1xHxW, Bx3xHxW, Bx1xHxW
        tgt_depth_syn, tgt_mask_syn = render_tgt_depth(
            self.homography_sampler_list[scale],
            mpi_all_sigma_src, disparity_all_src, xyz_tgt_BS3HW,
            G_tgt_src, K_src_inv, K_tgt, use_alpha=self.use_alpha)
        return tgt_depth_syn, tgt_mask_syn