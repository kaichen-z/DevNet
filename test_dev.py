from __future__ import absolute_import, division, print_function
from datetime import datetime
import cv2
import math
import json
import time
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms.functional import hflip
import torchvision
import datasets
from utils import *
from layers import *
from kitti_utils import *
import networks_dev as networks
from networks_dev.utils import get_embedder
from networks_dev.utils import HomographySample
from networks_dev.utils import get_disparity_list
from networks_dev.render_utils import predict_density_from_disparity
from networks_dev.render_utils import get_xyz_from_plane_disparity
from networks_dev.render_utils import render
import matplotlib.pyplot as plt

class TESTER:
    def __init__(self, options):
        now = datetime.now()
        current_time_date = now.strftime("%d%m%Y-%H:%M:%S")
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)
        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda:0")#not using cuda?
        self.num_scales = len(self.opt.scales)#scales = [0,1,2,3]'scales used in the loss'
        self.num_input_frames = len(self.opt.frame_ids)#frames = [0,-1,1]'frame to load'
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"
        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")
        '''---------- Structure ---------'''
        self.init_data()
        self.embedder, out_dim = get_embedder(self.opt.pos_encoding_multires)
        self.models["encoder"] = networks.ResnetEncoder(num_layers = self.opt.resnet_num_layers,
                                    pretrained = self.opt.imagenet_pretrained,
                                    pretrained_path = self.opt.depth_pretrained_path)
        self.models["depth"] = networks.DepthDecoder(num_ch_enc = self.models["encoder"].num_ch_enc,
            embedder = self.embedder, embedder_out_dim = out_dim,
            output_channels = self.opt.num_bins, use_alpha=self.opt.use_alpha)
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        if self.use_pose_net:#use_pose_net = True
            if self.opt.pose_model_type == "separate_resnet":#defualt=separate_resnet  choice = ['normal or shared']
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    num_layers = self.opt.num_layers,
                    pretrained = self.opt.imagenet_pretrained,
                    pretrained_path =  self.opt.depth_pretrained_path,
                    num_input_images = self.num_pose_frames)#num_input_images=2
                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)
            self.models["pose_encoder"].cuda()
            self.models["pose"].cuda()
            self.parameters_to_train += list(self.models["pose"].parameters())
            self.parameters_to_train += list(self.models["pose_encoder"].parameters())
        self.backproject = Backproject(self.opt.batch_size, self.opt.height, self.opt.width)
        self.project= Project(self.opt.batch_size, self.opt.height, self.opt.width)

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()
        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.log_path)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset,
                         "cityscapes_preprocessed": datasets.CityscapesPreprocessedDataset}
        self.dataset_k = datasets_dict[self.opt.dataset]
        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        #change trainset
        train_filenames_k = readlines(fpath.format("train"))
        splits_dir = "splits"
        val_filenames = readlines(os.path.join(splits_dir, self.opt.eval_split, "test_files.txt"))

        img_ext = '.png' if self.opt.png else '.jpg'
        num_train_samples = len(train_filenames_k)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs
        
        #dataloader for kitti
        train_dataset_k = self.dataset_k(
            self.opt.data_path, train_filenames_k, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext='.jpg')
        self.train_loader_k = DataLoader(
            train_dataset_k, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)

        val_dataset = self.dataset_k( 
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=False)
        self.val_iter = iter(self.val_loader)
        gt_path = os.path.join(splits_dir, self.opt.eval_split, "gt_depths.npz")
        self.gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)
        self.num_batch_k = train_dataset_k.__len__() // self.opt.batch_size
        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)#defualt=[0,1,2,3]'scales used in the loss'
            w = self.opt.width // (2 ** scale)
            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)#in layers.py
            self.backproject_depth[scale].to(self.device)
            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)
        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]
        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset_k), len(val_dataset)))
        self.save_opts()
        if self.opt.cutmix:
            self.cutmix = CutMix(beta=1.0)

    def set_train(self):
        """Convert all models to training mode
        """
        for k,m in self.models.items():
            m.train()

    def set_eval(self):
        for m in self.models.values():
            m.eval()

    def test(self):
        self.init_time = time.time()
        self.epoch_start = 0
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        self.test_epoch()
        self.total_training_time = time.time() - self.init_time
        print('====>total training time:{}'.format(sec_to_hm_str(self.total_training_time)))

    #------- Adding 
    def init_data(self,):
        H, W = self.opt.height, self.opt.width
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.homography_sampler_list = \
            [HomographySample(int(H), int(W), device=device),
             HomographySample(int(H / 2), int(W / 2), device=device),
             HomographySample(int(H / 4), int(W / 4), device=device),
             HomographySample(int(H / 8), int(W / 8), device=device)]
    
    def network_process(self, inputs, frame):
        if self.opt.cutmix:
            img = self.cutmix(inputs[('color_aug', frame, 0)])
        else:
            img = inputs[('color_aug', frame, 0)]
        B = img.size(0)
        disparity_list = get_disparity_list(self.opt, B, device=img.device) # B, S
        density_list = predict_density_from_disparity(self.disp_predictor, img, disparity_list)
        return density_list, disparity_list

    def network_process_test(self, inputs, frame):
        img = inputs[('color', frame, 0)]
        B = img.size(0)
        disparity_list = get_disparity_list(self.opt, B, device=img.device) # B, S
        density_list = predict_density_from_disparity(self.disp_predictor, img, disparity_list)
        return density_list, disparity_list

    def calculate_depth(self, inputs, scale, density_list, disparity):
        img = inputs[('color_aug', 0, 0)]
        K_scaled = inputs[("K", 0)][:, :3, :3]/ (2 ** (scale))
        K_scaled[:, 2, 2] = 1
        torch.cuda.synchronize()
        K_scaled_inv = torch.inverse(K_scaled)
        xyz_BS3HW = get_xyz_from_plane_disparity(
            self.homography_sampler_list[scale].meshgrid.to(img.device), \
                disparity.to(img.device), K_scaled_inv.to(img.device))
        depth_syn = render(density_list, xyz_BS3HW, self.opt.use_alpha)
        return depth_syn

    def disp_predictor(self, src_imgs_BCHW, disparity_BS):
        conv1_out, block1_out, block2_out, block3_out, block4_out = self.models["encoder"](src_imgs_BCHW)
        outputs = self.models["depth"]([conv1_out, block1_out, block2_out, block3_out, block4_out], disparity_BS)
        output_list = [outputs[("disp", 0)], outputs[("disp", 1)], outputs[("disp", 2)], outputs[("disp", 3)]]
        return output_list

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        print("Threads: " + str(torch.get_num_threads()))
        print("Training")
        self.set_train()
        self.every_epoch_start_time = time.time()
        for batch_idx, inputs in enumerate(self.train_loader_k):
            before_op_time = time.time()
            outputs, losses = self.process_batch(inputs)
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()
            duration = time.time() - before_op_time
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000#log_fre 's defualt = 250
            late_phase = self.step % 2000 == 0
            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)
                self.val()
            self.step += 1
            if batch_idx > 500:
                break
        self.model_lr_scheduler.step()
        self.every_epoch_end_time = time.time()
        print("====>training time of this epoch:{}".format(sec_to_hm_str(self.every_epoch_end_time-self.every_epoch_start_time)))
   
    def process_batch(self, inputs):
        for key, ipt in inputs.items():#inputs.values() has :12x3x196x640.
            inputs[key] = ipt.to(self.device)#put tensor in gpu memory
        if self.opt.pose_model_type == "shared":
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)#stacked frames processing color together
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]#? what does inputs mean?
            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]
            outputs = self.models["depth"](features[0])
        else:
            outputs = {}
            frame = 0 
            density_list, disparity_list = self.network_process(inputs, frame)
            for scale in self.opt.scales:
                outputs[("density", 0, scale)] = density_list[scale]
                outputs[("depth", frame, scale)] = self.calculate_depth(inputs, scale, density_list[scale], disparity_list)
            if self.opt.occlusion_mask:
                with torch.no_grad(): # To reduce memory request only apply to -1 one. 
                    density_list, disparity_list = self.network_process(inputs, -1) # (List of Featmaps), (B, S)
                    for scale in self.opt.scales:
                        outputs[("depth", -1, scale)] = self.calculate_depth(inputs, scale, density_list[scale], disparity_list)
        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)
        if self.use_pose_net:
            if self.train_teacher_and_pose:
                outputs.update(self.predict_poses(inputs, None))
            else: 
                with torch.no_grad():
                    outputs.update(self.predict_poses(inputs, None))
        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)
        return outputs, losses

    def predict_poses(self, inputs, features=None):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}
            for f_i in self.opt.frame_ids[1:]:
                #frame_ids = [0,-1,1]
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]#nerboring frames
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]
                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)
                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    #axisangle and translation are two 2*1*3 matrix
                    #f_i=-1,1
                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)
                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]
            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]
            axisangle, translation = self.models["pose"](pose_inputs)
            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])
        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()
        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)
            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)
            #self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses
        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            source_scale = 0
            depth = outputs[("depth", 0, scale)]
            depth = F.interpolate(depth, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]
                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":
                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]
                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)
                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)
                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)
                outputs[("sample", frame_id, scale)] = pix_coords
                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")
                if not self.opt.disable_automasking:
                    #doing this
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        losses = {}
        total_loss = 0
        for scale in self.opt.scales:
            #scales=[0,1,2,3]
            loss = 0
            reprojection_losses = []
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]
            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)
            if not self.opt.disable_automasking:
                #doing this 
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))
                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses
            elif self.opt.predictive_mask:
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)
                reprojection_losses *= mask
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda()) if torch.cuda.is_available() else   0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cpu())
                loss += weighting_loss.mean()
            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses
            if not self.opt.disable_automasking:
                if torch.cuda.is_available():
                    identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).cuda() * 0.00001
                else:
                    identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).cpu() * 0.00001
                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss
            if self.opt.occlusion_mask:
                depth_loss = self.generate_occlusion(inputs, outputs, scale)
                depth_loss = self.opt.depth_weight * depth_loss.mean()
                loss += depth_loss
            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)
            if not self.opt.disable_automasking:
                #outputs["identity_selection/{}".format(scale)] = (
                outputs["identity_selection/{}".format(0)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()
            loss += to_optimise.mean()
            disp = torch.reciprocal(outputs[("depth", 0, scale)])
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)
            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)#defualt=1e-3 something with get_smooth_loss function
            total_loss += loss
            losses["loss/{}".format(scale)] = loss
        total_loss /= self.num_scales
        if self.opt.use_var_loss:
            var_loss = self.opt.var_weight * outputs[("density", 0, scale)].var(1).mean()
            total_loss += var_loss
        losses["loss"] = total_loss 
        return losses

    def generate_occlusion(self, inputs, outputs, scale):
        depth = outputs[("depth", 0, scale)]
        tgt_depth = F.interpolate(depth, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
        T = outputs[("cam_T_cam", 0, -1)]
        depth = outputs[("depth", -1, scale)]
        src_depth = F.interpolate(depth, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
        cam_points = self.backproject(tgt_depth, torch.inverse(inputs[("K", 0)]))
        pix_src_tgt, tgt_src_depth1 = self.project(cam_points, inputs[("K", 0)], T) 
        tgt_src_depth2 = F.grid_sample(src_depth, pix_src_tgt, mode="nearest", padding_mode="border")
        tgt_src_transform = (tgt_src_depth1 - tgt_src_depth2).abs()
        variable_bar = (tgt_depth - src_depth).abs()  
        depth_loss = (tgt_src_transform/(tgt_src_depth1 + tgt_src_depth2))
        depth_loss_iden = (variable_bar/(tgt_depth + src_depth))
        mask = depth_loss < depth_loss_iden
        return depth_loss*mask

    def compute_depth_losses(self, inputs, outputs, losses):
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()
        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0
        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask
        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)
        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)
        depth_errors = compute_depth_errors(depth_gt, depth_pred)
        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch_idx {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)
        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)
                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)
                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)
                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()
        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)
        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        path = os.getcwd()
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)
        self.opt.load_weights_folder = os.path.join(path, self.opt.load_weights_folder)
        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))
        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

    def test_epoch(self):
        print("============> Validation{} <============".format(self.epoch))
        self.set_eval()
        pred_depths = []
        MIN_DEPTH = 1e-3
        MAX_DEPTH = 80
        HEIGHT, WIDTH = self.opt.height, self.opt.width
        print("-> Computing predictions with size {}x{}".format(HEIGHT, WIDTH))
        LOSS_L1 = []
        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                for key, ipt in data.items():#inputs.values() has :12x3x196x640.
                    data[key] = ipt.to(self.device)#put tensor in gpu memory
                frame = 0 
                density_list, disparity_list = self.network_process_test(data, frame)
                scale = 0
                depth = self.calculate_depth(data, scale, density_list[scale], disparity_list)
                depth = depth.cpu()[:, 0].numpy()
                pred_depths.append(depth)
        pred_depths = np.concatenate(pred_depths)
        errors = []
        ratios = []
        for i in range(pred_depths.shape[0]):
            gt_depth = self.gt_depths[i]
            gt_height, gt_width = gt_depth.shape[:2]
            pred_depth = np.squeeze(pred_depths[i])
            pred_depth = cv2.resize(pred_depth, (gt_width, gt_height))
            if self.opt.eval_split == "eigen":
                mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
                crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                                0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
                crop_mask = np.zeros(mask.shape)
                crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                mask = np.logical_and(mask, crop_mask)
            error_map = np.zeros(pred_depth.shape)
            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]
            pred_depth *= self.opt.pred_depth_scale_factor
            if not self.opt.disable_median_scaling:
                ratio = np.median(gt_depth) / np.median(pred_depth)
                ratios.append(ratio)
                pred_depth *= ratio
            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
            errors.append(compute_errors(gt_depth, pred_depth))
        mean_errors = np.array(errors).mean(0)
        print("\n  " + ("{:>8} | " * 7).format("abs_rel",
                                            "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
        print("\n-> Done!")
        self.set_train()

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def sum_params(model):
    s = []
    for p in model.parameters():
        dims = p.size()
        n = p.cpu().data.numpy()
        s.append(np.sum(n))
    return sum(s)

class CutMix:
    def __init__(self, beta):
        self.beta = beta
    def __call__(self, images):
        batch_size = images.size(0)
        indices = torch.randperm(batch_size)
        lam = np.random.beta(self.beta, self.beta)
        lam = max(lam, 1 - lam)
        image_h, image_w = images.size(2), images.size(3)
        cut_h = np.int64(image_h * lam)
        cut_w = np.int64(image_w * lam)
        y1 = np.random.randint(image_h - cut_h + 1)
        x1 = np.random.randint(image_w - cut_w + 1)
        y2 = y1 + cut_h
        x2 = x1 + cut_w
        images[:, :, y1:y2, x1:x2] = images[indices, :, y1:y2, x1:x2]
        return images