from __future__ import absolute_import, division, print_function

import numpy as np

import torch
from scipy.spatial.transform import Rotation

# ----------- Modification ----------
class Embedder(object):
    # Positional encoding (section 5.1)
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d
        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]
        if self.kwargs["log_sampling"]:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d
        self.embed_fns = embed_fns
        self.out_dim = out_dim
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires):
    embed_kwargs = {"include_input": True,
                "input_dims": 1,
                "max_freq_log2": multires - 1,
                "num_freqs": multires,
                "log_sampling": True,
                "periodic_fns": [torch.sin, torch.cos],}
    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj):
        return eo.embed(x)
    return embed, embedder_obj.out_dim

# ---------- Modification ----------
class HomographySample:
    def __init__(self, H, W, device=None):
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device
        self.Height_tgt = H
        self.Width_tgt = W
        self.meshgrid = self.grid_generation(self.Height_tgt, self.Width_tgt, self.device)
        self.meshgrid = self.meshgrid.permute(2, 0, 1).contiguous()  # 3xHxW
        self.n = self.plane_normal_generation(self.device)
    @staticmethod
    def grid_generation(H, W, device):
        x = np.linspace(0, W-1, W)
        y = np.linspace(0, H-1, H)
        # Inversing the order due to the usage of numpy
        xv, yv = np.meshgrid(x, y)  # HxW
        xv = torch.from_numpy(xv.astype(np.float32)).to(dtype=torch.float32, device=device)
        yv = torch.from_numpy(yv.astype(np.float32)).to(dtype=torch.float32, device=device)
        ones = torch.ones_like(xv)
        meshgrid = torch.stack((xv, yv, ones), dim=2)  # HxWx3
        return meshgrid #(H,W,3); x - [0, W]; y - [0, H]
    @staticmethod
    def plane_normal_generation(device):
        n = torch.tensor([0, 0, 1], dtype=torch.float32, device=device)
        return n
    def sample(self, src_BCHW, d_src_B, G_tgt_src,
                K_src_inv, K_tgt):
            """
            Coordinate system: x, y are the image directions, z is pointing to depth direction
            :param src_BCHW: torch tensor float, 0-1, rgb/rgba. BxCxHxW
                            Assume to be at position P=[I|0]
            :param d_src_B: distance of image plane to src camera origin
            :param G_tgt_src: Bx4x4
            :param K_src_inv: Bx3x3
            :param K_tgt: Bx3x3
            :return: tgt_BCHW
            """
            # parameter processing ------ begin ------
            B, channels, Height_src, Width_src = src_BCHW.size(0), src_BCHW.size(1), src_BCHW.size(2), src_BCHW.size(3)
            R_tgt_src = G_tgt_src[:, 0:3, 0:3]
            t_tgt_src = G_tgt_src[:, 0:3, 3]
            Height_tgt = self.Height_tgt
            Width_tgt = self.Width_tgt
            R_tgt_src = R_tgt_src.to(device=src_BCHW.device)
            t_tgt_src = t_tgt_src.to(device=src_BCHW.device)
            K_src_inv = K_src_inv.to(device=src_BCHW.device)
            K_tgt = K_tgt.to(device=src_BCHW.device)
            # the goal is compute H_src_tgt, that maps a tgt pixel to src pixel
            # so we compute H_tgt_src first, and then inverse
            n = self.n.to(device=src_BCHW.device)
            n = n.unsqueeze(0).repeat(B, 1)  # Bx3
            # Bx3x3 - (Bx3x1 * Bx1x3)
            d_src_B33 = d_src_B.reshape(B, 1, 1).repeat(1, 3, 3)  # B -> Bx3x3
            R_tnd = R_tgt_src - torch.matmul(t_tgt_src.unsqueeze(2), n.unsqueeze(1)) / -d_src_B33
            H_tgt_src = torch.matmul(K_tgt,
                                    torch.matmul(R_tnd, K_src_inv))
            # From source to Target
            with torch.no_grad():
                H_src_tgt = inverse(H_tgt_src)
            # create tgt image grid, and map to src
            meshgrid_tgt_homo = self.meshgrid.to(src_BCHW.device)
            # 3xHxW -> Bx3xHxW
            meshgrid_tgt_homo = meshgrid_tgt_homo.unsqueeze(0).expand(B, 3, Height_tgt, Width_tgt)
            # wrap meshgrid_tgt_homo to meshgrid_src
            meshgrid_tgt_homo_B3N = meshgrid_tgt_homo.view(B, 3, -1)  # Bx3xHW
            meshgrid_src_homo_B3N = torch.matmul(H_src_tgt, meshgrid_tgt_homo_B3N)  # Bx3x3 * Bx3xHW -> Bx3xHW
            # Bx3xHW -> Bx3xHxW -> BxHxWx3
            meshgrid_src_homo = meshgrid_src_homo_B3N.view(B, 3, Height_tgt, Width_tgt).permute(0, 2, 3, 1)
            meshgrid_src = meshgrid_src_homo[:, :, :, 0:2] / meshgrid_src_homo[:, :, :, 2:]  # BxHxWx2
            np_meshgrid_src = meshgrid_src.cpu().detach().numpy()
            valid_mask_x = np.logical_and(np_meshgrid_src[:, :, :, 0] < Width_src, np_meshgrid_src[:, :, :, 0] > -1)
            valid_mask_y = np.logical_and(np_meshgrid_src[:, :, :, 1] < Height_src, np_meshgrid_src[:, :, :, 1] > -1)
            valid_mask = np.logical_and(valid_mask_x, valid_mask_y)  # BxHxW
            valid_mask = torch.tensor(valid_mask).to(src_BCHW.device)
            # sample from src_BCHW
            # normalize meshgrid_src to [-1,1]
            meshgrid_src[:, :, :, 0] = (meshgrid_src[:, :, :, 0]+0.5) / (Width_src * 0.5) - 1
            meshgrid_src[:, :, :, 1] = (meshgrid_src[:, :, :, 1]+0.5) / (Height_src * 0.5) - 1
            tgt_BCHW = torch.nn.functional.grid_sample(src_BCHW, grid=meshgrid_src, padding_mode='border',
                                                    align_corners=False)
            # BxCxHxW, BxHxW
            return tgt_BCHW, valid_mask

def inverse(matrices):
    inverse = None
    max_tries = 5
    while (inverse is None) or (torch.isnan(inverse)).any():
        torch.cuda.synchronize()
        inverse = torch.inverse(matrices)
        # Break out of the loop when the inverse is successful or there"re no more tries
        max_tries -= 1
        if max_tries == 0:
            break
    # Raise an Exception if the inverse contains nan
    if (torch.isnan(inverse)).any():
        raise Exception("Matrix inverse contains nan!")
    return inverse

def get_disparity_list(opt, B, device):
    S_coarse = opt.num_bins
    disparity_start, disparity_end = opt.disparity_start, opt.disparity_end
    if not opt.uniform_disparity:
        disparity_coarse = torch.linspace(
            disparity_start, disparity_end, S_coarse, dtype=torch.float32,
            device=device).unsqueeze(0).repeat(B, 1)
        return disparity_coarse # B, S
    elif opt.uniform_disparity:
        disparity_coarse = uniformly_sample_disparity_from_linspace_bins(
            batch_size=B, num_bins=S_coarse,
            start=disparity_start,
            end=disparity_end, device=device)
        return disparity_coarse # B, S

def uniformly_sample_disparity_from_linspace_bins(batch_size, num_bins, start, end, device):
    assert start > end
    B, S = batch_size, num_bins
    bin_edges = torch.linspace(start, end, num_bins+1, dtype=torch.float32, device=device)  # S+1
    interval = bin_edges[1] - bin_edges[0]  # scalar
    bin_edges_start = bin_edges[0:-1].unsqueeze(0).repeat(B, 1)  # S -> BxS
    random_float = torch.rand((B, S), dtype=torch.float32, device=device) # BxS
    disparity_array = bin_edges_start + interval * random_float
    return disparity_array  # BxS

'''----------------Transformation for pose matrix'''
def transformation_from_parameters(axisangle, translation, invert=False):
    R = rot_from_axisangle(axisangle)
    t = translation.clone()
    if invert:
        R = R.transpose(1, 2)
        t *= -1
    T = get_translation_matrix(t)
    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)
    return M

def get_translation_matrix(translation_vector):
    T = torch.zeros(translation_vector.shape[0], 4, 4).cuda()
    t = translation_vector.contiguous().view(-1, 3, 1)
    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t
    return T

def rot_from_axisangle(vec):
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)
    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca
    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)
    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC
    rot = torch.zeros((vec.shape[0], 4, 4)).cuda()
    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1
    return rot