import torch
from .utils import HomographySample

# ---------- Modification ----------
def get_xyz_from_plane_disparity(meshgrid_homo, mpi_disparity, K_inv):
    """:param meshgrid_homo: 3xHxW :param mpi_disparity: BxS :param K_inv: Bx3x3 :return: """
    B, S = mpi_disparity.size()
    H, W = meshgrid_homo.size(1), meshgrid_homo.size(2)
    mpi_depth = torch.reciprocal(mpi_disparity)  # BxS from the disparity to depth
    K_inv_Bs33 = K_inv.unsqueeze(1).repeat(1, S, 1, 1).reshape(B * S, 3, 3)
    meshgrid_homo = meshgrid_homo.unsqueeze(0).unsqueeze(1).repeat(B, S, 1, 1, 1) # 3xHxW -> BxSx3xHxW
    meshgrid_homo_Bs3N = meshgrid_homo.reshape(B * S, 3, -1)
    xyz = torch.matmul(K_inv_Bs33, meshgrid_homo_Bs3N)  # BSx3xHW
    xyz = xyz.reshape(B, S, 3, H * W) * mpi_depth.unsqueeze(2).unsqueeze(3)  # BxSx3xHW
    xyz_BS3HW = xyz.reshape(B, S, 3, H, W)
    return xyz_BS3HW

def predict_density_from_disparity(disp_predictor, imgs, disparity_coarse):
    density_list = disp_predictor(imgs, disparity_coarse)  # BxS_coarsex1xHxW
    return density_list

def render(sigma_BS1HW, xyz_BS3HW, use_alpha):
    if not use_alpha:
        depth_syn = plane_volume_rendering(sigma_BS1HW, xyz_BS3HW)
    else:
        depth_syn = alpha_composition(sigma_BS1HW, xyz_BS3HW[:, :, 2:])
    return depth_syn

def alpha_composition(alpha_BK1HW, value_BKCHW):
    B, K, _, H, W = alpha_BK1HW.size()
    alpha_comp_cumprod = torch.cumprod(1 - alpha_BK1HW, dim=1)  # BxKx1xHxW
    preserve_ratio = torch.cat((torch.ones((B, 1, 1, H, W), dtype=alpha_BK1HW.dtype, device=alpha_BK1HW.device),
                                alpha_comp_cumprod[:, 0:K-1, :, :, :]), dim=1)  # BxKx1xHxW
    weights = alpha_BK1HW * preserve_ratio  # BxKx1xHxW
    value_composed = torch.sum(value_BKCHW * weights, dim=1, keepdim=False)  # Bx3xHxW
    return value_composed
    
def plane_volume_rendering(sigma_BS1HW, xyz_BS3HW):
    B, _, _, H, W = sigma_BS1HW.size()
    xyz_diff_BS3HW = xyz_BS3HW[:, 1:, :, :, :] - xyz_BS3HW[:, 0:-1, :, :, :]  # Bx(S-1)x3xHxW
    xyz_dist_BS1HW = torch.norm(xyz_diff_BS3HW, dim=2, keepdim=True)  # Bx(S-1)x1xHxW
    xyz_dist_BS1HW = torch.cat((xyz_dist_BS1HW, torch.full((B, 1, 1, H, W), fill_value=1e3,
                                dtype=xyz_BS3HW.dtype, device=xyz_BS3HW.device)), dim=1)  # BxSx3xHxW
    transparency = torch.exp(-sigma_BS1HW * xyz_dist_BS1HW)  # BxSx1xHxW
    alpha = 1 - transparency # BxSx1xHxW
    # pytorch.cumprod is like: [a, b, c] -> [a, a*b, a*b*c], we need to modify it to [1, a, a*b]
    transparency_acc = torch.cumprod(transparency + 1e-6, dim=1)  # Bx(S-1)x1xHxW
    transparency_acc = torch.cat((torch.ones((B, 1, 1, H, W), dtype=transparency.dtype, device=transparency.device),
                                  transparency_acc[:, 0:-1, :, :, :]), dim=1)  # BxSx1xHxW
    weights = transparency_acc * alpha  # BxSx1xHxW
    depth_out = weighted_sum_disp(xyz_BS3HW, weights) # Bxs
    return depth_out

def weighted_sum_disp(xyz_BS3HW, weights):
    # Weights BxSx1xHxW
    weights_sum = torch.sum(weights, dim=1, keepdim=False)  # Bx1xHxW
    depth_out = torch.sum(weights * xyz_BS3HW[:, :, 2:, :, :], dim=1, keepdim=False)/(weights_sum + 1e-5)  # Bx1xHxW
    return depth_out

def inverse_matrix(RT):
    # RT Bx4x4
    RT_inv = torch.eye(4,4)[None].repeat(RT.size(0),1,1)
    R = RT[:, :3, :3] # B, 3, 3
    T = RT[:, :3, 3:] # B, 3, 1
    R_inv = torch.linalg.inv(R) # B, 3, 3
    T_inv = - torch.matmul(R_inv, T) # B, 3, 1
    RT_inv[:, :3, :3] = R_inv
    RT_inv[:, :3, 3:] = T_inv
    return RT_inv # B, 3, 3 

def get_tgt_xyz_from_plane_disparity(xyz_src_BS3HW, G_tgt_src):
    """:param xyz_src_BS3HW: BxSx3xHxW
       :param G_tgt_src: Bx4x4 """
    B, S, _, H, W = xyz_src_BS3HW.size()
    G_tgt_src_Bs33 = G_tgt_src.unsqueeze(1).repeat(1, S, 1, 1).reshape(B*S, 4, 4)
    xyz_tgt = transform_G_xyz(G_tgt_src_Bs33, xyz_src_BS3HW.reshape(B*S, 3, H*W))  # Bsx3xHW
    xyz_tgt_BS3HW = xyz_tgt.reshape(B, S, 3, H, W)  # BxSx3xHxW
    return xyz_tgt_BS3HW

def transform_G_xyz(G, xyz, is_return_homo=False):
    """:param G: Bx4x4
       :param xyz: Bx3xN"""
    assert len(G.size()) == len(xyz.size())
    if len(G.size()) == 2:
        G_B44 = G.unsqueeze(0)
        xyz_B3N = xyz.unsqueeze(0)
    else:
        G_B44 = G
        xyz_B3N = xyz
    xyz_B4N = torch.cat((xyz_B3N, torch.ones_like(xyz_B3N[:, 0:1, :])), dim=1)
    G_xyz_B4N = torch.matmul(G_B44, xyz_B4N)
    if is_return_homo:
        return G_xyz_B4N
    else:
        return G_xyz_B4N[:, 0:3, :]

def render_tgt_depth(H_sampler: HomographySample,
                         mpi_sigma_src, mpi_disparity_src,
                         xyz_tgt_BS3HW, G_tgt_src,
                         K_src_inv, K_tgt, use_alpha=False,
                         is_bg_depth_inf=False):
    """:param H_sampler:
       :param mpi_sigma_src: BxSx1xHxW
       :param mpi_disparity_src: BxS
       :param xyz_tgt_BS3HW: BxSx3xHxW
       :param G_tgt_src: Bx4x4
       :param K_src_inv: Bx3x3
       :param K_tgt: Bx3x3 """
    B, S, _, H, W = mpi_sigma_src.size()
    mpi_depth_src = torch.reciprocal(mpi_disparity_src)  # BxS
    # note that here we concat the mpi_src with xyz_tgt, because H_sampler will sample them for tgt frame
    # mpi_src is the same in whatever frame, but xyz has to be in tgt frame
    mpi_xyz_src = torch.cat((mpi_sigma_src, xyz_tgt_BS3HW), dim=2)  # BxSx(3+1+3)xHxW
    # homography warping of mpi_src into tgt frame
    G_tgt_src_Bs44 = G_tgt_src.unsqueeze(1).repeat(1, S, 1, 1).contiguous().reshape(B*S, 4, 4)  # Bsx4x4
    K_src_inv_Bs33 = K_src_inv.unsqueeze(1).repeat(1, S, 1, 1).contiguous().reshape(B*S, 3, 3)  # Bsx3x3
    K_tgt_Bs33 = K_tgt.unsqueeze(1).repeat(1, S, 1, 1).contiguous().reshape(B*S, 3, 3)  # Bsx3x3
    # BsxCxHxW, BsxHxW
    tgt_mpi_xyz_BsCHW, tgt_mask_BsHW = H_sampler.sample(mpi_xyz_src.view(B*S, 4, H, W),
                                                        mpi_depth_src.view(B*S),
                                                        G_tgt_src_Bs44,
                                                        K_src_inv_Bs33,
                                                        K_tgt_Bs33)
    # mpi composition
    tgt_mpi_xyz = tgt_mpi_xyz_BsCHW.view(B, S, 4, H, W)
    tgt_sigma_BS1HW = tgt_mpi_xyz[:, :, 0:1, :, :]
    tgt_xyz_BS3HW = tgt_mpi_xyz[:, :, 1:, :, :]
    tgt_mask_BSHW = tgt_mask_BsHW.view(B, S, H, W)
    tgt_mask_BSHW = torch.where(tgt_mask_BSHW,
                                torch.ones((B, S, H, W), dtype=torch.float32, device=mpi_sigma_src.device),
                                torch.zeros((B, S, H, W), dtype=torch.float32, device=mpi_sigma_src.device))
    # Bx3xHxW, Bx1xHxW, Bx1xHxW
    tgt_z_BS1HW = tgt_xyz_BS3HW[:, :, -1:]
    tgt_sigma_BS1HW = torch.where(tgt_z_BS1HW >= 0,
                                  tgt_sigma_BS1HW,
                                  torch.zeros_like(tgt_sigma_BS1HW, device=tgt_sigma_BS1HW.device))
    tgt_depth_syn = render(tgt_sigma_BS1HW, tgt_xyz_BS3HW,
                                              use_alpha=use_alpha)
    tgt_mask = torch.sum(tgt_mask_BSHW, dim=1, keepdim=True)  # Bx1xHxW
    return tgt_depth_syn, tgt_mask