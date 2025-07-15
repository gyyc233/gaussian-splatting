#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!

    3D 高斯点渲染系统 的核心渲染函数
        用于将 3D 高斯点云（带有位置、缩放、旋转、不透明度、球谐函数颜色等属性）渲染为 2D 图像，并支持反向传播优化

    viewpoint_camera: 包含相机内参、外参、投影矩阵等信息
    pc: 3D 高斯点云模型
    pipe: 渲染管道, 控制渲染流程的参数（如是否在 Python 中计算协方差、SH 转换等）
    bg_color: 背景颜色（必须在 GPU 上）
    scaling_modifier: 缩放系数，缩放高斯点大小
    separate_sh: 是否将 DC 和 SH 特征分开处理
    override_color: 如果提供，则忽略 SH，使用指定颜色
    use_trained_exp: 是否应用训练后的曝光参数（仅训练时使用）
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # 初始化屏幕空间点，尝试保留梯度信息
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # 设置光栅化参数
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    # 创建高斯光栅化实例
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None

    if override_color is None:
        # 若启用 pipe.convert_SHs_python，则手动计算每个点的 SH 到 RGB 转换，否则使用模型自带的SH特征
        if pipe.convert_SHs_python:
            # pc.get_features 获取点云的 SH 系数
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            # 方向向量: 相机指向高斯点中心
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0) # 限制张量的最小值
        else:
            # 不在 Python 中计算颜色，而是将 SH 特征传给光栅化器，由其内部完成 SH → RGB 的转换
            if separate_sh:
                # 分别提取 DC 和 Rest SH 特征
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # 将可见的高斯进行光栅化(diff_gaussian_rasterization)，并获取其半径（在屏幕上）
    # 输入: 3d高斯点模型参数，输出: 渲染的图像，每个高斯点在屏幕上的半径，深度图
    if separate_sh:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        # 应用曝光变换
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible. 那些被视锥体剔除或者半径为0的高斯点不可见
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1) # 现在在0-1之间

    # render 最终渲染图像 [3,h,w]
    # viewspace_points 屏幕空间坐标，每个高斯点投影到屏幕空间的坐标，用于反向优化高斯点，分裂，克隆
    # visibility_filter 获取所有在屏幕上可见的高斯点索引
    # radii 每个高斯点在屏幕上占据的半径大小
    # depth 每个像素的深度

    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image
        }
    
    return out
