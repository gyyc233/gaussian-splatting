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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch
import cv2

"""
定义相机模型，计算试图变换，投影变换
"""

class Camera(nn.Module):
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image, invdepthmap,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, is_test_view = False
                 ):
        """
        Args:
            resolution (tuple): image resolution
            colmap_id (int):  COLMAP 数据集中对应相机的 ID
            R (torch.Tensor): camera rotation matrix (base world frame)
            T (torch.Tensor): translation vector
            FoVx (float): 相机在水平方向 (FoVx) 的视场角
            FoVy (float): 相机在垂直方向 (FoVy) 的视场角
            depth_params: 包含深度图相关的参数，如缩放因子、偏移等
            image (PIL.Image): 输入的图像
            invdepthmap (np.ndarray): 表示像素到相机的距离
            image_name (str): 图像文件名
            uid: 唯一标识符，用于区分不同的相机实例
        """
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        # 图像转张量并提取颜色
        resized_image_rgb = PILtoTorch(image, resolution)
        gt_image = resized_image_rgb[:3, ...]

        # 如果图像是 RGBA 格式，则提取 Alpha 通道；否则创建一个全 1 的掩码作为默认 Alpha
        self.alpha_mask = None
        if resized_image_rgb.shape[0] == 4:
            self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
        else: 
            self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))

        # 根据训练/测试标志修改 Alpha 掩码，模拟部分遮挡效果
        if train_test_exp and is_test_view:
            if is_test_dataset:
                self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
            else:
                self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0

        # 图像归一化并记录尺寸
        self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        # 逆深度处理
        self.invdepthmap = None
        self.depth_reliable = False
        if invdepthmap is not None:
            self.depth_mask = torch.ones_like(self.alpha_mask)
            self.invdepthmap = cv2.resize(invdepthmap, resolution)
            self.invdepthmap[self.invdepthmap < 0] = 0
            self.depth_reliable = True

            if depth_params is not None:
                if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                    self.depth_reliable = False
                    self.depth_mask *= 0
                
                if depth_params["scale"] > 0:
                    self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]

            if self.invdepthmap.ndim != 2:
                self.invdepthmap = self.invdepthmap[..., 0]
            self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)

        # 远近裁剪平面设置
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        # 试图变换
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        # 投影变换
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()

        # 合成完整的投影变换
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)

        # 相机在世界坐标系中的位置
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        """
        Camera model
        Args:
            width (int): image width
            height (int): image height
            fovy (float): 相机在垂直方向 (FoVy) 的视场角
            fovx (float): 相机在水平方向 (FoVx) 的视场角
            znear (float): near clipping plane 定义相机的近裁剪平面
            zfar (float): far clipping plane 定义相机的远裁剪平面
            world_view_transform (torch.Tensor): world to view transform 世界到视图的观测变换
            full_proj_transform (torch.Tensor): full projection transform 投影变换矩阵
        """
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform) # 相机到世界
        self.camera_center = view_inv[3][:3] # 相机的世界坐标

