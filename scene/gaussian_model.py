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
import numpy as np
# inverse_sigmoid(x)：将 sigmoid 输出值还原为原始 logit 值
# get_expon_lr_func(lr_init, lr_final, max_steps)：构建指数衰减的学习率调度器
# build_rotation(scaling, rotation)：根据旋转四元数构建旋转矩阵
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn # 定义神经网络层与激活函数
import os
import json
from utils.system_utils import mkdir_p # 递归创建目录
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH # RGB2SH(color)：将 RGB 颜色转换为球谐系数（Spherical Harmonics），用于光照建模
from simple_knn._C import distCUDA2 # distCUDA2(points)：使用 CUDA 计算点与点之间的平方距离
from utils.graphics_utils import BasicPointCloud # BasicPointCloud：封装点云的基本信息（位置、颜色、法线），用于初始化高斯模型
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        """
        为高斯模型设置激活函数和协方差矩阵
        """

        # 构建高斯点的协方差矩阵
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            # R S S^T R^T
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            # 返回协方差矩阵的独立元素
            return symm
        
        # 将函数名做一次封装
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        # build_covariance_from_scaling_rotation 函数的参数通过 covariance_activation传入调用
        # 将build_covariance_from_scaling_rotation 注册为类的一个属性，后续通过 self.covariance_activation(...) 的方式调用它
        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default"):
        """
        初始化高斯模型的基本属性和函数映射
        sh_degree 球谐函数阶数
        optimizer_type 优化器类型 "default" 或 "sparse_adam"
        """
        self.active_sh_degree = 0 # 当前激活的 SH 阶数，训练过程中逐步增加
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  # 最大 SH 阶数
        self._xyz = torch.empty(0) # 高斯点的 3D 坐标
        self._features_dc = torch.empty(0) # 球谐系数的直流分量（DC component）
        self._features_rest = torch.empty(0) # 球谐系数的其余分量（Rest）
        self._scaling = torch.empty(0) # 缩放参数（log space）
        self._rotation = torch.empty(0) # 旋转四元数
        self._opacity = torch.empty(0) # 不透明度
        self.max_radii2D = torch.empty(0) # 每个点在图像空间的最大半径
        self.xyz_gradient_accum = torch.empty(0) # 梯度累计
        self.denom = torch.empty(0) # 归一化因子
        self.optimizer = None # 优化器
        self.percent_dense = 0 # 控制增密的密度比例
        self.spatial_lr_scale = 0 # 空间学习率缩放因子
        self.setup_functions()

    def capture(self):
        """
        将 GaussianModel 状态打包为一个元组 
        """
        return (
            self.active_sh_degree, # 当前球谐函数阶数
            self._xyz, # 高斯点三维坐标
            self._features_dc, # 球谐系数中的直流分量
            self._features_rest, # 球谐系数的其余部分（高阶光照信息）
            self._scaling, # 缩放参数
            self._rotation, # 旋转四元数，每个高斯点的方向
            self._opacity, # 不透明度
            self.max_radii2D, # 每个点在图像空间中的最大半径
            self.xyz_gradient_accum, # 高斯点的梯度累计
            self.denom, # 归一化因子
            self.optimizer.state_dict(), # 优化器的状态字典（如 Adam 的动量、历史梯度等）
            self.spatial_lr_scale, # 学习率缩放因子
        )
    
    def restore(self, model_args, training_args):
        """
        将之前通过 capture 方法保存的模型状态恢复
        从model_args元组中恢复 GaussianModel 状态，从training_args中恢复训练参数
        """
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args) # 重新初始化优化器（optimizer）和学习率调度器，优化器不能直接复制，必须根据当前参数重新创建
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    # @property 是一个装饰器（decorator），将类中的某个方法“伪装”成属性，可以像访问属性一样使用，不需要加括号调用

    @property
    def get_scaling(self):
        """
        将 log-space 缩放参数转换为实际缩放值
        """
        # self._scaling 是以 log 形式存储的，用 exp() 激活得到真实尺度
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        """
        对 self._rotation 应用 torch.nn.functional.normalize
        """
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        """
        返回高斯点的 3D 坐标
        """
        return self._xyz
    
    @property
    def get_features(self):
        """
        合并球谐系数的直流部分和高阶部分
        """
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        """
        获取球谐函数低频部分
        """
        return self._features_dc
    
    @property
    def get_features_rest(self):
        """
        获取球谐函数高频部分
        """
        return self._features_rest
    
    @property
    def get_opacity(self):
        """
        将不透明度从 logit 空间映射回 [0,1] 区间
        """
        # 使用 sigmoid() 函数限制输出范围
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        # 调用注册在 setup_functions 中的 build_covariance_from_scaling_rotation
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        """
        逐步提高球谐函数的阶数
        """

        # 初始时只使用低频光照（如 SH 阶数 0，仅颜色），随着训练推进，增加阶数到max_sh_degree
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        """
        通过点云数据，相机，学习率初始化高斯模型，将点云准尉可训练的高斯模型
        """
        self.spatial_lr_scale = spatial_lr_scale
        # 点云数据与点云言责转 torch.Tensor
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())

        # 创建 (N, 3, SH_coeffs) 的球谐系数张量
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color # dc
        features[:, 3:, 1:] = 0.0 # 高阶球谐系数初始为0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        # distCUDA2(...): 使用 CUDA 快速计算每个点与其最近邻点之间距离的平方
        # torch.clamp_min 设置最小值为 0.0000001，防止取对数失败
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        # torch.log(...): 将距离转换为 log 空间，每个点在 x/y/z 三个轴上使用相同的缩放值 → 各向同性初始化
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        # 创建大小为 (N, 4) 的零张量，表示 N 个点的四元数旋转 [w,x,y,z]
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        # 四元数 w=1,初始化为无旋转
        rots[:, 0] = 1

        # 设置所有点的初始不透明度，经过sigmoid映射后为0.1
        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # ==== 参数注册 ====
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True)) # 将点云坐标转为可学习参数
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True)) # 注册球谐系数直流分量
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True)) # 注册球谐系数高频部分

        # 注册缩放与旋转
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))

        # 注册高斯体不透明度
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        # 注册每个点在图像空间中出现的最大 2D 半径
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        """
        为训练准备优化器与学习率控制器
        """
        self.percent_dense = training_args.percent_dense # 点云密度控制参数

        # 累积每个点的梯度大小
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # 归一化因子
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # 参数分组与学习率设置
        # xyz	position_lr_init * spatial_lr_scale	位置学习率根据场景大小缩放
        # f_dc	feature_lr	直流球谐系数（颜色）学习率
        # f_rest	feature_lr / 20	高阶球谐系数学习率更低，防止过拟合
        # opacity	opacity_lr	不透明度学习率
        # scaling	scaling_lr	缩放学习率
        # rotation	rotation_lr	旋转四元数学习率
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        # 确定优化器，lr=0.0 表示初始学习率为 0，实际学习率由 lr_scheduler 控制
        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        # 创建位置，曝光学习率调度器（scheduler），两者都使用指数衰减学习率（get_expon_lr_func），支持延迟启动，从低到高的学习率变化
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        # 在训练过程中根据迭代步数动态更新学习率

        # 若未加载预训练曝光参数
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        # 更新点位置的学习率
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        """
        构建ply属性列表
        """
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz'] # 位置，法线（全为零）
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        
        # SH rest
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))

        # 不确定度
        l.append('opacity')

        # 缩放与旋转
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))

        return l

    def save_ply(self, path):
        """
        将高斯模型保存为ply
        """
        mkdir_p(os.path.dirname(path))

        # 提取模型参数并转为 numpy
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz) # 全零初始化
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        # 'f4' 表示每个属性是一个 32 位浮点数（float）
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        # 创建一个空的 numpy 结构化数组 elements，其大小与点数量一致
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        # 所有特征拼接成二维数组 attributes（shape (N, D)）
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        # 每行转换为 tuple 后填充进结构化数组
        elements[:] = list(map(tuple, attributes))
        # 保存为ply
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        """
        重置模型中所有高斯点的不透明度使其不超过其百分之一，防止某些点变得过于密集
        """

        # self.get_opacity: 当前经过 sigmoid 激活后的不透明度（在 [0, 1] 区间）
        # inverse_opacity_activation(...): 将其转换回 logit 空间（即 sigmoid 输入空间），便于优化器继续训练
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01)) # 限制为0.01
        # 替换优化器中不确定度张量
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        """
        从ply中读取3D高斯模型
        """
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        """
        替换优化器中的指定张量并将重置它的特征
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        """
        根据 mask 剔除优化器中的某些高斯点，并保留其余点的状态（如 Adam 的动量和方差），以支持模型动态更新
        """
        optimizable_tensors = {}

        # 遍历每个参数组
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            
            # 获取当前参数组的优化器状态（Adam 中的 exp_avg, exp_avg_sq）
            # 使用 mask 裁剪这些状态张量 → 只保留未被剔除的点的状态
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]] # 删除旧张量的优化器状态记录
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True))) # 替换为新裁剪后的张量
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0] # 恢复其动量/方差状态
            else:
                # 如果该参数组没有优化器状态（如曝光参数），就直接裁剪张量
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def prune_points(self, mask):
        """
        根据给定的 mask 剔除某些高斯点，并更新模型和优化器中的对应参数
        剔除不满足条件的高斯点（如太小、太大、不透明度太低等）
        mask 布尔张量，True 表示要剔除的点
        """
        valid_points_mask = ~mask # ~mask 表示保留下来的点
        optimizable_tensors = self._prune_optimizer(valid_points_mask) # 清理优化器

        # 更新模型参数，保证只有“有效的”高斯点被保留下来参与后续训练/渲染
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        """
        将新增的高斯点张量拼接到优化器中现有的参数组上，并保留优化器状态
        tensors_dict -> 张量字典 tensors_dict，其中包含新增点的所有参数（如 xyz、opacity、scaling 等）
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]] #在新增参数中获取对应新张量

            # 获取优化器状态，如果存在，则表示该参数已经被优化器跟踪
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                # 新加入的点没有历史动量和历史方差 → 设置为零
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        """
        将新增的高斯点（来自克隆或分裂）合并到模型中，并更新优化器状态和统计信息,用于完成增密操作
        """
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        # 拼接新旧张量并更新模型参数
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))

        # 重置梯度累积
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        """
        对梯度较大且缩放较大的高斯点进行分裂 split, 生成多个新点并更新模型与优化器状态
        """

        # 根据梯度和缩放大小选择需要分裂的点，然后通过随机采样生成多个新点，并更新模型参数
        # 是实现动态增密的核心操作之一
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda") # 全零张量
        # grads是视图空间中点的梯度（来自 rasterizer 光栅化）
        # grads 填充前 grads.shape[0] 个位置
        padded_grad[:grads.shape[0]] = grads.squeeze()

        # padded_grad >= grad_threshold 筛选梯度大于阈值的点
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        # 当前缩放大于某个基于场景范围的比例 self.percent_dense * scene_extent
        # 同时满足两个条件才会被分裂
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        # 生成新的点云数据
        stds = self.get_scaling[selected_pts_mask].repeat(N,1) # 提取选中点的缩放参数并重复 N 次

        # 在以原点为中心、缩放为标准差的正态分布中采样
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)

        # 构建旋转矩阵并重复 N 次
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)

        # 使用旋转矩阵将采样点映射到世界坐标系并加到原始点上
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)

        # 缩放缩小为原来的 1/(0.8*N)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))

        # 其它属性从原始点复制，新点继承旧点的颜色、光照、方向等信息
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        # 将新点拼接到现有张量中
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        # 原始点保留，新增点加入后统一裁剪
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))

        # prune_filter 包含原始 mask 和新增点的 False 掩码（即新增点保留）
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        """
        对梯度较大，缩放较小的高斯点进行克隆
        """
        # Extract points that satisfy the gradient condition
        # 检测高斯点的梯度与梯度阈值
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        # 检测高斯点的当前缩放小于某个基于场景范围的比例 self.percent_dense * scene_extent
        # 两个条件同时满足则进行高斯点克隆，3dgs参数也进行克隆
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        # 将新点拼接到现有张量中
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        """
        TODO: 根据梯度信息对高斯点进行增密（克隆或分裂）并剔除低质量点（如不透明度过低、过大等 
        """
        grads = self.xyz_gradient_accum / self.denom # 计算每个点的平均梯度
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii # 表示每个点在图像空间中的大小，用于后续裁剪判断

        # 执行克隆与分裂
        # 梯度大，缩放小的点->克隆
        # 梯度小，缩放大的点->分裂
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        # 构建剔除掩码
        # get_opacity < min_opacity	不透明度过低的点
        # max_radii2D > max_screen_size	在图像空间中过大的点
        # scaling.max(...) > 0.1 * extent	在世界空间中过大的点
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        # 剔除低质量高斯点
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        # 释放cuda显存
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        """
        添加增密操作后高斯点的梯度累积信息
        """
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
