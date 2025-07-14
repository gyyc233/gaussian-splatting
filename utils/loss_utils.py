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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass

C1 = 0.01 ** 2
C2 = 0.03 ** 2

# 对 torch.autograd.Function 自动求导函数类的拓展
class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        """
        前向传播
        """

        # 使用 fusedssim 函数（CUDA 实现）计算两张图像的 SSIM map
        ssim_map = fusedssim(C1, C2, img1, img2)
        # 保存输入图像用于后续梯度计算
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        # ssim_map 包含每个像素点的SSIM值
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        """
        SSIM 的梯度反向传播
        """
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None

def l1_loss(network_output, gt):
    """
    计算两个张量之间的 L1 损失 MAE
    """

    # 返回一个标量，表示像素级差值的绝对值的平均值
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    """
    计算两个张量之间的 L2 损失 MSE
    """

    # 返回一个标量，表示像素级差值平方的平均值
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    """
    生成一个 一维高斯窗口
    window_size: 卷积核大小
    sigma: 高斯分布的标准差
    """
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])

    # 归一化后的一维高斯权重向量
    return gauss / gauss.sum()

def create_window(window_size, channel):
    """
    根据一维高斯窗口构建 二维卷积核，用于 SSIM 计算
    window_size: 卷积核大小
    channel: 输入图像的通道数
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1) # 创建一维高斯权重向量并增加一个维度变为 [window_size, 1]

    # _1D_window.mm(_1D_window.t()) 矩阵相乘得到二维高斯核，然后增加维度到4
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)

    # 将二维核复制到每个通道上 contiguous()：确保内存连续
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())

    # 返回形状为 [channel, 1, window_size, window_size] 的张量
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    """
    计算两张图像之间的结构相似性
    img1: 输入图像1
    img2: 输入图像2
    window_size: 卷积核大小
    size_average: 是否返回所有像素点的平均 SSIM 值
    """
    channel = img1.size(-3)
    window = create_window(window_size, channel) # 生成二维高斯核

    # 如果 img1 在 GPU 上，则将高斯窗口也移动到相同设备上，保证后续卷积运算兼容
    if img1.is_cuda:
        window = window.cuda(img1.get_device())

    # 使高斯窗口的数据类型（如 float32）与输入图像保持一致
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """
    两个图像 SSIM 结构衡量指标计算
    SSIM 主要考虑亮度，对比度，结构 三种相似性
    """

    # 图像像素平均亮度
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel) 
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # 对比度通过灰度标准差衡量
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq

    # 结构性通过相关性系数衡量
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    # 经验性常数
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()


