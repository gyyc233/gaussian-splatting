from typing import Sequence

from itertools import chain

import torch
import torch.nn as nn
from torchvision import models

from .utils import normalize_activation

"""
LPIPS 感知损失模块 定义预训练模型预线性变换层模块
"""

def get_network(net_type: str):
    """
    根据 net_type 返回对应CNN特征提取器，支持 AlexNet, VGG16， squeezenet
    """
    if net_type == 'alex':
        return AlexNet()
    elif net_type == 'squeeze':
        return SqueezeNet()
    elif net_type == 'vgg':
        return VGG16()
    else:
        raise NotImplementedError('choose net_type from [alex, squeeze, vgg].')


class LinLayers(nn.ModuleList):
    """
    线性变换映射类，对每层特征图添加 1×1 卷积层，用于加权合并感知差异
    """
    # 在 LPIPS 类中，LinLayers 被用来
    # 1. 对每层特征差异进行线性加权
    # 2. 通过训练好的权重合并不同层的感知误差
    # 3. 提升感知一致性与视觉质量匹配度

    def __init__(self, n_channels_list: Sequence[int]):
        super(LinLayers, self).__init__([
            nn.Sequential(
                nn.Identity(),
                nn.Conv2d(nc, 1, 1, 1, 0, bias=False)
            ) for nc in n_channels_list
        ])

        for param in self.parameters():
            param.requires_grad = False


class BaseNet(nn.Module):
    """
    感知损失模块（LPIPS）的基类
    """
    def __init__(self):
        super(BaseNet, self).__init__()

        # register buffer
        self.register_buffer(
            'mean', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer(
            'std', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def set_requires_grad(self, state: bool):
        """
        控制是否启用梯度更新
        """
        for param in chain(self.parameters(), self.buffers()):
            param.requires_grad = state

    def z_score(self, x: torch.Tensor):
        """
        输入图像减去均值、除以标准差，确保与 CNN 预训练模型一致
        """
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor):
        """
        提取指定层的特征并归一化
        """
        x = self.z_score(x)

        output = []
        for i, (_, layer) in enumerate(self.layers._modules.items(), 1):
            x = layer(x)

            # 提取指定层的激活
            if i in self.target_layers:
                output.append(normalize_activation(x))
            if len(output) == len(self.target_layers):
                break
        return output


class SqueezeNet(BaseNet):
    """
    基于 SqueezeNet 的感知特征提取
    """
    def __init__(self):
        super(SqueezeNet, self).__init__()

        # 加载预训练 squeezenet1_1 的特征提取部分
        self.layers = models.squeezenet1_1(True).features
         # 指定需要提取的卷积层编号
        self.target_layers = [2, 5, 8, 10, 11, 12, 13]
        # 各目标层输出通道数列表，用于线性变换
        self.n_channels_list = [64, 128, 256, 384, 384, 512, 512]
        # 冻结网络参数，不更新梯度
        self.set_requires_grad(False)


class AlexNet(BaseNet):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.layers = models.alexnet(True).features
        self.target_layers = [2, 5, 8, 10, 12]
        self.n_channels_list = [64, 192, 384, 256, 256]

        self.set_requires_grad(False)


class VGG16(BaseNet):
    def __init__(self):
        super(VGG16, self).__init__()

        self.layers = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.target_layers = [4, 9, 16, 23, 30]
        self.n_channels_list = [64, 128, 256, 512, 512]

        self.set_requires_grad(False)
