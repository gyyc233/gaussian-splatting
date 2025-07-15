import torch
import torch.nn as nn

from .networks import get_network, LinLayers
from .utils import get_state_dict


class LPIPS(nn.Module):
    r"""Creates a criterion that measures
    Learned Perceptual Image Patch Similarity (LPIPS).
    LPIPS类 感知相似性损失，用于GS中图像质量评估指标

    Arguments:
        net_type (str): the network type to compare the features: 
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    def __init__(self, net_type: str = 'alex', version: str = '0.1'):

        assert version in ['0.1'], 'v0.1 is only supported now'

        super(LPIPS, self).__init__()

        # pretrained network
        # 提取预训练模型
        self.net = get_network(net_type)

        # linear layers
        # 线性映射层
        self.lin = LinLayers(self.net.n_channels_list)
        self.lin.load_state_dict(get_state_dict(net_type, version))

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        感知损失前向计算
        Arguments:
            x: 渲染图像 [B, C, H, W]
            y: 真实图像 [B, C, H, W]
            返回：感知损失值 [B, 1, 1, 1]
        """

        # 使用预训练网络提取多尺度特征（如 Conv 层激活）
        feat_x, feat_y = self.net(x), self.net(y)

        # 对每层特征图计算 (fx - fy)^2
        diff = [(fx - fy) ** 2 for fx, fy in zip(feat_x, feat_y)]
        
        # l(d) 是该层的可学习线性变换
        # mean((2, 3), keepdim=True) 表示对 H/W 取平均，保留维度
        res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin)]

        # 将所有层的加权结果相加，得到最终感知损失
        return torch.sum(torch.cat(res, 0), 0, True)
