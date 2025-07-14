import torch

def mse(img1, img2):
    """
    计算两张图像之间的 均方误差 MSE
    img1, img2: 输入图像张量，形状为 [B, C, H, W] [Batch, Channel, Height, Width]
    """
    # .view(img1.shape[0], -1)：将图像展平成二维张量，形状变为 [B, C*H*W]
    # .mean(1, keepdim=True)：对每个 batch 的所有像素取平均，得到一个形状为 [B, 1] 的张量
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    """
    计算两张图像之间的 峰值信噪比 PSNR, PSNR 是衡量图像重建质量的重要指标，单位是 dB（分贝），数值越高代表图像质量越好
    """
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

    # 单位 db
    return 20 * torch.log10(1.0 / torch.sqrt(mse))