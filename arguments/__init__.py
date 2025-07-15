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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    """
    通用参数基类，被多个子类继承（如 ModelParams, PipelineParams, OptimizationParams）
    """
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            # 所有以 _ 开头的字段都会被注册为命令行参数，并支持短格式
            # -s 或 --source_path 对应 _source_path
            shorthand = False
            # 如果属性名以下划线 _ 开头，则将其视为缩写
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    """
    3DGS模型参数类
    """
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3 # 球谐函数最大阶数
        self._source_path = "" # 数据集根目录
        self._model_path = "" # 模型输出目录
        self._images = "images" # 输入图像目录
        self._depths = "" # 深度图目录
        self._resolution = -1 # 图像分辨率
        self._white_background = False # 是否使用白色背景
        self.train_test_exp = False # 是否训练或测试曝光参数
        self.data_device = "cuda" # 数据设备
        self.eval = False # 是否进入评估模式
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        # 将 source_path 转换为绝对路径，防止相对路径问题
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    """
    渲染管线配置参数
    """
    def __init__(self, parser):
        self.convert_SHs_python = False # 是否在 Python 中手动计算 RGB 到 SH系数 的转换（否则由光栅化器处理）
        self.compute_cov3D_python = False # 是否在 Python 中预计算 3D 协方差（否则由光栅化器内部处理）
        self.debug = False # 是否启用调试模式
        self.antialiasing = False # 是否启用抗锯齿功能（提升图像质量）
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    """
    3DGS训练优化参数
    """
    def __init__(self, parser):
        self.iterations = 30_000 # 总迭代次数
        self.position_lr_init = 0.00016 # 高斯球位置学习率初始值
        self.position_lr_final = 0.0000016 # 高斯球位置学习率最终值
        self.position_lr_delay_mult = 0.01 # 高斯球位置学习率延迟系数
        self.position_lr_max_steps = 30_000 # 位置学习率衰减步数上限
        self.feature_lr = 0.0025 # SH特征学习率
        self.opacity_lr = 0.025 # 不透明度学习率
        self.scaling_lr = 0.005 # 缩放学习率
        self.rotation_lr = 0.001 # 旋转学习率
        self.exposure_lr_init = 0.01 # 曝光学习率初始值
        self.exposure_lr_final = 0.001 # 曝光学习率最终值
        self.exposure_lr_delay_steps = 0 # 曝光学习率延迟开始步数
        self.exposure_lr_delay_mult = 0.0 # 曝光学习率延迟系数
        self.percent_dense = 0.01 # 控制稀疏 Adam 中“密集”区域比例
        self.lambda_dssim = 0.2 # SSIM 损失在总损失中的权重
        self.densification_interval = 100 # 多少次迭代后执行一次点云稠密化
        self.opacity_reset_interval = 3000 # 多少次迭代后重置不透明度
        self.densify_from_iter = 500 # 从第几次迭代开始稠密化
        self.densify_until_iter = 15_000 # 到第几次迭代为止稠密化
        self.densify_grad_threshold = 0.0002 # 分裂高斯点的梯度阈值
        self.depth_l1_weight_init = 1.0 # 深度 L1 损失初始权重
        self.depth_l1_weight_final = 0.01 # 深度 L1 损失最终权重
        self.random_background = False # 是否使用随机背景色进行训练
        self.optimizer_type = "default" # 使用的优化器类型
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    """
    断点续训和参数继承
    """
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
