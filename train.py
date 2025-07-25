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

import os # 操作系统相关的操作，如路径处理
import torch # 用于张量计算和模型训练
from random import randint # 随机整数
from utils.loss_utils import l1_loss, ssim # L1 损失函数和结构相似性指标（SSIM），用于衡量图像重建质量
from gaussian_renderer import render, network_gui # 渲染函数和 GUI 界面交互模块，用于可视化训练结果
import sys # 提供对 Python 解释器相关操作的支持，比如退出程序
from scene import Scene, GaussianModel # 表示场景和高斯模型的类，用于构建 3D 场景并管理高斯点
from utils.general_utils import safe_state, get_expon_lr_func # 安全状态和学习率
import uuid # 生成唯一标识符，通常用于创建唯一的模型保存路径
from tqdm import tqdm # 显示进度条，方便观察训练过程
from utils.image_utils import psnr # 峰值信噪比（PSNR）评估图像质量
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams # 封装模型、渲染管线和优化器参数的类

# 尝试导入 SummaryWriter 用于训练日志记录
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# 尝试导入 fused_ssim 来加速 SSIM 计算
try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

# 损失函数 SparseGaussianAdam
# SPARSE_ADAM_AVAILABLE决定了separate_sh，如果能够使用稀疏高斯优化器，则会将sh分离
# 只有在稀疏加速器可用时，才会启用 SparseGaussianAdam 优化器，否则会用普通的优化器
# sh分离得到两个组件，一个是color，后面传递给C++底层时叫colors_precomp；另一个是sh，就是球谐函数的其他组件（非直流量）
# colors_precomp：每个高斯点的直接颜色值（RGB），不再用球谐动态计算，适合静态颜色渲染或加速推理
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True # 默认没有 SparseGaussianAdam
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    """
    3dgs 模型训练
    dataset: ModelParams
    opt: OptimizationParams
    pipe: PipelineParams
    testing_iterations: test_iterations
    saving_iterations: save_iterations
    checkpoint_iterations: checkpoint_iterations
    checkpoint: checkpoint_path
    debug_from: debug_from
    """

    # if not condition 当 condition 为假时执行
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    
    # 创建 gaussians model 与 训练集 scene 实例
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)

    # 初始化优化器（如 Adam 或 SparseAdam）、学习率调度器
    gaussians.training_setup(opt)

    # 加载检查点，默认检查点文件为空
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # 根据数据集配置选择白色或黑色背景
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    # 将背景颜色转换为 PyTorch 张量并放到 GPU 上
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 用于测量每次迭代的时间
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # 只有当优化器类型是 "sparse_adam" 且系统支持时才启用稀疏优化
    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 

    # 使用指数衰减方式控制深度图的 L1 损失权重，随着训练逐步降低正则化强度
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    # 获取所有训练用的相机视角，并保存它们的索引
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    # 记录损失值
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    # 创建训练进度条，显示当前训练进度
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # 运行训练时，会构建GaussianModel和Scene对象，一个放高斯模型，另一个存训练数据，训练时，会进行迭代目标轮数（默认30000）
    for iteration in range(first_iter, opt.iterations + 1):
        # network_gui 不影响渲染主流程
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             # 获取渲染后的图像，内部使用的是diff_gaussian_rasterization子模块，执行gaussian渲染（保留梯度信息用于反向传播）
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]

        #             # torch.clamp(..., min=0, max=1.0)* 255).byte() 图像像素值限制在 [0, 1] 转为标准图像，再将浮点性张量转为8位整数
        #             # .permute(1, 2, 0).contiguous().cpu().numpy()改变维度顺序从[C,H,W]→[H,W,C],确保内存连续，再复制回cpu转为numpy数组
        #             # memoryview() 内存视图对象，可以高效传递给图像显示库（如 PyQt、OpenCV）
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())

        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record() # 记录本次迭代开始时间戳

        # 根据迭代步数动态更新学习率
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # 每迭代1000次就将sh阶数逐步提到最大
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        # 从训练视角列表中随机选择一个相机视角（viewpoint）用于训练或渲染
        # 如果 viewpoint_stack 为空（第一次运行或一轮遍历完成），则重新填充它
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))

        rand_idx = randint(0, len(viewpoint_indices) - 1) # 从剩余视角中随机选择一个索引
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # 基于这个训练视角viewpoint_cam进行渲染，内部完成3d->2d泼溅和体渲染过程
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)

        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # 如果当前视角有 alpha_mask（如前景/背景分割掩码），则应用到渲染结果上
        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss 
        # 训练渲染完以后，计算loss并反向传播
        gt_image = viewpoint_cam.original_image.cuda() # 当前视角真实图像转到gpu
        Ll1 = l1_loss(image, gt_image) # 计算训练视角与真实图像的l1 loss
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        # loss是由两部分组成，L1 loss和SSIM
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization 深度正则化
        # depth_l1_weight(iteration) 根据当前迭代次数返回深度损失权重（可能随着训练过程衰减或增长）
        # viewpoint_cam.depth_reliable 表示该视角的深度图是否可靠,不可靠跳过深度损失计算
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"] # 模型渲染的深度图
            mono_invdepth = viewpoint_cam.invdepthmap.cuda() # 相机视角的深度图
            depth_mask = viewpoint_cam.depth_mask.cuda() # 深度图中有效区域的掩码

            # 计算深度损失，屏蔽不可靠区域，最终去均值得到深度损失
            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            # 将深度损失加权后加入总损失函数中
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth

            Ll1depth = Ll1depth.item() # 将 tensor 转换为 Python float 保存
        else:
            Ll1depth = 0

        # 参数反向传播
        # TODO: 这里的反向传播调用 Differential Gaussian Rasterization init.py 中的 backward() 静态函数
        loss.backward()

        iter_end.record() # 记录本次迭代结束时间戳

        # 高斯点优化与密度自适应
        # with 是将一些语句打包（高斯自适应密度调整等），在执行这些流程时，不执行梯度计算
        with torch.no_grad():
            # Progress bar
            # 使用 指数移动平均（EMA） 来平滑损失值，避免波动过大
            # loss.item() 是当前迭代的总损失（包括 L1 + SSIM）
            # Ll1depth 是本次深度图损失
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save 
            # training_report 训练评估模块
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)

            # 如果当前迭代在 saving_iterations 列表中，则保存高斯模型快照
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification 点云密度自适应调整
            # iteration < opt.densify_until_iter 表示当前迭代仍在允许 点云动态调整 的阶段
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                # 跟踪每个可见高斯点在屏幕中曾达到的最大半径
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])

                # 添加屏幕上可见的高斯点
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # 
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    # 根据不透明度，尺寸大小，屏幕点半径，判断要对该点进行克隆或者分裂
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                # 定期重置高斯点不透明度
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                # 曝光参数优化器更新（如果启用）
                gaussians.exposure_optimizer.step()
                # 清空梯度缓存
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)

                # 若启用的稀疏优化器
                if use_sparse_adam:
                    # 仅对可见点进行优化
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    # 否则所有高斯点同一更新
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            # 检查点保存
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):
    """
    创建模型保存路径，并初始化 TensorBoard 日志记录器
    """
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    """
    训练评估模块
    tb_writer: TensorBoard 写入器
    iteration: 当前迭代次数
    Ll1: L1损失
    loss: 总损失(L1 + SSIM + 深度损失)
    l1_loss: L1 损失函数
    elapsed: 当前迭代耗时
    testing_iterations: 需要执行验证的迭代次数列表
    scene: 场景对象，包含高斯点云和相机视角
    renderFunc: 渲染函数
    renderArgs: 渲染函数参数
    train_test_exp: 是否使用训练后的曝光参数
    """

    # 写入训练损失到 TensorBoard
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache() # 清空gpu缓存
        # scene.getTestCameras()：获取所有测试视角
        # scene.getTrainCameras()：获取训练视角，这里只取部分（每隔 5 个取一个）
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        # 对每个视角进行渲染与评估
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):

                    # renderFunc(...)：调用 render(...) 函数进行图像渲染
                    # torch.clamp(..., 0.0, 1.0)：确保图像像素值在合法范围内
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    
                    if train_test_exp:
                        # 这种情况只取图像的一半
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]

                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    # 计算l1损失与峰值信噪比
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                # 取平均
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))

                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            # 记录点与不确定度分与与高斯点总数
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")

    lp = ModelParams(parser) # 模型参数
    op = OptimizationParams(parser) # 优化器参数
    pp = PipelineParams(parser) # 渲染管线参数

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    # 指定在哪些迭代次数进行测试评估/保存高斯模型，默认在第 7000 和 30000 次时触发
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)

    # 如果设置了 --detect_anomaly 参数，PyTorch 会开启自动梯度异常检测
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # 训练主函数
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
