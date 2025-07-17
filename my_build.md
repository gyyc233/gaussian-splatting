- [3DGS win11 环境配置与部署验证](#3dgs-win11-环境配置与部署验证)
  - [准备项](#准备项)
  - [环境安装](#环境安装)
  - [第三方库说明](#第三方库说明)
  - [演示视频地址](#演示视频地址)

# 3DGS win11 环境配置与部署验证

## 准备项

- install git：https://gitforwindows.org/
- install conda: https://www.anaconda.com/download
- cuda toolkit 12.4: https://developer.nvidia.com/cuda-toolkit-archive
  - check for installation：`nvcc --version`
- visual studio 2022 : https://visualstudio.microsoft.com/zh-hans/downloads/
  - C++
  - vs2022添加到系统环境变量`C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.43.34808\bin\Hostx64\x64` to `path`
- install colmap sfm 计算相机位姿与点云重建 :https://github.com/colmap/colmap/releases
  - 添加到环境变量
- install imagemagick: https://imagemagick.org/script/download.php
- install ffmpeg: https://www.ffmpeg.org/download.html
- 检查系统环境变量
- 3dgs代码仓库地址：https://github.com/graphdeco-inria/gaussian-splatting
  - git clone git@github.com:graphdeco-inria/gaussian-splatting.git --recursive
  - git submodule update --init --recursive
  - 检查`diff-gaussian-rasterization`与`glm`
- 测试数据集下载 `https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip`
- 图形交互软件 `https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/binaries/viewers.zip`

## 环境安装

reference issue `https://github.com/graphdeco-inria/gaussian-splatting/issues/146`

```bash
conda create -n 3dgs python=3.10
conda activate 3dgs
# conda和vs2022联动的插件
conda install -c conda-forge vs2022_win-64
cd <path>/gaussian-splatting

git submodule update --init --recursive

# pyttorch 版本要与cuda对应，都是12.4
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

conda list torch
SET DISTUTILS_USE_SDK=1

pip install submodules\diff-gaussian-rasterization
pip install submodules\simple-knn
pip install plyfile
pip install tqdm

pip install opencv-python
pip install opencv-contrib-python
```

**train your dataset**
```bash
python train.py -s data/tandt/train

# viewer
.\viewers\bin\SIBR_gaussianViewer_app -m ./output/
```

## 第三方库说明

- diff-gaussian-rasterization

借助cuda实现了 3dgs--2dgs 光栅化和渲染的前向传播，损失函数对球谐系数渲染颜色，高斯球位置，缩放，旋转，协方差矩阵的反向优化

- fused-ssim

提供高效的结构相似性指数测量(SSIM)方法，比较两张图像的相似度

- simple-knn

寻找空间点的knn

## 演示视频地址

[3DGS搭建流程](https://www.bilibili.com/video/BV1FGVrzYErs/?share_source=copy_web&vd_source=b68bf04fb94c260d1929f4c1da59dada)
