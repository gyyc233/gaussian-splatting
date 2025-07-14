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
import traceback # 打印异常堆栈信息
import socket # TCP socket
import json
from scene.cameras import MiniCam # 根据接收到的相机参数创建虚拟相机对象

# 设置socket监听的IP端口
host = "127.0.0.1"
port = 6009

conn = None
addr = None

# 创建socket监听 socket.AF_INET: IPV4; socket.SOCK_STREAM: tcp 流协议
listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 基于 TCP socket 的通信接口模块

def init(wish_host, wish_port):
    """
    初始化一个TCP服务端socket
    """
    global host, port, listener
    host = wish_host
    port = wish_port
    listener.bind((host, port)) # 绑定地址与端口
    listener.listen() # 开始监听
    listener.settimeout(0) # 非阻塞模式

def try_connect():
    """
    尝试接受来自客户端的连接请求
    """
    global conn, addr, listener
    try:
        conn, addr = listener.accept() # 接收连接，得到一个新的 socket 对象 conn 和客户端地址 addr
        print(f"\nConnected by {addr}")
        conn.settimeout(None) # 将该连接设为阻塞模式（即一直等待数据到来）
    except Exception as inst:
        pass
            
def read():
    """
    从已建立的连接中读取客户端发送的消息
    """
    global conn
    messageLength = conn.recv(4) # 前 4 字节表示后续消息长度（小端序）
    messageLength = int.from_bytes(messageLength, 'little')
    message = conn.recv(messageLength)
    return json.loads(message.decode("utf-8"))

def send(message_bytes, verify):
    """
    向客户端发送图像数据和验证字符串
    """
    global conn
    # message_bytes: 图像字节流
    if message_bytes != None:
        conn.sendall(message_bytes)
    
    # 校验码
    conn.sendall(len(verify).to_bytes(4, 'little'))
    conn.sendall(bytes(verify, 'ascii'))

def receive():
    """
    从 GUI 客户端接收渲染指令和相机参数
    """
    message = read() # 读取客户端发送的json数据

    width = message["resolution_x"]
    height = message["resolution_y"]

    if width != 0 and height != 0:
        try:
            # 解析消息内容
            do_training = bool(message["train"])
            fovy = message["fov_y"]
            fovx = message["fov_x"]
            znear = message["z_near"]
            zfar = message["z_far"]
            do_shs_python = bool(message["shs_python"]) # 是否在 Python 端计算 SH 系数
            do_rot_scale_python = bool(message["rot_scale_python"])
            keep_alive = bool(message["keep_alive"])
            scaling_modifier = message["scaling_modifier"]

            # view_matrix 转为张量
            world_view_transform = torch.reshape(torch.tensor(message["view_matrix"]), (4, 4)).cuda()
            world_view_transform[:,1] = -world_view_transform[:,1]
            world_view_transform[:,2] = -world_view_transform[:,2]

            # 获取MVP矩阵并反转y轴适配渲染系统坐标系
            full_proj_transform = torch.reshape(torch.tensor(message["view_projection_matrix"]), (4, 4)).cuda()
            full_proj_transform[:,1] = -full_proj_transform[:,1]

            # 构建相机对象
            custom_cam = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)
        except Exception as e:
            print("")
            traceback.print_exc()
            raise e
        return custom_cam, do_training, do_shs_python, do_rot_scale_python, keep_alive, scaling_modifier
    else:
        return None, None, None, None, None, None