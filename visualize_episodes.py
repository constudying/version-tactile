
"""
该文件是一个工具脚本，用于对机器人仿真生成的数据进行可视化和分析，包括：
加载 HDF5 数据集：读取仿真任务中的状态、动作和图像。
保存视频：将仿真图像序列保存为 MP4 格式视频。
绘制状态和动作图：可视化机器人关节状态和控制命令。
分析时间戳：检查仿真或数据采集中的时间序列数据。

文件总结
    核心功能：
        提供了加载、可视化、分析机器人仿真数据的工具函数。
        适用于 HDF5 格式的机器人仿真任务数据。
    模块结构：
        数据加载：load_hdf5。
        视频保存：save_videos。
        数据可视化：visualize_joints 和 visualize_timestamp。
        主逻辑：main。
    适用场景：
        (dt)检查仿真数据质量。
        可视化机器人关节状态和命令。
        生成任务视频演示。
"""


import os
import numpy as np
import cv2
import h5py
import argparse

import matplotlib.pyplot as plt
from constants import DT

import IPython
e = IPython.embed

JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
STATE_NAMES = JOINT_NAMES + ["gripper"]


"""
从指定目录的 HDF5 文件中加载仿真数据。
数据内容包括：
    qpos：机器人关节的位置。
    qvel：关节速度。
    action：控制命令。
    image_dict：从多个相机视角采集的图像数据

返回值：
    qpos, qvel, action：对应每个时间步的数据。
    image_dict：包含相机图像的字典，键为相机名称。 
"""
def load_hdf5(dataset_dir, dataset_name):
    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        is_sim = root.attrs['sim']
        qpos = root['/observations/qpos'][()]
        qvel = root['/observations/qvel'][()]
        action = root['/action'][()]
        image_dict = dict()
        for cam_name in root[f'/observations/images/'].keys():
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]

    return qpos, qvel, action, image_dict

"""
主函数
功能：
    从命令行获取数据集路径和集编号。
    调用 load_hdf5 加载数据。
    将图像保存为视频，并绘制关节状态和命令图。
"""
def main(args):
    dataset_dir = args['dataset_dir']
    episode_idx = args['episode_idx']
    dataset_name = f'episode_{episode_idx}'

    qpos, qvel, action, image_dict = load_hdf5(dataset_dir, dataset_name)
    save_videos(image_dict, DT, video_path=os.path.join(dataset_dir, dataset_name + '_video.mp4'))
    visualize_joints(qpos, action, plot_path=os.path.join(dataset_dir, dataset_name + '_qpos.png'))
    # visualize_timestamp(t_list, dataset_path) # TODO addn timestamp back


"""
将仿真生成的图像序列保存为 MP4 格式的视频文件。
支持两种输入格式：
    图像序列为列表，每个时间步包含不同相机的图像。
    图像序列为字典，每个键对应一个相机视角的视频。

处理逻辑：
    将所有相机图像横向拼接。
    使用 OpenCV 的 VideoWriter 将帧序列保存为视频。
"""
def save_videos(video, dt, video_path=None):
    if isinstance(video, list):
        cam_names = list(video[0].keys())
        h, w, _ = video[0][cam_names[0]].shape
        w = w * len(cam_names)
        fps = int(1/dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for ts, image_dict in enumerate(video):
            images = []
            for cam_name in cam_names:
                image = image_dict[cam_name]
                image = image[:, :, [2, 1, 0]] # swap B and R channel
                images.append(image)
            images = np.concatenate(images, axis=1)
            out.write(images)
        out.release()
        print(f'Saved video to: {video_path}')
    elif isinstance(video, dict):
        cam_names = list(video.keys())
        all_cam_videos = []
        for cam_name in cam_names:
            all_cam_videos.append(video[cam_name])
        all_cam_videos = np.concatenate(all_cam_videos, axis=2) # width dimension

        n_frames, h, w, _ = all_cam_videos.shape
        fps = int(1 / dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for t in range(n_frames):
            image = all_cam_videos[t]
            image = image[:, :, [2, 1, 0]]  # swap B and R channel
            out.write(image)
        out.release()
        print(f'Saved video to: {video_path}')

"""
状态和动作可视化.绘制关节状态和命令(画二维状态图)
功能：
    绘制机器人关节的状态 (qpos) 和对应的控制命令 (command)。
    每个关节的状态与命令绘制在一行图中。
处理逻辑：
    使用 matplotlib 绘图。
    支持设置 y 轴范围 (ylim) 和标签自定义 (label_overwrite)。
"""
def visualize_joints(qpos_list, command_list, plot_path=None, ylim=None, label_overwrite=None):
    if label_overwrite:
        label1, label2 = label_overwrite
    else:
        label1, label2 = 'State', 'Command'

    qpos = np.array(qpos_list) # ts, dim
    command = np.array(command_list)
    num_ts, num_dim = qpos.shape
    h, w = 2, num_dim
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(w, h * num_figs))

    # plot joint state
    all_names = [name + '_left' for name in STATE_NAMES] + [name + '_right' for name in STATE_NAMES]
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(qpos[:, dim_idx], label=label1)
        ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
        ax.legend()

    # plot arm command
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(command[:, dim_idx], label=label2)
        ax.legend()

    if ylim:
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved qpos plot to: {plot_path}')
    plt.close()

"""
时间戳分析,可视化时间戳
功能：
    检查数据采集的时间序列信息。
    绘制帧间时间差分 (dt) 以及时间戳序列。
处理逻辑：
    从时间戳中计算帧间时间差，并检查是否稳定。
    输出时间序列和差分图。

帧间时间差分 (dt):是相邻时间戳之间的时间差，用来描述两帧之间的时间间隔。
dt 的稳定性和大小决定了数据采集或仿真的时间步长是否均匀。
如果 dt 的波动较大，可能说明系统存在不稳定的情况。

"""
def visualize_timestamp(t_list, dataset_path):
    plot_path = dataset_path.replace('.pkl', '_timestamp.png')
    h, w = 4, 10
    fig, axs = plt.subplots(2, 1, figsize=(w, h*2))
    # process t_list
    t_float = []
    for secs, nsecs in t_list:
        t_float.append(secs + nsecs * 10E-10)
    t_float = np.array(t_float)

    ax = axs[0]
    ax.plot(np.arange(len(t_float)), t_float)
    ax.set_title(f'Camera frame timestamps')
    ax.set_xlabel('timestep')
    ax.set_ylabel('time (sec)')

    ax = axs[1]
    ax.plot(np.arange(len(t_float)-1), t_float[:-1] - t_float[1:])
    ax.set_title(f'dt')
    ax.set_xlabel('timestep')
    ax.set_ylabel('time (sec)')

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved timestamp plot to: {plot_path}')
    plt.close()

"""
命令行接口
参数说明：
    --dataset_dir：HDF5 数据集的路径（必需）。
    --episode_idx：数据集中集编号（可选，默认为 0）。

"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset dir.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', required=False)
    main(vars(parser.parse_args()))
