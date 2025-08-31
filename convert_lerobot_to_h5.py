#!/usr/bin/env python3
"""
将LeRobot格式的数据集转换为HDF5格式
用于兼容其他机器学习框架
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from imitation_learning_lerobot.envs import EnvFactory
from lerobot.common.datasets import LeRobotDataset

def parse_args():
    parser = argparse.ArgumentParser(description="将LeRobot格式转换为HDF5格式")
    parser.add_argument(
        '--env.type',
        type=str,
        dest='env_type',
        required=True,
        help='环境类型 (dishwasher, pick_and_place, pick_box等)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='HDF5输出目录 (默认: outputs/datasets/{env_name}_hdf5)'
    )
    return parser.parse_args()

def convert_lerobot_to_h5(env_type: str, output_dir: str = None):
    """将LeRobot格式数据集转换为HDF5格式"""
    
    # 获取环境类
    env_cls = EnvFactory.get_strategies(env_type)
    
    # 设置输出目录
    if output_dir is None:
        output_dir = Path(__file__).parent / Path("outputs/datasets") / Path(f"{env_type}_hdf5")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载LeRobot数据集
    lerobot_root = Path(__file__).parent / Path("outputs/datasets") / Path(env_type)
    if not lerobot_root.exists():
        print(f"错误: LeRobot数据集目录不存在: {lerobot_root}")
        return
    
    print(f"正在加载LeRobot数据集: {lerobot_root}")
    dataset = LeRobotDataset(env_type, root=lerobot_root)
    
    # 获取episode数量
    num_episodes = len(dataset.meta.episodes)
    print(f"找到 {num_episodes} 个episode")
    
    # 转换每个episode
    for episode_idx in range(num_episodes):
        print(f"正在转换 episode {episode_idx:06d}...")
        
        # 获取episode数据
        episode_data = dataset[episode_idx]
        
        # 创建HDF5文件
        h5_filename = output_dir / f"episode_{episode_idx:06d}.hdf5"
        
        with h5py.File(h5_filename, 'w') as h5_file:
            # 创建observations组
            obs_group = h5_file.create_group('observations')
            
            # 保存agent_pos
            agent_pos = episode_data['observation.state'].numpy()
            obs_group.create_dataset('agent_pos', data=agent_pos, compression='gzip')
            
            # 创建pixels组
            pixels_group = obs_group.create_group('pixels')
            
            # 保存每个相机的图像数据
            for camera in env_cls.cameras:
                camera_key = f'observation.images.{camera}'
                if camera_key in episode_data:
                    # 获取图像数据 (从视频解码)
                    images = episode_data[camera_key]  # 应该是 [T, H, W, C] 格式
                    
                    # 确保数据类型正确
                    if images.dtype != np.uint8:
                        images = (images * 255).astype(np.uint8)
                    
                    # 保存到HDF5
                    pixels_group.create_dataset(
                        camera, 
                        data=images, 
                        compression='gzip',
                        chunks=(1, images.shape[1], images.shape[2], images.shape[3])
                    )
            
            # 保存actions
            actions = episode_data['action'].numpy()
            h5_file.create_dataset('actions', data=actions, compression='gzip')
            
            # 保存元数据
            h5_file.attrs['episode_length'] = len(actions)
            h5_file.attrs['env_type'] = env_type
            h5_file.attrs['robot_type'] = env_cls.robot_type
            h5_file.attrs['fps'] = dataset.fps
        
        print(f"  ✓ 已保存: {h5_filename}")
    
    print(f"\n转换完成! HDF5文件保存在: {output_dir}")
    print(f"总共转换了 {num_episodes} 个episode")

def main():
    args = parse_args()
    
    print(f"开始转换环境: {args.env_type}")
    print(f"输出目录: {args.output_dir or '默认'}")
    
    try:
        convert_lerobot_to_h5(args.env_type, args.output_dir)
    except Exception as e:
        print(f"转换过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
