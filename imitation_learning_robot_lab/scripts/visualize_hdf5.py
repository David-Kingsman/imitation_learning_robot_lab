#!/usr/bin/env python3
"""
HDF5 Dataset Visualization Script
Adapted for imitation_learning_robot_lab project data format
"""

import os
import numpy as np
import cv2
import h5py
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

def load_hdf5(dataset_path):
    """Load HDF5 dataset"""
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist: {dataset_path}')
        return None, None, None
    
    with h5py.File(dataset_path, 'r') as root:
        # Load data
        agent_pos = root['/observations/agent_pos'][()]
        actions = root['/actions'][()]
        
        # Load image data
        image_dict = {}
        if 'observations/pixels' in root:
            for cam_name in root['/observations/pixels'].keys():
                image_dict[cam_name] = root[f'/observations/pixels/{cam_name}'][()]
    
    return agent_pos, actions, image_dict

def visualize_trajectories(agent_pos, actions, plot_path):
    """Visualize trajectories"""
    num_ts, num_dim = agent_pos.shape
    
    # Create subplots
    fig, axs = plt.subplots(num_dim, 1, figsize=(10, 2 * num_dim))
    if num_dim == 1:
        axs = [axs]
    
    # State names
    state_names = ['X', 'Y', 'Z', 'Gripper']
    
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        
        # Plot state
        ax.plot(agent_pos[:, dim_idx], label='State', linewidth=2)
        ax.set_title(f'{state_names[dim_idx]} Trajectory')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Position/State')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f'Saved trajectory plot to: {plot_path}')
    plt.close()

def save_videos(image_dict, video_path, fps=10):
    """Save video files"""
    if not image_dict:
        print("No image data to save")
        return
    
    for cam_name, video in image_dict.items():
        if len(video.shape) != 4:  # (T, H, W, C)
            print(f"Skipping {cam_name}: incorrect data format")
            continue
            
        n_frames, h, w, c = video.shape
        if c != 3:
            print(f"Skipping {cam_name}: not 3-channel image")
            continue
            
        # Create video writer
        video_path_cam = video_path.replace('.mp4', f'_{cam_name}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path_cam, fourcc, fps, (w, h))
        
        for frame_idx in range(n_frames):
            frame = video[frame_idx]
            # Ensure image is uint8 format
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            # Convert RGB to BGR (OpenCV format)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f'Saved video to: {video_path_cam}')

def main():
    parser = argparse.ArgumentParser(description='HDF5 Dataset Visualization Tool')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Dataset directory path')
    parser.add_argument('--episode_idx', type=int, help='Specify episode index')
    parser.add_argument('--traj', action='store_true', help='Visualize trajectories')
    parser.add_argument('--video', action='store_true', help='Generate videos')
    
    args = parser.parse_args()
    
    if args.episode_idx is not None:
        # Process single episode
        dataset_name = f'episode_{args.episode_idx:06d}'
        dataset_path = os.path.join(args.dataset_dir, dataset_name + '.hdf5')
        
        print(f"Processing single episode: {dataset_name}")
        agent_pos, actions, image_dict = load_hdf5(dataset_path)
        
        if agent_pos is None:
            return
        
        base_path = os.path.join(args.dataset_dir, dataset_name)
        
        if args.traj:
            visualize_trajectories(agent_pos, actions, base_path + '_trajectory.png')
        if args.video:
            save_videos(image_dict, base_path + '_video.mp4')
            
    else:
        # Process all episodes in directory
        print(f"Processing all episodes in directory: {args.dataset_dir}")
        
        file_list = [f for f in os.listdir(args.dataset_dir) if f.endswith('.hdf5')]
        file_list.sort()
        
        if not file_list:
            print("No HDF5 files found in directory")
            return
        
        print(f"Found {len(file_list)} HDF5 files")
        
        for filename in file_list:
            episode_name = filename[:-5]  # Remove .hdf5 suffix
            dataset_path = os.path.join(args.dataset_dir, filename)
            
            print(f"\nProcessing: {episode_name}")
            agent_pos, actions, image_dict = load_hdf5(dataset_path)
            
            if agent_pos is None:
                continue
            
            base_path = os.path.join(args.dataset_dir, episode_name)
            
            if args.traj:
                visualize_trajectories(agent_pos, actions, base_path + '_trajectory.png')
            if args.video:
                save_videos(image_dict, base_path + '_video.mp4')

if __name__ == '__main__':
    main()
