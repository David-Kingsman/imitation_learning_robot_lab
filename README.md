# Imitation Learning for LeRobot

A comprehensive framework for imitation learning using the LeRobot library, featuring multiple robotic environments, teleoperation interfaces, and data collection pipelines.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Data Collection](#data-collection)
- [Training](#training)
- [Evaluation](#evaluation)
- [Supported Environments](#supported-environments)
- [Teleoperation Interfaces](#teleoperation-interfaces)
- [Data Formats](#data-formats)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project implements imitation learning pipelines for robotic manipulation tasks using the LeRobot framework. It provides a complete workflow from data collection through teleoperation to policy training and evaluation.

## Features

- **Multiple Robotic Environments**: Dishwasher, pick-and-place, pick-box, and bartending scenarios
- **Teleoperation Interfaces**: JoyCon controllers and keyboard input support
- **Data Collection**: Automated data collection with video encoding
- **Training Pipeline**: End-to-end imitation learning training
- **Evaluation Tools**: Policy rollout and performance assessment
- **Flexible Data Formats**: Support for both HDF5 and LeRobot dataset formats

## Installation

### Prerequisites

- Python 3.8+
- Conda or Miniconda
- FFmpeg (for video processing)
- CUDA-compatible GPU (optional, for accelerated training)

### Environment Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd imitation_learning_lerobot
   ```

2. **Create and activate conda environment**
   ```bash
   conda create -n lerobot python=3.8
   conda activate lerobot
   ```

3. **Install LeRobot**
   ```bash
   cd lerobot
   pip install -e .
   cd ..
   ```

4. **Install project dependencies**
   ```bash
   pip install -e .
   ```

5. **Install additional requirements**
   ```bash
   pip install opencv-python h5py numpy
   ```

6. **Verify FFmpeg installation**
   ```bash
   ffmpeg -version
   ```

## Project Structure

```
imitation_learning_lerobot/
├── envs/                          # Environment implementations
│   ├── dishwasher_env.py         # Dishwasher manipulation environment
│   ├── pick_and_place_env.py     # Pick and place tasks
│   ├── pick_box_env.py           # Box picking environment
│   └── bartend_env.py            # Bartending tasks
├── teleoperation/                 # Teleoperation interfaces
│   ├── joycon/                   # JoyCon controller handlers
│   └── keyboard/                 # Keyboard input handlers
├── scripts/                       # Main execution scripts
│   ├── collect_data.py           # Automated data collection
│   ├── collect_data_teleoperation.py  # Teleoperation data collection
│   ├── convert_h5_to_lerobot.py  # Data format conversion
│   └── train_*.py                # Training scripts
├── configs/                       # Environment configurations
├── arm/                          # Robot arm models and configurations
├── assets/                       # 3D models and scene files
└── outputs/                      # Generated datasets and checkpoints
```

## Quick Start

### 1. Data Collection

**Automated data collection:**
```bash
conda activate lerobot
python ./imitation_learning_lerobot/scripts/collect_data.py \
  --env.type=dishwasher \
  --episode=100
```

**Teleoperation data collection:**
```bash
conda activate lerobot
python ./imitation_learning_lerobot/scripts/collect_data_teleoperation.py \
  --env.type=pick_box \
  --handler.type=joycon
```

### 2. Data Format Conversion

Convert HDF5 data to LeRobot format:
```bash
python ./imitation_learning_lerobot/scripts/convert_h5_to_lerobot.py \
  --env.type=dishwasher
```

### 3. Training

Train imitation learning policy:
```bash
python ./imitation_learning_lerobot/scripts/train_dishwasher.py
```

### 4. Evaluation

Evaluate trained policy:
```bash
python ./imitation_learning_lerobot/scripts/rollout_dishwasher.sh
```

## Data Collection

### Automated Collection

The automated data collection script generates synthetic demonstrations using predefined policies:

```bash
python ./imitation_learning_lerobot/scripts/collect_data.py \
  --env.type=<environment_type> \
  --episode=<number_of_episodes>
```

**Parameters:**
- `--env.type`: Environment type (dishwasher, pick_and_place, pick_box, bartend)
- `--episode`: Number of episodes to collect

### Teleoperation Collection

Collect human demonstrations using teleoperation interfaces:

```bash
python ./imitation_learning_lerobot/scripts/collect_data_teleoperation.py \
  --env.type=<environment_type> \
  --handler.type=<interface_type>
```

**Supported interfaces:**
- `joycon`: Nintendo JoyCon controllers
- `keyboard`: Keyboard input

**Supported environments:**
- `pick_box`: Box picking and manipulation
- `transfer_cube`: Cube transfer tasks
- `dishwasher`: Dishwasher loading/unloading

### Data Output

- **HDF5 format**: `outputs/datasets/{env_name}_hdf5/`
- **LeRobot format**: `outputs/datasets/{env_name}/`
- **Video files**: MP4 encoded with H.264 codec
- **Metadata**: Parquet files with episode information

## Training

### Training Scripts

Each environment has a dedicated training script:

```bash
# Dishwasher environment
python ./imitation_learning_lerobot/scripts/train_dishwasher.py

# Pick and place environment
python ./imitation_learning_lerobot/scripts/train_pick_and_place.py

# Pick box environment
python ./imitation_learning_lerobot/scripts/train_pick_box.py
```

### Training Configuration

Training parameters can be modified in the respective training scripts:
- Learning rate
- Batch size
- Number of epochs
- Model architecture
- Data augmentation

### Checkpoint Management

Trained models are saved in:
```
outputs/checkpoints/{env_name}/
├── model_best.pth
├── model_latest.pth
└── training_logs/
```

## Evaluation

### Policy Rollout

Evaluate trained policies using rollout scripts:

```bash
# Dishwasher environment
bash scripts/rollout_dishwasher.sh

# Pick and place environment
bash scripts/rollout_pick_and_place.sh
```

### Performance Metrics

Evaluation includes:
- Success rate
- Task completion time
- Trajectory smoothness
- Policy consistency

## Supported Environments

### 1. Dishwasher Environment
- **Task**: Load/unload dishes from dishwasher
- **Robots**: Dual-arm setup with grippers
- **Cameras**: Overhead and wrist-mounted cameras
- **State space**: 14-dimensional robot state
- **Action space**: 14-dimensional joint commands

### 2. Pick and Place Environment
- **Task**: Pick objects and place them in target locations
- **Robots**: Single or dual-arm configuration
- **Cameras**: Multiple viewpoints for object tracking
- **State space**: Variable based on robot configuration
- **Action space**: Joint or end-effector commands

### 3. Pick Box Environment
- **Task**: Grasp and manipulate boxes
- **Robots**: Single-arm with gripper
- **Cameras**: RGB-D sensors for depth perception
- **State space**: 7-dimensional robot state
- **Action space**: 7-dimensional joint commands

### 4. Bartending Environment
- **Task**: Mix and serve drinks
- **Robots**: Dual-arm with specialized end-effectors
- **Cameras**: Overhead and close-up views
- **State space**: 14-dimensional robot state
- **Action space**: 14-dimensional joint commands

## Teleoperation Interfaces

### JoyCon Controllers

**Setup:**
1. Connect JoyCon controllers via Bluetooth
2. Install required dependencies: `pip install joycon-python`
3. Run calibration script if needed

**Usage:**
```bash
python ./imitation_learning_lerobot/scripts/collect_data_teleoperation.py \
  --env.type=pick_box \
  --handler.type=joycon
```

**Controls:**
- Left stick: Translation
- Right stick: Rotation
- Triggers: Gripper control
- Buttons: Mode switching

### Keyboard Interface

**Usage:**
```bash
python ./imitation_learning_lerobot/scripts/collect_data_teleoperation.py \
  --env.type=pick_box \
  --handler.type=keyboard
```

**Controls:**
- WASD: Translation
- QE: Rotation
- Space: Gripper control
- Enter: Start/stop recording

## Data Formats

### HDF5 Format

Raw teleoperation data stored in HDF5 files:

```
episode_000000.hdf5
├── observations/
│   ├── agent_pos          # [T, state_dim] robot state
│   └── pixels/            # [T, H, W, 3] camera images
│       ├── overhead_cam
│       └── wrist_cam
└── actions                 # [T, action_dim] robot actions
```

### LeRobot Format

Training-ready dataset format:

```
{env_name}/
├── videos/                 # MP4 encoded video files
│   └── chunk-000/
│       └── observation.images.{camera}/
│           └── episode_000000.mp4
├── metadata/               # Episode information
│   └── chunk-000.parquet
└── dataset_info.json       # Dataset configuration
```

## Troubleshooting

### Common Issues

#### 1. Video Output is Black
**Problem**: Generated videos appear black in some players
**Solution**: Videos use H.264 encoding for compatibility. Use VLC, mpv, or other H.264-compatible players.

#### 2. Module Import Errors
**Problem**: `ModuleNotFoundError: No module named 'imitation_learning_lerobot'`
**Solution**: Ensure conda environment is activated and package is installed:
```bash
conda activate lerobot
pip install -e .
```

#### 3. NumPy Compatibility Issues
**Problem**: NumPy version conflicts
**Solution**: Use compatible NumPy version:
```bash
pip install "numpy<2"
```

#### 4. FFmpeg Not Found
**Problem**: Video encoding fails
**Solution**: Install FFmpeg:
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

### Performance Optimization

- **GPU acceleration**: Enable CUDA for faster training
- **Data loading**: Use SSD storage for large datasets
- **Memory management**: Adjust batch sizes based on available RAM
- **Video encoding**: Use H.264 codec for compatibility and speed

## Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Install development dependencies
4. Make changes and add tests
5. Submit a pull request

### Code Style

- Follow PEP 8 guidelines
- Use type hints for function parameters
- Add docstrings for all public functions
- Include unit tests for new features

### Testing

Run tests before submitting changes:
```bash
python -m pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{imitation_learning_lerobot,
  title={Imitation Learning for LeRobot},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/imitation_learning_lerobot}
}
```

## Acknowledgments

- LeRobot team for the underlying framework
- MuJoCo for physics simulation
- OpenCV for computer vision utilities
- PyTorch for deep learning infrastructure

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review troubleshooting section
- Contact the maintainers

---

**Note**: This project is actively maintained. Please check for updates and report any issues you encounter.
