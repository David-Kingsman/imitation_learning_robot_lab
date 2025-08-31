# Imitation Learning Robot Lab

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

   ```bash
   # Clone the repository
   git clone https://github.com/David-Kingsman/imitation_learning_robot_lab
   cd imitation_learning_robot_lab
   # Create and activate conda environment
   conda create -n lerobot python=3.10
   conda activate lerobot
   # Install LeRobot
   cd lerobot & pip install -e .
   cd ..
   # Install project dependencies
   pip install -e .
   ```

## Project Structure

```yaml
imitation_learning_robot_lab/
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

## Data Collection

**Automated data collection:**
```bash
python ./imitation_learning_robot_lab/scripts/collect_data.py \
  --env.type=dishwasher \
  --episode=100
```

**Teleoperation data collection:**
```bash
python ./imitation_learning_robot_lab/scripts/collect_data_teleoperation.py \
  --env.type=pick_box \    
  --handler.type=joycon     # keyboard/joycon
```

**Visualize teleoperation data collection hdf5 files:**
```bash
python ./imitation_learning_robot_lab/scripts/visualize_hdf5.py --dataset_dir outputs/datasets/pick_box_hdf5 --traj --video
```

## Data Format Conversion

Convert HDF5 data to LeRobot format:
```bash
python ./imitation_learning_robot_lab/scripts/convert_h5_to_lerobot.py \
  --env.type=dishwasher
```

## Training

Train imitation learning policy:
```bash
python ./scripts/train_dishwasher.py
```

### 4. Evaluation

Evaluate trained policy:
```bash
python ./scripts/rollout_dishwasher.sh
```


## Data Collection

### Automated Collection

The automated data collection script generates synthetic demonstrations using predefined policies:

```bash
python ./imitation_learning_robot_lab/scripts/collect_data.py \
  --env.type=pick_box \
  --episode=100
```

**Parameters:**
- `--env.type`: Environment type 
  - `pick_box`: Box picking and manipulation 
  - `pick_and_place`: Pick and place tasks  
  - `transfer_cube`: Cube transfer tasks 
  - `dishwasher`: Dishwasher loading/unloading 
  - `bartend`: Bartending tasks 
- `--episode`: Number of episodes to collect (default: 100)

### Teleoperation Collection

Collect human demonstrations using teleoperation interfaces:

```bash
python ./imitation_learning_robot_lab/scripts/collect_data_teleoperation.py \
  --env.type=pick_box \
  --handler.type=keyboard
```

**Supported interfaces:**
- `keyboard`: Keyboard input (fully supported)
- `joycon`: Nintendo JoyCon controllers (requires pyjoycon module)

**Supported environments:**
- `pick_box`: Box picking and manipulation (keyboard support)
- `transfer_cube`: Cube transfer tasks (no keyboard support)
- `dishwasher`: Dishwasher loading/unloading (no keyboard support)
- `pick_and_place`: Pick and place tasks (no keyboard support)
- `bartend`: Bartending tasks (no keyboard support)

**Note**: Currently only `pick_box` environment supports keyboard control. Other environments require JoyCon controllers or custom keyboard handlers.

**Keyboard Controls for pick_box:**
- **Start Control**: Right Ctrl key
- **Pause Control**: Right Shift key
- **Stop Episode**: Enter key
- **X-axis Movement**: Keypad 1 (-) / 7 (+)
- **Y-axis Movement**: Keypad 4 (-) / 6 (+)
- **Z-axis Movement**: Keypad 2 (-) / 8 (+)
- **Gripper Control**: Keypad 3 (close) / 9 (open)

### Quick Start Examples

**Collect data with automatic policy:**
```bash
# Collect 50 episodes of pick_box data
python ./imitation_learning_robot_lab/scripts/collect_data.py --env.type=pick_box --episode=50

# Collect 100 episodes of pick_and_place data  
python ./imitation_learning_robot_lab/scripts/collect_data.py --env.type=pick_and_place --episode=100
```

**Collect data with keyboard teleoperation:**
```bash
# Use keyboard to control pick_box robot
python ./imitation_learning_robot_lab/scripts/collect_data_teleoperation.py --env.type=pick_box --handler.type=keyboard
```

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
python ./imitation_learning_robot_lab/scripts/train_dishwasher.py

# Pick and place environment
python ./imitation_learning_robot_lab/scripts/train_pick_and_place.py

# Pick box environment
python ./imitation_learning_robot_lab/scripts/train_pick_box.py
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
python ./imitation_learning_robot_lab/scripts/collect_data_teleoperation.py \
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
python ./imitation_learning_robot_lab/scripts/collect_data_teleoperation.py \
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


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.




## Acknowledgments



