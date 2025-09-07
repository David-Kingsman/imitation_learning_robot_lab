# Imitation Learning Robot Lab

基于LeRobot库的机器人模仿学习框架，支持多种机器人环境和遥操作接口。

## 功能特性

- **多机器人环境**: 洗碗机、抓取放置、抓取盒子、调酒等场景
- **遥操作接口**: JoyCon手柄和键盘输入支持
- **数据收集**: 自动化数据收集和视频编码
- **训练管道**: 端到端模仿学习训练
- **评估工具**: 策略回放和性能评估

## 安装

```bash
# 克隆仓库
git clone https://github.com/David-Kingsman/imitation_learning_robot_lab
cd imitation_learning_robot_lab

# 创建并激活conda环境
conda create -n lerobot python=3.10
conda activate lerobot

# 安装LeRobot
cd lerobot && pip install -e .
cd ..

# 安装项目依赖
pip install -e .
```

## 快速开始

### 数据收集

**自动数据收集:**
```bash
python ./imitation_learning_robot_lab/scripts/collect_data.py \
  --env.type=dishwasher \
  --episode=100
```

**遥操作数据收集:**
```bash
python ./imitation_learning_robot_lab/scripts/collect_data_teleoperation.py \
  --env.type=pick_box \
  --handler.type=keyboard
```

### 训练

```bash
python ./scripts/train_dishwasher.py
```

### 评估

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



## Acknowledgments
The repo was referenced to Lerobot, Cybaster/imitation_learning_lerobot


