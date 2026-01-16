# R1 Humanoid Robot - Dexterous Manipulation Project

This project contains a complete pipeline for dexterous manipulation tasks using the R1 humanoid robot's right arm.

---

## Table of Contents

- [1. Data Collection](#1-data-collection)
- [2. Dataset Preparation](#2-dataset-preparation)
- [3. Model Training](#3-model-training)
- [4. Policy Control](#4-policy-control)
- [Troubleshooting](#troubleshooting)

---

## 1. Data Collection

### Script: `data_collection_keyboard.py`

Collects demonstration data by teleoperating the robot using keyboard and mouse.

### Prerequisites

**Before running, you must:**

1. Open Isaac Sim
2. Load the robot scene (`r1_scene.usd`)
3. Press Play to start the simulation
4. Ensure ROS2 bridge is active

### Running

```bash
conda activate isaaclab
source /opt/ros/humble/setup.bash
python data_collection_keyboard.py
```

### Keyboard Controls

| Key | Function |
|-----|----------|
| `A/W` | Shoulder joint +/- |
| `S/D` | Upper arm joint +/- |
| `E/R` | Lower arm joint +/- |
| `F/T` | Forearm joint +/- |
| `Q/Y` | Wrist pitch +/- |
| `Z/X` | Wrist roll +/- |
| `C/V` | Thumb roll +/- |
| `SPACE` | Cycle finger positions (open → half → closed) |
| `B` | Toggle thumb roll |

### Recording Controls

| Key | Function |
|-----|----------|
| `SPACE` | Start/Stop recording (also ends episode) |
| `H` | Go to home position |
| `P` | Show current positions |
| `ESC` | Exit |

### Output

Data is saved in LeRobot format under the `output/` directory:

- **Parquet files**: Joint states and actions
- **MP4 videos**: RGB camera recordings
- **Metadata**: Episode info, statistics

---

## 2. Dataset Preparation

### Script: `fix_dataset_metadata.py`

Fixes and validates the collected dataset to ensure LeRobot compatibility.

### Running

```bash
python fix_dataset_metadata.py
```

### What it does

- Renames files to LeRobot format (`episode_000000.parquet`)
- Creates metadata files (`tasks.jsonl`, `episodes.jsonl`, `episodes_stats.jsonl`)
- Computes image and state statistics
- Updates `info.json` with correct format

---

## 3. Model Training

### Diffusion Policy

```bash
python lerobot/scripts/train.py \
    --policy.type=diffusion \
    --dataset.repo_id=local \
    --dataset.root=/path/to/your/dataset \
    --training.batch_size=32 \
    --training.steps=100000 \
    --output_dir=outputs/train/diffusion_my_task
```

### ACT Policy

```bash
python lerobot/scripts/train.py \
    --policy.type=act \
    --dataset.repo_id=local \
    --dataset.root=/path/to/your/dataset \
    --policy.chunk_size=100 \
    --training.batch_size=8 \
    --training.steps=100000 \
    --output_dir=outputs/train/act_my_task
```

---

## 4. Policy Control

### 4.1 Diffusion Policy Control

#### Script: `diffusion_policy_control.py`

### Prerequisites

**Before running, you must:**

1. Open Isaac Sim
2. Load the robot scene (`r1_scene.usd`)
3. Press Play to start the simulation
4. Ensure ROS2 bridge is active

### Configuration

Edit the script to set your model path:

```python
MODEL_PATH = "/path/to/outputs/train/diffusion_my_task/checkpoints/100000/pretrained_model"
```

### Running

```bash
conda activate isaaclab
source /opt/ros/humble/setup.bash
python diffusion_policy_control.py
```

### Commands

| Command | Function |
|---------|----------|
| `ENTER` | Start/stop control |
| `h` | Go to home position |
| `q` | Quit |

---

### 4.2 ACT Policy Control

#### Script: `act_policy_control.py`

### Prerequisites

**Before running, you must:**

1. Open Isaac Sim
2. Load the robot scene (`r1_scene.usd`)
3. Press Play to start the simulation
4. Ensure ROS2 bridge is active

### Configuration

Edit the script to set your model path:

```python
MODEL_PATH = "/path/to/outputs/train/act_my_task/checkpoints/100000/pretrained_model"
```

### Running

```bash
conda activate isaaclab
source /opt/ros/humble/setup.bash
python act_policy_control.py
```

### Commands

| Command | Function |
|---------|----------|
| `ENTER` | Start/stop control |
| `h` | Go to home position |
| `q` | Quit |

---

## ROS2 Topics

| Topic | Type | Direction | Description |
|-------|------|-----------|-------------|
| `/joint_command` | `JointState` | Publish | Joint position commands |
| `/joint_states` | `JointState` | Subscribe | Current joint positions |
| `/rgb` | `Image` | Subscribe | RGB camera image |

---

## Troubleshooting

### NumPy Version Error

```
AttributeError: _ARRAY_API not found
```

**Solution:** `pip install 'numpy<2.0'`

### LeRobot Import Error

```
ModuleNotFoundError: No module named 'lerobot.constants'
```

**Solution:** Check LeRobot version:

- v0.3.x: `from lerobot.constants import ...`
- v0.4.x: `from lerobot.utils.constants import ...`

### No ROS2 Topics

Make sure Isaac Sim is running with the scene loaded and Play is pressed.

```bash
# Check available topics
ros2 topic list

# Should see: /joint_states, /joint_command, /rgb
```

### Policy Not Moving Robot

1. Verify model path is correct
2. Check that `pretrained_model` folder contains `config.json` and `model.safetensors`
3. Ensure Isaac Sim simulation is running (not paused)

---

## Tips

- Collect 30-50 episodes for initial training
- Use diverse starting positions
- Include both successful and failed attempts
- Save checkpoints frequently during training
- Always go to home position before starting policy control
