# ROS2 Configuration - Data Collection Pipeline

This document explains the ROS2 settings, data types, and data flow in the `data_collection_keyboard.py` script.

---

## ROS2 Topics Overview

| Topic | Message Type | Direction | Description |
|-------|--------------|-----------|-------------|
| `/joint_command` | `sensor_msgs/JointState` | Publish | Commands sent to robot joints |
| `/joint_states` | `sensor_msgs/JointState` | Subscribe | Current joint positions from simulation |
| `/rgb` | `sensor_msgs/Image` | Subscribe | RGB camera image (640x360) |
| `/wrist_camera` | `sensor_msgs/Image` | Subscribe | Wrist camera (optional) |
| `/top_camera` | `sensor_msgs/Image` | Subscribe | Top camera (optional) |

---

## ROS2 Nodes

### 1. Robot_Keyboard_Controller (Publisher)

**File Location:** Lines 316-821

```python
# Creates publisher for joint commands
self.publisher = self.create_publisher(JointState, '/joint_command', 10)
```

**Published Message Structure:**

```python
msg = JointState()
msg.header.stamp = self.get_clock().now().to_msg()
msg.header.frame_id = 'base_link'
msg.name = ['right_shoulder_link_joint', 'right_arm_top_link_joint', ...]  # 17 joints
msg.position = [0.17, -1.46, 0.35, ...]  # Joint positions in radians
msg.velocity = [0.0, 0.0, ...]  # Velocities (not used)
msg.effort = [0.0, 0.0, ...]  # Efforts (not used)
```

---

### 2. JointStates_Subscriber

**File Location:** Lines 823-842

```python
class JointStates_Subscriber(Node):
    def __init__(self):
        super().__init__('joint_states_subscriber')
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_states_callback,
            10)
    
    def joint_states_callback(self, data):
        global joint_states
        joint_states['names'] = list(data.name)
        joint_states['positions'] = np.array(data.position, dtype=float)
```

**Received Data:**

- `data.name`: List of joint names (all robot joints)
- `data.position`: Current joint positions in radians
- `data.velocity`: Current joint velocities (if available)

---

### 3. RGB_Camera_Subscriber

**File Location:** Lines 844-858

```python
class RGB_Camera_Subscriber(Node):
    def __init__(self):
        super().__init__('rgb_camera_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/rgb',
            self.camera_callback,
            10)
    
    def camera_callback(self, data):
        global rgb_image
        rgb_image = cv2.resize(bridge.imgmsg_to_cv2(data, "bgr8"), (vid_W, vid_H), cv2.INTER_LINEAR)
```

**Image Processing:**

- Input: ROS2 Image message from `/rgb` topic
- Conversion: `cv_bridge` converts to OpenCV BGR format
- Resize: Scaled to 640x360 pixels
- Output: `numpy.ndarray` shape `(360, 640, 3)`, dtype `uint8`

---

### 4. JointCommand_Subscriber

**File Location:** Lines 72-92

```python
class JointCommand_Subscriber(Node):
    def __init__(self):
        super().__init__('joint_command_subscriber')
        self.subscription = self.create_subscription(
            JointState,
            '/joint_command',
            self.joint_command_callback,
            10)
    
    def joint_command_callback(self, data):
        global action
        if len(data.position) > 0:
            action = np.array(data.position, dtype=float)
```

**Purpose:** Captures the commanded joint positions (action) for recording.

---

## Data Recording

### Data_Recorder Node

**File Location:** Lines 899-1100+

**Recording Rate:** 10 Hz (100ms between frames)

```python
self.Hz = 10
self.timer = self.create_timer(1/self.Hz, self.timer_callback)
```

### Recorded Data Fields

| Field | Shape | Description |
|-------|-------|-------------|
| `observation.state` | (17,) | Joint positions from `/joint_states` |
| `action` | (17,) | Commanded joint positions from `/joint_command` |
| `episode_index` | int | Episode number |
| `frame_index` | int | Frame number within episode |
| `timestamp` | float | Time in seconds since episode start |
| `index` | int | Global frame index across all episodes |
| `task_index` | int | Task identifier (default: 0) |

### Joint Names (17 joints for right arm)

```python
right_arm_joint_names = [
    "right_shoulder_link_joint",
    "right_arm_top_link_joint",
    "right_arm_bottom_link_joint",
    "right_forearm_link_joint",
    "wrist_pitch_joint_r",
    "wrist_roll_joint_r",
    "thumb_joint_roll_r",
    "index_proximal_joint_r",
    "middle_proximal_joint_r",
    "ring_proximal_joint_r",
    "little_proximal_joint_r",
    "thumb_proximal_joint_r",
    "index_proximal_joint_r_1",
    "middle_proximal_joint_r_1",
    "ring_proximal_joint_r_1",
    "little_proximal_joint_r_1",
    "thumb_proximal_joint_r_1",
]
```

---

## Output Data Format

### Directory Structure

```
output/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
├── videos/
│   └── observation.images.rgb/
│       └── chunk-000/
│           ├── episode_000000.mp4
│           └── ...
└── meta/
    ├── info.json
    ├── stats.json
    ├── tasks.jsonl
    ├── episodes.jsonl
    └── episodes/
        └── chunk-000/
            └── file-000.parquet
```

### Parquet File Schema

Each parquet file contains one episode with columns:

- `observation.state`: List of 17 floats
- `action`: List of 17 floats
- `episode_index`: int64
- `frame_index`: int64
- `timestamp`: float64
- `index`: int64
- `task_index`: int64

### Video Format

- **Codec:** H.264 (mp4v)
- **Resolution:** 640 x 360
- **Frame Rate:** 10 FPS
- **Color:** BGR (OpenCV default)

---

## ROS2 QoS Settings

All subscribers and publishers use:

- **Queue Size:** 10
- **Reliability:** Best Effort (default)
- **Durability:** Volatile (default)

---

## Data Flow Diagram

```
Isaac Sim (Simulation)
    │
    ├──► /joint_states ──► JointStates_Subscriber ──► observation.state
    │         (JointState)
    │
    ├──► /rgb ──► RGB_Camera_Subscriber ──► video frames
    │     (Image)
    │
    └──◄ /joint_command ◄── Robot_Keyboard_Controller
              (JointState)        │
                                  └──► JointCommand_Subscriber ──► action
                                  
                                          │
                                          ▼
                                   Data_Recorder (10 Hz)
                                          │
                                          ▼
                              ┌───────────────────────┐
                              │  output/              │
                              │  ├── data/*.parquet   │
                              │  ├── videos/*.mp4     │
                              │  └── meta/*.json      │
                              └───────────────────────┘
```
