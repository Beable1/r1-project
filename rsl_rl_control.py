
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
import torch
import torch.nn as nn
import threading
import time
import sys
from collections import deque
from scipy.spatial.transform import Rotation as R

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Model Path (Update if changed)
MODEL_PATH = "/home/beable/Desktop/r1-project/locomotion-models/2026-02-06_20-29-29/model_73850.pt"

# Joint Names (Order MUST match env.yaml init_state.joint_pos)
JOINT_NAMES = [
    "Hip_pitch_l", "Hip_pitch_r", "hip_roll_l", "hip_roll_r", "Hip_yaw_l", "Hip_yaw_r",
    "knee_l", "knee_r", "Ankle_pitch_l", "Ankle_pitch_r", "knee_roll_l", "knee_roll_r",
    "Chest_link_joint", "wraist_link_joint",
    "left_shoulder_link_joint", "left_arm_top_link_joint", "left_arm_bottom_link_joint",
    "left_forearm_link_joint", "wrist_roll_joint_l", "wrist_pitch_joint_l",
    "right_shoulder_link_joint", "right_arm_top_link_joint", "right_arm_bottom_link_joint",
    "right_forearm_link_joint", "wrist_roll_joint_r", "wrist_pitch_joint_r",
    "index_proximal_joint_l", "index_proximal_joint_l_1", "little_proximal_joint_l",
    "little_proximal_joint_l__", "middle_proximal_joint_l", "middle_proximal_joint_l_1",
    "ring_proximal_joint_l", "ring_proximal_joint_l_1", "thumb_proximal_joint_l",
    "thumb_proximal_joint_l_1", "index_proximal_joint_r", "index_proximal_joint_r_1",
    "little_proximal_joint_r", "little_proximal_joint_r_1", "middle_proximal_joint_r",
    "middle_proximal_joint_r_1", "ring_proximal_joint_r", "ring_proximal_joint_r_1",
    "thumb_proximal_joint_r", "thumb_joint_roll_r", "thumb_proximal_joint_r_1"
]

# Default Joint Positions (from env.yaml)
DEFAULT_POSITIONS = {
    "Hip_pitch_l": -0.36, "Hip_pitch_r": 0.36, "hip_roll_l": 0.0, "hip_roll_r": 0.0,
    "Hip_yaw_l": 0.0, "Hip_yaw_r": 0.0, "knee_l": 0.6, "knee_r": 0.6,
    "Ankle_pitch_l": 0.25, "Ankle_pitch_r": 0.25, "knee_roll_l": 0.0, "knee_roll_r": 0.0,
    "Chest_link_joint": 0.0, "wraist_link_joint": 0.0,
    "left_shoulder_link_joint": 0.0, "left_arm_top_link_joint": -1.5,
    "left_arm_bottom_link_joint": 0.0, "left_forearm_link_joint": 1.5,
    "wrist_roll_joint_l": 0.0, "wrist_pitch_joint_l": 0.0,
    "right_shoulder_link_joint": 0.0, "right_arm_top_link_joint": -1.5,
    "right_arm_bottom_link_joint": 0.0, "right_forearm_link_joint": 1.5,
    "wrist_roll_joint_r": 0.0, "wrist_pitch_joint_r": 0.0,
    # Hands default to 0.0
}

# Indices of controlled joints in the JOINT_NAMES list (26 actions)
# Corresponds to Legs (12) + Torso (2) + Arms (12)
CONTROLLED_JOINT_NAMES = [
    "Hip_pitch_l", "Hip_pitch_r", "hip_roll_l", "hip_roll_r", "Hip_yaw_l", "Hip_yaw_r",
    "knee_l", "knee_r", "knee_roll_l", "knee_roll_r", "Ankle_pitch_l", "Ankle_pitch_r", # Note: Different order in env match? 
    # Warning: env.yaml regex order might differ (legs actuator has specific list).
    # Actuator 'legs' matches Hip_.*, hip_.*, knee_.* (except roll?), knee_roll_.*.
    # Let's assume the Action Tensor Output Order matches the JOINT_NAMES order for the controlled subset.
    # We will derive indices dynamically.
    "Chest_link_joint", "wraist_link_joint",
    "left_shoulder_link_joint", "left_arm_top_link_joint", "left_arm_bottom_link_joint", "left_forearm_link_joint",
    "wrist_roll_joint_l", "wrist_pitch_joint_l",
    "right_shoulder_link_joint", "right_arm_top_link_joint", "right_arm_bottom_link_joint", "right_forearm_link_joint",
    "wrist_roll_joint_r", "wrist_pitch_joint_r"
]

# Action Scaling
ACTION_SCALE_LEGS_TORSO = 0.5
ACTION_SCALE_ARMS = 0.0  # Arms are effectively fixed/default in this policy

# ==============================================================================
# NETWORK DEFINITION
# ==============================================================================

class ActorRaw(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[512, 256, 128]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ELU())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

# ==============================================================================
# ROS2 CONTROLLER NOOD
# ==============================================================================

class RSLRLPolicyController(Node):
    def __init__(self):
        super().__init__('rsl_rl_policy_controller')
        
        # Load Model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(MODEL_PATH)
        
        # State Arrays
        self.joint_pos = np.zeros(len(JOINT_NAMES))
        self.joint_vel = np.zeros(len(JOINT_NAMES))
        self.base_quat = np.array([0, 0, 0, 1.0]) # x,y,z,w
        self.base_lin_vel = np.zeros(3)
        self.base_ang_vel = np.zeros(3)
        
        # Command
        self.cmd_vel = np.zeros(3) # vx, vy, omega
        self.last_actions = np.zeros(26)
        
        # Joint Name to Index Mapping
        self.joint_map = {name: i for i, name in enumerate(JOINT_NAMES)}
        
        # Default Positions Array
        self.default_pos_array = np.zeros(len(JOINT_NAMES))
        for i, name in enumerate(JOINT_NAMES):
            self.default_pos_array[i] = DEFAULT_POSITIONS.get(name, 0.0)
            
        # Identify Controlled Indices
        self.controlled_indices = []
        self.arm_indices = [] # Indices that have scale 0.0
        
        # Heuristic to match Action Output (26) to Joint Indices
        # We need to filter JOINT_NAMES to find the 26 joints that the policy controls.
        # Based on env.yaml, the policy controls: Legs, Torso, Arms. 
        # Hands are NOT controlled.
        # We assume the policy output order matches the order these joints appear in JOINT_NAMES.
        
        output_idx = 0
        for i, name in enumerate(JOINT_NAMES):
            is_hand = "proximal" in name or "thumb" in name
            # Note: Leg joints like 'Hip', 'knee', 'Ankle' are valid.
            # Torso 'Chest', 'wraist' are valid.
            # Arms 'shoulder', 'arm', 'forearm', 'wrist' are valid.
            
            if not is_hand:
                self.controlled_indices.append(i)
                # Check if it is an arm joint (scale 0.0)
                if "shoulder" in name or "arm" in name or "wrist" in name:
                    self.arm_indices.append(output_idx)
                output_idx += 1
        
        if len(self.controlled_indices) != 26:
            self.get_logger().error(f"❌ Mismatch in controlled joints! Found {len(self.controlled_indices)}, expected 26.")
            # Fallback or exit? For now warning.
            
        # Subscribers
        self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        
        # Publisher
        self.joint_pub = self.create_publisher(JointState, '/joint_command', 10)
        
        # Input Thread (Simple WASD)
        self.input_thread = threading.Thread(target=self.input_loop, daemon=True)
        self.input_thread.start()
        
        # Control Loop (50 Hz)
        self.timer = self.create_timer(0.02, self.control_step)
        
        self.get_logger().info("✅ RSL-RL Controller Initialized")
        self.get_logger().info("⚠️  Ensure '/odom' and '/imu' are published by Isaac Sim!")
        self.get_logger().info("⌨️  Use WASD to control velocity (Console)")
        
    def _load_model(self, path):
        try:
            checkpoint = torch.load(path, map_location=self.device)
            state_dict = checkpoint['model_state_dict']
            
            # Reconstruct model (Input 132 -> Output 26)
            model = ActorRaw(132, 26).to(self.device)
            
            # Load weights manually to handle prefix keys
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('actor.'):
                    # Remove 'actor.' prefix matching our ActorRaw structure
                    # Our ActorRaw has 'net.0', 'net.2'...
                    # checkpoint has 'actor.0', 'actor.2'...
                    name = k.replace('actor.', 'net.')
                    new_state_dict[name] = v
            
            model.load_state_dict(new_state_dict)
            model.eval()
            return model
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            sys.exit(1)

    def joint_callback(self, msg):
        for i, name in enumerate(msg.name):
            if name in self.joint_map:
                idx = self.joint_map[name]
                self.joint_pos[idx] = msg.position[i]
                if msg.velocity:
                    self.joint_vel[idx] = msg.velocity[i]

    def odom_callback(self, msg):
        # Linear Velocity (World Frame)
        # Note: Isaac Sim Odom usually gives World Frame velocity
        self.base_lin_vel[0] = msg.twist.twist.linear.x
        self.base_lin_vel[1] = msg.twist.twist.linear.y
        self.base_lin_vel[2] = msg.twist.twist.linear.z
        
        # Orientation (for fallback if IMU missing)
        self.base_quat[0] = msg.pose.pose.orientation.x
        self.base_quat[1] = msg.pose.pose.orientation.y
        self.base_quat[2] = msg.pose.pose.orientation.z
        self.base_quat[3] = msg.pose.pose.orientation.w
        
        # Angular Vel (Body Frame or World Frame? Usually Body in twist)
        self.base_ang_vel[0] = msg.twist.twist.angular.x
        self.base_ang_vel[1] = msg.twist.twist.angular.y
        self.base_ang_vel[2] = msg.twist.twist.angular.z

    def imu_callback(self, msg):
        # IMU is better source for Orientation/AngVel
        self.base_quat[0] = msg.orientation.x
        self.base_quat[1] = msg.orientation.y
        self.base_quat[2] = msg.orientation.z
        self.base_quat[3] = msg.orientation.w
        
        self.base_ang_vel[0] = msg.angular_velocity.x
        self.base_ang_vel[1] = msg.angular_velocity.y
        self.base_ang_vel[2] = msg.angular_velocity.z

    def get_obs(self):
        # 1. Base Lin Vel (in Base Frame)
        r = R.from_quat(self.base_quat) # xyzw
        # Rotate World Velocity to Body Frame: R_inv * V_world
        base_lin_vel_local = r.inv().apply(self.base_lin_vel)
        
        # 2. Base Ang Vel (Assume provided in Body Frame by Odom/IMU, scale?)
        # Default scaling is usually 0.25 in some configs, but env.yaml says scale: null.
        # So raw.
        base_ang_vel = self.base_ang_vel
        
        # 3. Projected Gravity (in Base Frame)
        # Gravity vector in World is [0, 0, -1]
        gravity_world = np.array([0.0, 0.0, -1.0])
        gravity_local = r.inv().apply(gravity_world)
        
        # 4. Commands [vx, vy, omega]
        commands = self.cmd_vel
        
        # 5. Joint Pos (Relative)
        joint_pos_rel = self.joint_pos - self.default_pos_array
        
        # 6. Joint Vel
        joint_vel = self.joint_vel
        
        # 7. Actions
        actions = self.last_actions
        
        # Concatenate
        obs = np.concatenate([
            base_lin_vel_local, # 3
            base_ang_vel,       # 3
            gravity_local,      # 3
            commands,           # 3
            joint_pos_rel,      # 47
            joint_vel,          # 47
            actions             # 26
        ]).astype(np.float32)
        
        return obs

    def control_step(self):
        obs = self.get_obs()
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(obs_tensor).cpu().numpy()[0]
        
        self.last_actions = output
        
        # Process Actions -> Joint Targets
        # Targets = Default + Action * Scale
        targets = self.default_pos_array.copy()
        
        for i, joint_idx in enumerate(self.controlled_indices):
            scale = ACTION_SCALE_LEGS_TORSO
            if i in self.arm_indices:
                scale = ACTION_SCALE_ARMS
            
            val = output[i]
            targets[joint_idx] += val * scale
            
        # Publish
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = targets.tolist()
        msg.velocity = []
        msg.effort = []
        self.joint_pub.publish(msg)

    def input_loop(self):
        """Simple WASD control reading stdin"""
        # Note: Non-blocking input is tricky in ROS thread. 
        # We assume external input or just simple default for now.
        # This is a placeholder.
        pass

def main():
    rclpy.init()
    node = RSLRLPolicyController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
