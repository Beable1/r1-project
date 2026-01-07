#!/usr/bin/env python3
"""
ACT (Action Chunking Transformer) Policy Controller for R1 Humanoid Robot
Loads a trained ACT model and controls the robot like teleop.

Usage:
    conda activate isaaclab  # or lerobot environment
    python3 act_policy_control.py
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
import numpy as np
import torch
import threading
import time
import cv2
from cv_bridge import CvBridge
from collections import deque

# LeRobot imports
from lerobot.policies.act.modeling_act import ACTPolicy


class ACTPolicyController(Node):
    """ROS2 Node that controls robot using trained ACT Policy"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        super().__init__('act_policy_controller')
        
        self.device = device
        self.bridge = CvBridge()
        
        # Video dimensions (must match training - from config)
        self.vid_H = 360
        self.vid_W = 640
        
        # Joint names (same as teleop)
        self.joint_names = [
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
            "thumb_proximal_joint_r_1"
        ]
        
        # Home positions (robot's natural starting pose - from data_collection_keyboard.py)
        self.home_positions = {
            "right_shoulder_link_joint": 0.17,
            "right_arm_top_link_joint": -1.46,
            "right_arm_bottom_link_joint": 0.35,
            "right_forearm_link_joint": 1.7,
            "wrist_pitch_joint_r": 0.0,
            "wrist_roll_joint_r": 0.0,
            "thumb_joint_roll_r": 0.0,
            "index_proximal_joint_r": 0.0,
            "middle_proximal_joint_r": 0.0,
            "ring_proximal_joint_r": 0.0,
            "little_proximal_joint_r": 0.0,
            "thumb_proximal_joint_r": 0.0,
            "index_proximal_joint_r_1": 0.0,
            "middle_proximal_joint_r_1": 0.0,
            "ring_proximal_joint_r_1": 0.0,
            "little_proximal_joint_r_1": 0.0,
            "thumb_proximal_joint_r_1": 0.0,
        }
        
        # Current joint positions from simulation
        self.current_positions = {name: 0.0 for name in self.joint_names}
        self.current_rgb_image = np.zeros((self.vid_H, self.vid_W, 3), dtype=np.uint8)
        
        # Load the model
        self.get_logger().info(f"🧠 Loading ACT Policy from: {model_path}")
        self._load_model(model_path)
        
        # Policy config - ACT specific
        self.n_obs_steps = self.policy.config.n_obs_steps  # Should be 1 for ACT
        self.chunk_size = self.policy.config.chunk_size  # 100 for this model
        self.n_action_steps = self.policy.config.n_action_steps  # 100 for this model
        
        # ACT uses n_obs_steps=1 typically, so we only need current observation
        # But we still maintain a buffer for consistency
        self.state_buffer = deque(maxlen=self.n_obs_steps)
        self.image_buffer = deque(maxlen=self.n_obs_steps)
        
        # Initialize buffers with zeros
        for _ in range(self.n_obs_steps):
            self.state_buffer.append(np.zeros(17, dtype=np.float32))
            self.image_buffer.append(np.zeros((3, self.vid_H, self.vid_W), dtype=np.float32))
        
        # Action queue (for temporal action chunking - ACT predicts chunk_size actions)
        self.action_queue = deque(maxlen=self.chunk_size)
        
        # ROS2 Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_states_callback, 10)
        self.rgb_sub = self.create_subscription(
            Image, '/rgb', self.rgb_callback, 10)
        
        # ROS2 Publisher
        self.joint_cmd_pub = self.create_publisher(JointState, '/joint_command', 10)
        
        # Control loop timer (10Hz like data collection)
        self.control_rate = 10.0  # Hz
        self.control_timer = self.create_timer(1.0 / self.control_rate, self.control_loop)
        
        # State
        self.is_running = False
        self.inference_count = 0
        
        self.get_logger().info("✅ ACT Policy Controller initialized!")
        self.get_logger().info(f"   n_obs_steps: {self.n_obs_steps}, chunk_size: {self.chunk_size}")
        self.get_logger().info(f"   n_action_steps: {self.n_action_steps}")
        self.get_logger().info("   Press ENTER to start/stop control, 'q' to quit")
    
    def _load_model(self, model_path: str):
        """Load the pretrained ACT policy"""
        try:
            self.policy = ACTPolicy.from_pretrained(model_path)
            self.policy.to(self.device)
            self.policy.eval()
            
            # Reset policy's internal action queue
            self.policy.reset()
            
            self.get_logger().info(f"✅ Model loaded successfully!")
            self.get_logger().info(f"   Device: {self.device}")
            
        except Exception as e:
            self.get_logger().error(f"❌ Failed to load model: {e}")
            raise
    
    def joint_states_callback(self, msg):
        """Receive current joint states from simulation"""
        for i, name in enumerate(msg.name):
            if name in self.current_positions:
                self.current_positions[name] = msg.position[i]
    
    def rgb_callback(self, msg):
        """Receive RGB camera image"""
        try:
            self.current_rgb_image = cv2.resize(
                self.bridge.imgmsg_to_cv2(msg, "bgr8"),
                (self.vid_W, self.vid_H),
                cv2.INTER_LINEAR
            )
        except Exception as e:
            self.get_logger().warning(f"RGB decode error: {e}")
    
    def _update_observation_buffers(self):
        """Add current observation to buffers"""
        # Get current state (17 joint positions)
        state = np.array([self.current_positions[name] for name in self.joint_names], dtype=np.float32)
        
        # Get current image (BGR -> RGB, HWC -> CHW, normalize to 0-1)
        image = self.current_rgb_image[:, :, ::-1].copy()  # BGR -> RGB
        image = image.transpose(2, 0, 1).astype(np.float32) / 255.0  # HWC -> CHW, 0-1
        
        # Add to buffers
        self.state_buffer.append(state)
        self.image_buffer.append(image)
    
    def _prepare_batch(self):
        """Prepare batch dict for ACT model
        
        ACT expects:
        - observation.state: (batch, state_dim) for n_obs_steps=1
        - observation.images.rgb: (batch, C, H, W) for n_obs_steps=1
        """
        # Get the most recent observation
        state = self.state_buffer[-1]  # (17,)
        image = self.image_buffer[-1]  # (3, H, W)
        
        # Create tensors with batch dimension
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)  # (1, 17)
        image_tensor = torch.from_numpy(image).unsqueeze(0).to(self.device)  # (1, 3, H, W)
        
        batch = {
            "observation.state": state_tensor,
            "observation.images.rgb": image_tensor,
        }
        
        return batch
    
    @torch.no_grad()
    def _run_inference(self):
        """Run ACT policy inference and get action chunk"""
        batch = self._prepare_batch()
        
        # ACT's select_action handles normalization internally
        # It returns (chunk_size, action_dim) or (action_dim,) based on implementation
        action = self.policy.select_action(batch)
        
        # Convert to numpy
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        
        # Handle different output shapes
        if action.ndim == 1:
            # Single action returned
            self.action_queue.clear()
            self.action_queue.append(action)
        else:
            # Multiple actions returned (chunk)
            self.action_queue.clear()
            for i in range(len(action)):
                self.action_queue.append(action[i])
        
        self.inference_count += 1
    
    def control_loop(self):
        """Main control loop running at control_rate Hz"""
        if not self.is_running:
            return
        
        try:
            # Update observation buffers
            self._update_observation_buffers()
            
            # Check if we need new actions
            if len(self.action_queue) == 0:
                self._run_inference()
            
            # Get next action from queue
            if len(self.action_queue) > 0:
                action = self.action_queue.popleft()
                
                # Send joint command
                self._send_joint_command(action)
                
                # Log periodically
                if self.inference_count % 10 == 0 and len(self.action_queue) == self.chunk_size - 1:
                    self.get_logger().info(
                        f"🤖 Inference #{self.inference_count} | "
                        f"Action[0:3]: [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}]"
                    )
                    
        except Exception as e:
            self.get_logger().error(f"Inference error: {e}")
            import traceback
            traceback.print_exc()
            self.stop_control()
    
    def _send_joint_command(self, action: np.ndarray):
        """Send joint command to robot"""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.name = list(self.joint_names)
        msg.position = action.tolist()
        msg.velocity = [0.0] * len(self.joint_names)
        msg.effort = [0.0] * len(self.joint_names)
        
        self.joint_cmd_pub.publish(msg)
    
    def go_home(self, duration_sec: float = 3.0):
        """Smoothly move robot to home position before starting control"""
        self.get_logger().info("🏠 Moving to home position...")
        
        # Get home positions as numpy array
        home_action = np.array([self.home_positions[name] for name in self.joint_names], dtype=np.float32)
        
        # Send home position commands for duration_sec at control rate
        steps = int(duration_sec * self.control_rate)
        for i in range(steps):
            self._send_joint_command(home_action)
            time.sleep(1.0 / self.control_rate)
        
        self.get_logger().info("🏠 Home position reached!")
    
    def start_control(self):
        """Start the control loop"""
        # First, go to home position
        self.go_home(duration_sec=3.0)
        
        # Reset buffers
        self.state_buffer.clear()
        self.image_buffer.clear()
        self.action_queue.clear()
        
        for _ in range(self.n_obs_steps):
            self.state_buffer.append(np.zeros(17, dtype=np.float32))
            self.image_buffer.append(np.zeros((3, self.vid_H, self.vid_W), dtype=np.float32))
        
        # Reset policy's internal state
        self.policy.reset()
        
        self.is_running = True
        self.inference_count = 0
        self.get_logger().info("▶️  Control STARTED")
    
    def stop_control(self):
        """Stop the control loop"""
        self.is_running = False
        self.get_logger().info("⏸️  Control STOPPED")
    
    def toggle_control(self):
        """Toggle control on/off"""
        if self.is_running:
            self.stop_control()
        else:
            self.start_control()


def main():
    # Model path - ACT model
    MODEL_PATH = "/home/beable/lerobot/outputs/train/act_dexterous_chunk100/checkpoints/020000/pretrained_model"
    
    rclpy.init()
    
    # Create controller node
    controller = ACTPolicyController(model_path=MODEL_PATH)
    
    # Spin ROS2 in background
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(controller)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    
    print("\n" + "="*60)
    print("🧠 ACT POLICY CONTROLLER")
    print("="*60)
    print("\nCommands:")
    print("  h      - Go to home position")
    print("  ENTER  - Start/stop control (goes to home first)")
    print("  q      - Quit")
    print("\n" + "-"*60)
    
    # Wait a bit for first messages
    time.sleep(1.0)
    
    try:
        while rclpy.ok():
            user_input = input("\n> Command (h=home, ENTER=toggle, q=quit): ").strip().lower()
            
            if user_input == 'q':
                print("\nExiting...")
                break
            elif user_input == 'h':
                controller.go_home()
            else:
                controller.toggle_control()
    
    except KeyboardInterrupt:
        print("\n\nStopped.")
    
    finally:
        controller.stop_control()
        try:
            rclpy.shutdown()
        except Exception:
            pass  # Ignore if already shutdown
        executor_thread.join(timeout=1.0)


if __name__ == '__main__':
    main()
