#!/usr/bin/env python3
"""
Diffusion Policy Controller for R1 Humanoid Robot
Loads a trained diffusion model and controls the robot like teleop.

Usage:
    conda activate isaaclab  # or lerobot environment
    python3 diffusion_policy_control.py
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
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.constants import ACTION, OBS_IMAGES, OBS_STATE


class DiffusionPolicyController(Node):
    """ROS2 Node that controls robot using trained Diffusion Policy"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        super().__init__('diffusion_policy_controller')
        
        self.device = device
        self.bridge = CvBridge()
        
        # Video dimensions (must match training - from data_collection_keyboard.py)
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
        self.get_logger().info(f"🧠 Loading Diffusion Policy from: {model_path}")
        self._load_model(model_path)
        
        # Policy config
        self.n_obs_steps = self.policy.config.n_obs_steps
        self.n_action_steps = self.policy.config.n_action_steps
        self.horizon = self.policy.config.horizon
        
        # Our own observation and action queues (bypass select_action's buggy handling)
        self.state_queue = deque(maxlen=self.n_obs_steps)
        self.image_queue = deque(maxlen=self.n_obs_steps)
        self.action_queue = deque(maxlen=self.n_action_steps)
        
        # Initialize queues with zeros
        for _ in range(self.n_obs_steps):
            self.state_queue.append(torch.zeros(1, 17, device=self.device))
            self.image_queue.append(torch.zeros(1, 1, 3, self.vid_H, self.vid_W, device=self.device))
        
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
        
        self.get_logger().info("✅ Diffusion Policy Controller initialized!")
        self.get_logger().info(f"   n_obs_steps: {self.n_obs_steps}, n_action_steps: {self.n_action_steps}")
        self.get_logger().info("   Press ENTER to start/stop control, 'q' to quit")
    
    def _load_model(self, model_path: str):
        """Load the pretrained diffusion policy"""
        try:
            self.policy = DiffusionPolicy.from_pretrained(model_path)
            self.policy.to(self.device)
            self.policy.eval()
            
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
    
    def _prepare_observation(self):
        """Prepare current observation tensors and add to queues
        
        Returns current observation as tensors:
        - state: (1, 17)
        - image: (1, 1, 3, H, W) - (batch, num_cameras, C, H, W)
        """
        # Get current state (17 joint positions)
        state = np.array([self.current_positions[name] for name in self.joint_names], dtype=np.float32)
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)  # (1, 17)
        
        # Get current image (BGR -> RGB, HWC -> CHW, normalize to 0-1)
        image = self.current_rgb_image[:, :, ::-1].copy()  # BGR -> RGB
        image = image.transpose(2, 0, 1).astype(np.float32) / 255.0  # HWC -> CHW, 0-1
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(1).to(self.device)  # (1, 1, 3, H, W)
        
        # Add to queues
        self.state_queue.append(state_tensor)
        self.image_queue.append(image_tensor)
        
        return state_tensor, image_tensor
    
    def _build_batch_from_queues(self):
        """Build a batch from observation queues for diffusion model
        
        Returns batch with shapes:
        - observation.state: (1, n_obs_steps, 17)
        - observation.images: (1, n_obs_steps, num_cameras, C, H, W)
        """
        # Stack state queue: list of (1, 17) -> (1, n_obs_steps, 17)
        states = torch.cat(list(self.state_queue), dim=0)  # (n_obs_steps, 17)
        states = states.unsqueeze(0)  # (1, n_obs_steps, 17)
        
        # Stack image queue: list of (1, 1, 3, H, W) -> (1, n_obs_steps, 1, 3, H, W)
        images = torch.cat(list(self.image_queue), dim=0)  # (n_obs_steps, 1, 3, H, W)
        images = images.unsqueeze(0)  # (1, n_obs_steps, 1, 3, H, W)
        
        batch = {
            OBS_STATE: states,
            OBS_IMAGES: images,
        }
        
        return batch
    
    @torch.no_grad()
    def _run_inference(self):
        """Run diffusion policy inference with manual queue management
        
        Bypasses select_action to avoid dimension issues.
        Manually handles:
        - Observation queuing
        - Normalization
        - Action generation
        - Action chunking
        - Unnormalization
        """
        # Update observation queues
        self._prepare_observation()
        
        # If we have actions in queue, use them
        if len(self.action_queue) > 0:
            action = self.action_queue.popleft()
            self.inference_count += 1
            return action.cpu().numpy()
        
        # Need to generate new action chunk
        batch = self._build_batch_from_queues()
        
        # Normalize inputs
        batch = self.policy.normalize_inputs(batch)
        
        # Generate actions using diffusion model
        actions = self.policy.diffusion.generate_actions(batch)
        
        # Unnormalize actions
        actions = self.policy.unnormalize_outputs({ACTION: actions})[ACTION]
        
        # actions shape: (1, horizon, 17)
        actions = actions.squeeze(0)  # (horizon, 17)
        
        # Take n_action_steps actions and put in queue
        for i in range(self.n_action_steps):
            if i < len(actions):
                self.action_queue.append(actions[i])
        
        # Return first action
        action = self.action_queue.popleft()
        self.inference_count += 1
        return action.cpu().numpy()
    
    def control_loop(self):
        """Main control loop running at control_rate Hz"""
        if not self.is_running:
            return
        
        try:
            # Run inference - select_action handles observation history and action chunking internally
            action = self._run_inference()
            
            # Send joint command
            self._send_joint_command(action)
            
            # Log periodically
            if self.inference_count % 10 == 0:
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
        
        # Reset our observation and action queues
        self.state_queue.clear()
        self.image_queue.clear()
        self.action_queue.clear()
        
        # Initialize observation queues with zeros
        for _ in range(self.n_obs_steps):
            self.state_queue.append(torch.zeros(1, 17, device=self.device))
            self.image_queue.append(torch.zeros(1, 1, 3, self.vid_H, self.vid_W, device=self.device))
        
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
    # Model path
    MODEL_PATH = "/home/beable/lerobot/outputs/train/diffusion_dexterous_box_40ep_finetuned/checkpoints/100000/pretrained_model"
    
    rclpy.init()
    
    # Create controller node
    controller = DiffusionPolicyController(model_path=MODEL_PATH)
    
    # Spin ROS2 in background
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(controller)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    
    print("\n" + "="*60)
    print("🧠 DIFFUSION POLICY CONTROLLER")
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
