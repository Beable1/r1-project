#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy, Image, JointState
from geometry_msgs.msg import TwistStamped
import threading
import copy
import time
import cv2
import os
import numpy as np
from cv_bridge import CvBridge
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sys
import termios
import tty
import subprocess
import select

import pinocchio as pin

class IKSolver:
    def __init__(self, urdf_path, ee_joint_name, controlled_joints):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.ee_id = self.model.getJointId(ee_joint_name)
        self.controlled_joints = controlled_joints
        
        self.q_idx = []
        self.v_idx = []
        for name in self.controlled_joints:
            if self.model.existJointName(name):
                j_id = self.model.getJointId(name)
                self.q_idx.append(self.model.joints[j_id].idx_q)
                self.v_idx.append(self.model.joints[j_id].idx_v)

        # Shoulder yaw limit override: [-1.1, 1.1] rad
        # Joint name in URDF: "right_shoulder_link_joint"
        try:
            if self.model.existJointName("right_shoulder_link_joint"):
                j_id = self.model.getJointId("right_shoulder_link_joint")
                idx_q = self.model.joints[j_id].idx_q
                if idx_q < self.model.nq:
                    self.model.lowerPositionLimit[idx_q] = -1.1
                    self.model.upperPositionLimit[idx_q] = 1.1
        except Exception:
            # Güvenli tarafta kal: limit ayarı başarısız olursa default limitler kullanılsın
            pass
                
    def solve(self, q_current_full, target_pos, target_rot=None, max_iter=50, eps=1e-3):
        q = np.copy(q_current_full)
        target_se3 = pin.SE3(target_rot if target_rot is not None else np.eye(3), target_pos)
        damp = 1e-4
        dt = 0.5
        
        for i in range(max_iter):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            
            iMd = self.data.oMi[self.ee_id].actInv(target_se3)
            err = pin.log(iMd).vector
            
            if np.linalg.norm(err) < eps:
                break
            
            J = pin.computeJointJacobian(self.model, self.data, q, self.ee_id)
            J_masked = np.zeros_like(J)
            for vj in self.v_idx:
                J_masked[:, vj] = J[:, vj]
                
            v = J_masked.T @ np.linalg.solve(J_masked @ J_masked.T + damp * np.eye(6), err)
            q = pin.integrate(self.model, q, v * dt)
            q = np.clip(q, self.model.lowerPositionLimit, self.model.upperPositionLimit)
            
        return q

# Mouse control
try:
    from pynput import mouse
    from pynput.mouse import Controller as MouseController
    MOUSE_AVAILABLE = True
    mouse_controller = MouseController()
except ImportError:
    print("⚠ pynput module not found. Mouse control disabled. To install: pip install pynput")
    MOUSE_AVAILABLE = False
    mouse_controller = None

# Get screen size (for mouse locking)
try:
    import subprocess
    result = subprocess.run(['xdpyinfo'], capture_output=True, text=True)
    for line in result.stdout.split('\n'):
        if 'dimensions' in line:
            dims = line.split()[1].split('x')
            SCREEN_WIDTH = int(dims[0])
            SCREEN_HEIGHT = int(dims[1])
            break
    else:
        SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080
except:
    SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080

bridge = CvBridge()

# Mouse control variables
mouse_control_enabled = False
mouse_sensitivity = 0.002  # Mouse movement sensitivity (radians/pixel) - faster
mouse_last_x = 0
mouse_last_y = 0
mouse_delta_x = 0
mouse_delta_y = 0
mouse_left_button = False
mouse_right_button = False
mouse_middle_button = False

record_data = False
vid_H = 360
vid_W = 640
wrist_camera_image = np.zeros((vid_H, vid_W, 3), np.uint8)
top_camera_image = np.zeros((vid_H, vid_W, 3), np.uint8)
rgb_image = np.zeros((vid_H, vid_W, 3), np.uint8)  # RGB image from /rgb topic
action = np.array([0.0] * 17, float)  # Joint positions as action (17 joints for right arm)
joint_states = {'names': [], 'positions': np.array([], float)}  # Joint states from /joint_states topic


class JointCommand_Subscriber(Node):
    """Subscribe to /joint_command topic to get action (joint positions)"""
    
    def __init__(self):
        super().__init__('joint_command_subscriber')
        self.subscription = self.create_subscription(
            JointState,
            '/joint_command',
            self.joint_command_callback,
            10)
        self.subscription
    
    def joint_command_callback(self, data):
        global action
        # Store joint positions as action
        # Expected joint order: right_shoulder_link_joint, right_arm_top_link_joint, etc.
        if len(data.position) > 0:
            action = np.array(data.position, dtype=float)
        else:
            # If no positions, keep current action (or set to zeros)
            action = np.array([0.0] * len(action), dtype=float) if len(action) > 0 else np.array([0.0] * 17, dtype=float)

class RobotArmController(Node):
    """Robot Arm Controller from control_robot_arms.py - unchanged"""
    def __init__(self):
        super().__init__('robot_arm_controller')
        
        # Topic aligned with control_right_arm.py (/joint_command)
        topic_name = self.declare_parameter('topic_name', '/joint_command').value
        self.publisher = self.create_publisher(JointState, topic_name, 10)
        
        # Robot joint names - must match JointNameArray in rod_graph.usda
        # Total 43 joints
        self.joint_names = [
            "Chest_link_joint",
            "hips_l",
            "hips_r",
            "wraist_link_joint",
            "hip_roll_l",
            "hip_roll_r",
            "head_jointt",
            "left_shoulder_link_joint",
            "right_shoulder_link_joint",
            "legs_l",
            "legs_r",
            "left_arm_top_link_joint",
            "right_arm_top_link_joint",
            "Feet_l",
            "Feet_r",
            "left_arm_bottom_link_joint",
            "right_arm_bottom_link_joint",
            "left_forearm_link_joint",
            "right_forearm_link_joint",
            "wrist_roll_joint_l",
            "wrist_roll_joint_r",
            "wrist_pitch_joint_l",
            "wrist_pitch_joint_r",
            "index_proximal_joint_l",
            "little_proximal_joint_l",
            "middle_proximal_joint_l",
            "ring_proximal_joint_l",
            "thumb_proximal_joint_l",
            "index_proximal_joint_r",
            "little_proximal_joint_r",
            "middle_proximal_joint_r",
            "ring_proximal_joint_r",
            "thumb_proximal_joint_r",
            "index_proximal_joint_l_1",
            "little_proximal_joint_l__",
            "middle_proximal_joint_l_1",
            "ring_proximal_joint_l_1",
            "thumb_proximal_joint_l_1",
            "index_proximal_joint_r_1",
            "little_proximal_joint_r_1",
            "middle_proximal_joint_r_1",
            "ring_proximal_joint_r_1",
            "thumb_proximal_joint_r_1"
        ]
        
        self.get_logger().info(f'Robot Arm Controller started')
        self.get_logger().info(f'Topic: {topic_name}')
        self.get_logger().info(f'Number of joints: {len(self.joint_names)}')
        
    def send_command(self, positions, velocities=None, duration_sec=3.0):
        """
        Send Joint command.
        
        Args:
            positions: List or dict. Joint positions (radians)
            velocities: List or dict. Joint velocities (radians/s) - optional
            duration_sec: Movement duration (seconds)
        """
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.name = self.joint_names
        
        # Prepare positions
        if isinstance(positions, dict):
            pos_list = [positions.get(name, 0.0) for name in self.joint_names]
        else:
            pos_list = list(positions) if len(positions) >= len(self.joint_names) else list(positions) + [0.0] * (len(self.joint_names) - len(positions))
            pos_list = pos_list[:len(self.joint_names)]
        
        # Velocities
        if velocities is None:
            vel_list = [0.0] * len(self.joint_names)
        elif isinstance(velocities, dict):
            vel_list = [velocities.get(name, 0.0) for name in self.joint_names]
        else:
            vel_list = list(velocities) if len(velocities) >= len(self.joint_names) else list(velocities) + [0.0] * (len(self.joint_names) - len(velocities))
            vel_list = vel_list[:len(self.joint_names)]

        msg.position = pos_list
        msg.velocity = vel_list
        msg.effort = [0.0] * len(self.joint_names)
        
        # Publish
        self.publisher.publish(msg)
        self.get_logger().info(f'Command sent: {pos_list[:5]}... (first 5 joints)')
        return msg
    
    def move_joint(self, joint_name, position, duration_sec=2.0):
        """Move a specific joint"""
        if joint_name not in self.joint_names:
            self.get_logger().error(f'Joint not found: {joint_name}')
            self.get_logger().info(f'Available joints: {self.joint_names}')
            return False
        
        positions = {joint_name: float(position)}
        self.send_command(positions, duration_sec=duration_sec)
        return True
    
    def move_all(self, positions, duration_sec=3.0):
        """Move all joints"""
        self.send_command(positions, duration_sec=duration_sec)
    
    def home(self, duration_sec=3.0):
        """Reset all joints to zero position"""
        positions = [0.0] * len(self.joint_names)
        self.send_command(positions, duration_sec=duration_sec)
        self.get_logger().info('Moving to Home position...')

def interactive_control(controller):
    """Interactive control loop - from control_robot_arms.py - unchanged"""
    print("\n" + "="*60)
    print("ROBOT ARM CONTROL")
    print("="*60)
    print("\nCommands:")
    print("  set <pos1> <pos2> ...     -> Set all joints to specified positions")
    print("  move <joint_name> <pos>   -> Move a specific joint")
    print("  home                     -> Reset all joints to zero position")
    print("  list                     -> List joint names")
    print("  quit                     -> Exit")
    print("\n" + "-"*60 + "\n")
    
    while rclpy.ok():
        try:
            command = input("Command> ").strip()
            
            if not command:
                continue
            
            parts = command.split()
            cmd = parts[0].lower()
            
            if cmd == 'quit' or cmd == 'exit' or cmd == 'q':
                print("Exiting...")
                break
            
            elif cmd == 'home':
                controller.home()
            
            elif cmd == 'list':
                print("\nJoint Names:")
                for i, name in enumerate(controller.joint_names):
                    print(f"  [{i}] {name}")
                print()
            
            elif cmd == 'set':
                if len(parts) < 2:
                    print("Usage: set <pos1> <pos2> ...")
                    continue
                try:
                    positions = [float(p) for p in parts[1:]]
                    controller.move_all(positions)
                except ValueError as e:
                    print(f"Error: Invalid position value - {e}")
            
            elif cmd == 'move':
                if len(parts) < 3:
                    print("Usage: move <joint_name> <position>")
                    continue
                try:
                    joint_name = parts[1]
                    position = float(parts[2])
                    controller.move_joint(joint_name, position)
                except ValueError as e:
                    print(f"Error: Invalid position value - {e}")
            
            else:
                print(f"Unknown command: {cmd}")
                print("Commands: set, move, home, list, quit")
        
        except EOFError:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def get_key():
    """Get single key press (non-blocking)"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def get_all_keys():
    """Get all simultaneously pressed keys (non-blocking, multi-key support)"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    keys = []
    try:
        tty.setraw(sys.stdin.fileno())
        # İlk tuşu bekle (blocking)
        ch = sys.stdin.read(1)
        keys.append(ch)
        # Read other keys arriving within a short duration (non-blocking)
        while True:
            if select.select([sys.stdin], [], [], 0.01)[0]:
                ch = sys.stdin.read(1)
                keys.append(ch)
            else:
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return keys


class Robot_Keyboard_Controller(Node):

    def __init__(self):
        super().__init__('robot_keyboard_controller')
        
        # Robot control publisher - Same as control_right_arm.py
        topic_name = self.declare_parameter('topic_name', '/joint_command').value
        self.publisher = self.create_publisher(JointState, topic_name, 10)
        
        # Right arm and finger joints (same as control_right_arm.py)
        self.joint_names = [
            "right_shoulder_link_joint",      # Omuz
            "right_arm_top_link_joint",        # Üst kol
            "right_arm_bottom_link_joint",     # Alt kol
            "right_forearm_link_joint",        # Ön kol
            "wrist_pitch_joint_r",             # Bilek pitch
            "wrist_roll_joint_r",               # Bilek roll
            "thumb_joint_roll_r",               # Baş parmak roll
            "index_proximal_joint_r",          # İşaret parmağı 1
            "middle_proximal_joint_r",         # Orta parmak 1
            "ring_proximal_joint_r",           # Yüzük parmağı 1
            "little_proximal_joint_r",          # Serçe parmak 1
            "thumb_proximal_joint_r",           # Baş parmak 1
            "index_proximal_joint_r_1",         # İşaret parmağı 2
            "middle_proximal_joint_r_1",        # Orta parmak 2
            "ring_proximal_joint_r_1",          # Yüzük parmağı 2
            "little_proximal_joint_r_1",       # Serçe parmak 2
            "thumb_proximal_joint_r_1"          # Baş parmak 2
        ]
        
        # Home positions (natural standing position of the robot)
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
        
        # Current positions (start from zero - like control_right_arm.py)
        self.current_positions = {name: 0.0 for name in self.joint_names}

        self.ik_solver = None
        try:
            urdf_path = "/home/beable/IsaacLab/IsaacLab/r1-new/urdf/r1-new.urdf"
            self.ik_solver = IKSolver(urdf_path, "wrist_pitch_joint_r", [
                "right_shoulder_link_joint",
                "right_arm_top_link_joint",
                "right_arm_bottom_link_joint",
                "right_forearm_link_joint",
                "wrist_roll_joint_r",
                "wrist_pitch_joint_r"
            ])
            self.use_ik = True
            
            # Start neutral
            q = pin.neutral(self.ik_solver.model)
            pin.forwardKinematics(self.ik_solver.model, self.ik_solver.data, q)
            self.target_se3 = pin.SE3(self.ik_solver.data.oMi[self.ik_solver.ee_id].rotation.copy(), 
                                      self.ik_solver.data.oMi[self.ik_solver.ee_id].translation.copy())
        except Exception as e:
            self.get_logger().error(f"IK Init failed: {e}")
            self.use_ik = False
        
        self.cart_step = 0.01
        self.rot_step = 0.05

        # Target positions: for non-blocking smooth movement (all joints)
        self.target_positions = self.current_positions.copy()
        
        # Step size (radians) - movement amount per key
        self.step_size = 0.01
        
        # Maximum velocity (radians/second) - fast response for mouse
        self.max_velocity = 4.0  # 3.0 rad/s - daha hızlı!
        self.control_rate = 200  # Hz - 200Hz kontrol döngüsü (düşük gecikme)
        self.smooth_step = self.max_velocity / self.control_rate  # Her adımda maksimum hareket

        # Fingers can be slightly faster: separate speed limit
        self.finger_max_velocity = 1.5  # rad/s - faster finger movement
        self.finger_smooth_step = self.finger_max_velocity / self.control_rate

        # Effort/Force limits (Newton-meters)
        # Low torque for fingers -> doesn't crush object, doesn't throw
        self.finger_max_effort = 5.0   # Nm - for fingers (keep low!)
        self.arm_max_effort = 50.0     # Nm - for arm joints

        # Non-blocking motion updater (200Hz): current_positions -> target_positions
        self._motion_timer = self.create_timer(1.0 / self.control_rate, self._motion_update)
        
        # Finger joints
        self.finger_joints = [
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
            "thumb_joint_roll_r",
        ]
        
        # Finger positions (3 stages)
        self.finger_full_open = 0.1
        self.finger_half_closed = 1.0
        self.finger_full_closed = 3.0
        self.finger_state = 0

        # Thumb roll toggle (single key open/close)
        # Note: thumb_joint_roll_r also moves manually with (c/v) +/-.
        # Toggle goes to non-blocking target.
        self.thumb_roll_open = -0.3
        self.thumb_roll_closed = 0.3
        self.thumb_roll_is_closed = False

        # Thumb-first / gating status
        self._finger_gate_active = False
        self._finger_gate_target = None
        self._finger_gate_other_joints = []
        
        # Keyboard mapping (same as control_right_arm.py)
        self.key_mappings = {
            # Shoulder
            's': ("right_shoulder_link_joint", +self.step_size),
            'd': ("right_shoulder_link_joint", -self.step_size),
            'a': ("right_arm_top_link_joint", +self.step_size),
            'w': ("right_arm_top_link_joint", -self.step_size),
            
            # Elbow
            'e': ("right_arm_bottom_link_joint", +self.step_size),
            'r': ("right_arm_bottom_link_joint", -self.step_size),
            'f': ("right_forearm_link_joint", +self.step_size),
            't': ("right_forearm_link_joint", -self.step_size),
            
            # Wrist
            'q': ("wrist_pitch_joint_r", +self.step_size),
            'y': ("wrist_pitch_joint_r", -self.step_size),
            'z': ("wrist_roll_joint_r", +self.step_size),
            'x': ("wrist_roll_joint_r", -self.step_size),

            # Thumb roll (3x faster)
            'c': ("thumb_joint_roll_r", +self.step_size * 3),
            'v': ("thumb_joint_roll_r", -self.step_size * 3),
        }
        
        # Mouse control settings
        self.mouse_enabled = False
        self.mouse_sensitivity = 0.002  # radians/pixel - faster control
        self.mouse_mode = 0  # 0: shoulder, 1: elbow, 2: wrist
        self.mouse_mode_names = ['Shoulder', 'Elbow', 'Wrist']
        
        # Joints to be controlled by mouse (per mode)
        self.mouse_joint_mappings = {
            0: {  # Shoulder mode
                'x': 'right_shoulder_link_joint',  # Horizontal movement
                'y': 'right_arm_top_link_joint',   # Vertical movement
            },
            1: {  # Elbow mode
                'x': 'right_arm_bottom_link_joint',
                'y': 'right_forearm_link_joint',
            },
            2: {  # Wrist mode
                'x': 'wrist_roll_joint_r',
                'y': 'wrist_pitch_joint_r',
            },
        }
        
        # Start mouse listener (if pynput exists)
        self.mouse_listener = None
        if MOUSE_AVAILABLE:
            self._setup_mouse_listener()
        
        self.get_logger().info(f'Robot Keyboard Controller started')
        self.get_logger().info(f'Topic: {topic_name}')
        self.get_logger().info(f'Number of joints: {len(self.joint_names)}')
        if MOUSE_AVAILABLE:
            self.get_logger().info('🖱️  Mouse control ready (Toggle with M)')
        
        # Go to home position at start
        self.send_command()
    
    def _setup_mouse_listener(self):
        """Start mouse listener"""
        global mouse_delta_x, mouse_delta_y, mouse_left_button, mouse_right_button, mouse_middle_button
        
        def on_move(x, y):
            global mouse_last_x, mouse_last_y, mouse_delta_x, mouse_delta_y
            if self.mouse_enabled:
                # Calculate delta from lock position (NOT cumulative)
                if hasattr(self, 'lock_position') and self.lock_position and mouse_controller:
                    # Delta = current pos - lock pos (direct, no accumulation)
                    mouse_delta_x = x - self.lock_position[0]
                    mouse_delta_y = y - self.lock_position[1]
                    # Send cursor back
                    mouse_controller.position = self.lock_position
                    # Set Last position to lock position
                    mouse_last_x, mouse_last_y = self.lock_position
                    return
                else:
                    mouse_delta_x = x - mouse_last_x
                    mouse_delta_y = y - mouse_last_y
            mouse_last_x = x
            mouse_last_y = y
        
        def on_click(x, y, button, pressed):
            global mouse_left_button, mouse_right_button, mouse_middle_button
            if button == mouse.Button.left:
                mouse_left_button = pressed
            elif button == mouse.Button.right:
                mouse_right_button = pressed
            elif button == mouse.Button.middle:
                mouse_middle_button = pressed
        
        def on_scroll(x, y, dx, dy):
            # Adjust sensitivity with Scroll
            global mouse_sensitivity
            if self.mouse_enabled:
                if dy > 0:
                    self.mouse_sensitivity = min(0.005, self.mouse_sensitivity * 1.2)
                else:
                    self.mouse_sensitivity = max(0.00005, self.mouse_sensitivity / 1.2)
                print(f"🖱️  Mouse sensitivity: {self.mouse_sensitivity:.5f}")
        
        try:
            self.mouse_listener = mouse.Listener(
                on_move=on_move,
                on_click=on_click,
                on_scroll=on_scroll
            )
            self.mouse_listener.start()
            self.get_logger().info('🖱️  Mouse listener started')
        except Exception as e:
            self.get_logger().warning(f'Mouse listener could not start: {e}')
            self.mouse_listener = None
    
    def toggle_mouse_control(self):
        """Toggle mouse control - cursor locked to bottom right when active"""
        global mouse_delta_x, mouse_delta_y
        self.mouse_enabled = not self.mouse_enabled
        mouse_delta_x = 0
        mouse_delta_y = 0
        
        if self.mouse_enabled:
            # İmleci sağ alt köşeye taşı
            if mouse_controller:
                self.lock_position = (SCREEN_WIDTH - 50, SCREEN_HEIGHT - 50)
                mouse_controller.position = self.lock_position
            print(f"\n🖱️  MOUSE CONTROL ACTIVE - Mode: {self.mouse_mode_names[self.mouse_mode]}")
            print(f"    🔒 Cursor locked to bottom right ({SCREEN_WIDTH-50}, {SCREEN_HEIGHT-50})")
            print("    Left Click: Close fingers | Right Click: Open fingers | Middle Click: Change mode")
            print("    Scroll: Adjust sensitivity | M: Disable mouse")
        else:
            self.lock_position = None
            print("\n⌨️  Klavye kontrolüne dönüldü - İmleç serbest")
    
    def cycle_mouse_mode(self):
        """Change mouse control mode (Shoulder -> Elbow -> Wrist)"""
        self.mouse_mode = (self.mouse_mode + 1) % 3
        print(f"🖱️  Mouse mode: {self.mouse_mode_names[self.mouse_mode]}")
    
    def process_mouse_input(self):
        """FPS STYLE Mouse control - USE REAL SIMULATION POSITION
        
        NO accumulation because:
        - Read ACTUAL simulation position at every command (/joint_states)
        - Add delta to that position
        - If simulation lags, it doesn't matter, we always start from current position
        """
        global mouse_delta_x, mouse_delta_y, mouse_left_button, mouse_right_button, mouse_middle_button
        global joint_states
        
        if not self.mouse_enabled:
            return
        
        # Get Delta and RESET IMMEDIATELY
        dx = mouse_delta_x
        dy = mouse_delta_y
        mouse_delta_x = 0
        mouse_delta_y = 0
        
        # Get ACTUAL positions from simulation
        actual_positions = {}
        if len(joint_states['names']) > 0 and len(joint_states['positions']) > 0:
            actual_positions = dict(zip(joint_states['names'], joint_states['positions']))
        
        # Movement mapping
        mapping = self.mouse_joint_mappings.get(self.mouse_mode, {})
        
        # X axis
        if 'x' in mapping and abs(dx) > 0:
            joint_name = mapping['x']
            # Get ACTUAL position (from simulation)
            if joint_name in actual_positions:
                real_pos = actual_positions[joint_name]
            else:
                real_pos = self.current_positions.get(joint_name, 0)
            
            # Calculate Delta and limit (FPS sensitivity)
            max_move = 0.1  # radians - max per single frame
            delta = dx * self.mouse_sensitivity
            delta = max(-max_move, min(max_move, delta))
            
            # New target = ACTUAL position + delta
            new_pos = real_pos + delta
            self.current_positions[joint_name] = new_pos
            self.target_positions[joint_name] = new_pos
        
        # Y axis
        if 'y' in mapping and abs(dy) > 0:
            joint_name = mapping['y']
            if joint_name in actual_positions:
                real_pos = actual_positions[joint_name]
            else:
                real_pos = self.current_positions.get(joint_name, 0)
            
            max_move = 0.1
            delta = -dy * self.mouse_sensitivity
            delta = max(-max_move, min(max_move, delta))
            
            new_pos = real_pos + delta
            self.current_positions[joint_name] = new_pos
            self.target_positions[joint_name] = new_pos
        
        # Left click: close fingers
        if mouse_left_button:
            for joint in self.finger_joints:
                if joint != 'thumb_joint_roll_r':
                    if joint in actual_positions:
                        real_pos = actual_positions[joint]
                    else:
                        real_pos = self.current_positions.get(joint, 0)
                    if real_pos < self.finger_full_closed:
                        new_pos = min(real_pos + 0.15, self.finger_full_closed)
                        self.current_positions[joint] = new_pos
                        self.target_positions[joint] = new_pos
        
        # Right click: open fingers
        if mouse_right_button:
            for joint in self.finger_joints:
                if joint != 'thumb_joint_roll_r':
                    if joint in actual_positions:
                        real_pos = actual_positions[joint]
                    else:
                        real_pos = self.current_positions.get(joint, 0)
                    if real_pos > self.finger_full_open:
                        new_pos = max(real_pos - 0.15, self.finger_full_open)
                        self.current_positions[joint] = new_pos
                        self.target_positions[joint] = new_pos
        
        # Send command
        self.send_command()
    
    def toggle_mouse_control(self):
        """Toggle mouse control - cursor locked to bottom right when active"""
        global mouse_delta_x, mouse_delta_y
        self.mouse_enabled = not self.mouse_enabled
        mouse_delta_x = 0
        mouse_delta_y = 0
        
        if self.mouse_enabled:
            # Reset base positions (new session)
            if hasattr(self, '_mouse_base_positions'):
                del self._mouse_base_positions
            
            # Move cursor to bottom right
            if mouse_controller:
                self.lock_position = (SCREEN_WIDTH - 50, SCREEN_HEIGHT - 50)
                mouse_controller.position = self.lock_position
            print(f"\n🖱️  MOUSE CONTROL ACTIVE - Mode: {self.mouse_mode_names[self.mouse_mode]}")
            print(f"    🔒 Cursor locked to bottom right ({SCREEN_WIDTH-50}, {SCREEN_HEIGHT-50})")
            print("    Left Click: Close fingers | Right Click: Open fingers | Middle Click: Change mode")
            print("    Scroll: Adjust sensitivity | M: Disable mouse")
        else:
            self.lock_position = None
            print("\n⌨️  Returned to keyboard control - Cursor free")
    
    def _motion_update(self):
        """Non-blocking movement: approach target with max_velocity and send command."""
        # Process Mouse input
        self.process_mouse_input()
        
        moved = False
        for joint_name in self.joint_names:
            current = self.current_positions[joint_name]
            target = self.target_positions.get(joint_name, current)
            diff = target - current

            if abs(diff) > 0.001:
                # Apply separate speed limit for fingers
                max_step = self.finger_smooth_step if joint_name in self.finger_joints else self.smooth_step
                step = max_step if abs(diff) > max_step else abs(diff)
                self.current_positions[joint_name] = current + (step if diff > 0 else -step)
                moved = True
            else:
                # Snap to exact target if very close
                self.current_positions[joint_name] = target

        # If Thumb reached target, activate other finger targets
        if self._finger_gate_active and self._finger_gate_target is not None:
            if self._thumb_at_target(self._finger_gate_target):
                if self._finger_gate_other_joints:
                    self.move_smooth({j: self._finger_gate_target for j in self._finger_gate_other_joints})
                self._finger_gate_active = False
                self._finger_gate_target = None
                self._finger_gate_other_joints = []

        # Don't publish continuously: only send when there is movement
        if moved:
            self.send_command()

    def _thumb_at_target(self, target, tol=0.03):
        """Are thumb joints close enough to target position?"""
        thumb_joints = [
            'thumb_joint_roll_r',
            'thumb_proximal_joint_r',
            'thumb_proximal_joint_r_1',
        ]
        for j in thumb_joints:
            if j not in self.current_positions:
                continue
            if abs(self.current_positions[j] - float(target)) > tol:
                return False
        return True
        

    def update_cartesian(self, delta_pos, delta_rot=None):
        if not getattr(self, "use_ik", False): return
        
        # Build q from current targets so we don't jump
        q = pin.neutral(self.ik_solver.model)
        for name, p in self.target_positions.items():
            if self.ik_solver.model.existJointName(name):
                j_id = self.ik_solver.model.getJointId(name)
                idx_q = self.ik_solver.model.joints[j_id].idx_q
                if idx_q < self.ik_solver.model.nq:
                    q[idx_q] = p
                    
        pin.forwardKinematics(self.ik_solver.model, self.ik_solver.data, q)
        pin.updateFramePlacements(self.ik_solver.model, self.ik_solver.data)
        self.target_se3 = pin.SE3(self.ik_solver.data.oMi[self.ik_solver.ee_id].rotation.copy(), 
                                  self.ik_solver.data.oMi[self.ik_solver.ee_id].translation.copy())
                                  
        if delta_pos is not None:
            self.target_se3.translation += np.array(delta_pos)
            
        if delta_rot is not None:
            r, p, y = delta_rot
            Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
            Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
            Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
            dR = Rz @ Ry @ Rx
            self.target_se3.rotation = self.target_se3.rotation @ dR
            
        q_res = self.ik_solver.solve(q, self.target_se3.translation, self.target_se3.rotation.copy())
        
        for name in self.ik_solver.controlled_joints:
            j_id = self.ik_solver.model.getJointId(name)
            idx_q = self.ik_solver.model.joints[j_id].idx_q
            self.target_positions[name] = float(q_res[idx_q])

    def update_position(self, joint_name, delta):
        """Update joint position and send command"""
        global action
        if joint_name not in self.current_positions:
            print(f'ERROR: Unknown joint: {joint_name}')
            return
        
        # Update position (step_size per key) -> update target (non-blocking)
        self.target_positions[joint_name] = self.target_positions.get(joint_name, self.current_positions[joint_name]) + delta

    def move_smooth(self, target_positions_dict):
        """(Non-blocking) update target positions. Movement happens via timer."""
        for joint_name, target in target_positions_dict.items():
            if joint_name in self.target_positions:
                self.target_positions[joint_name] = float(target)
    
    def send_command(self):
        """Send current positions as command"""
        global action
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        
        msg.name = list(self.joint_names)
        msg.position = [self.current_positions[name] for name in self.joint_names]
        msg.velocity = [0.0] * len(self.joint_names)
        msg.effort = [0.0] * len(self.joint_names)  # Effort zero - position control
        
        # Update Action (for data recording)
        action = np.array(msg.position, dtype=float)
        
        self.publisher.publish(msg)
    
    def cycle_fingers(self):
        """Cycle fingers in 3 stages"""
        self.finger_state = (self.finger_state + 1) % 3
        
        if self.finger_state == 0:
            position = self.finger_full_open
        elif self.finger_state == 1:
            position = self.finger_half_closed
        else:
            position = self.finger_full_closed
        
        # Non-blocking: set targets, movement happens via timer.
        # Request: Other fingers shouldn't close before thumb closes => gating.
        thumb_joints = [
            'thumb_joint_roll_r',
            'thumb_proximal_joint_r',
            'thumb_proximal_joint_r_1',
        ]
        thumb_joints = [j for j in thumb_joints if j in self.finger_joints]
        other_finger_joints = [j for j in self.finger_joints if j not in thumb_joints]

        closing = position > self.finger_full_open + 1e-6
        if closing and thumb_joints and other_finger_joints:
            # First thumb targets; trigger others when thumb reaches target
            self.move_smooth({j: position for j in thumb_joints})
            self._finger_gate_active = True
            self._finger_gate_target = float(position)
            self._finger_gate_other_joints = other_finger_joints
        else:
            # When opening or no thumb: all together
            self._finger_gate_active = False
            self._finger_gate_target = None
            self._finger_gate_other_joints = []
            self.move_smooth({j: position for j in self.finger_joints})

    def toggle_thumb_roll(self):
        """Toggle thumb roll joint with single key (non-blocking)."""
        self.thumb_roll_is_closed = not self.thumb_roll_is_closed
        target = self.thumb_roll_closed if self.thumb_roll_is_closed else self.thumb_roll_open
        self.move_smooth({'thumb_joint_roll_r': target})
        state = 'CLOSED' if self.thumb_roll_is_closed else 'OPEN'
        print(f"👍 Thumb roll toggle -> {state} (target={target:.3f} rad)")
    
    def home(self):
        """Move all joints to home position slowly (max 0.2 rad/s)"""
        print("🏠 Going to Home position...")
        # Non-blocking: set targets, movement will happen via timer
        self.move_smooth(self.home_positions)
        self.finger_state = 0
        print("🏠 Home target set (non-blocking)")
    
    def show_positions(self):
        """Show current positions"""
        print("\n" + "="*60)
        print("CURRENT JOINT POSITIONS")
        print("="*60)
        for joint_name in self.joint_names:
            print(f"  {joint_name:30s}: {self.current_positions[joint_name]:7.3f} rad")
        print("="*60 + "\n")

class JointStates_Subscriber(Node):
    """Subscribe to /joint_states topic to get current joint positions"""
    
    def __init__(self):
        super().__init__('joint_states_subscriber')
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_states_callback,
            10)
        self.subscription
        self.get_logger().info('🔍 Joint States Subscriber started - listening to /joint_states topic...')
    
    def joint_states_callback(self, data):
        global joint_states
        joint_states['names'] = list(data.name)
        joint_states['positions'] = np.array(data.position, dtype=float)
        # Debug: Log when first data arrives
        if len(joint_states['positions']) > 0:
            self.get_logger().info(f'✅ Joint states received: {len(joint_states["positions"])} joints, first 3: {joint_states["positions"][:3]}', once=True)

class RGB_Camera_Subscriber(Node):
    """Subscribe to /rgb topic for camera image"""
    
    def __init__(self):
        super().__init__('rgb_camera_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/rgb',
            self.camera_callback,
            10)
        self.subscription 

    def camera_callback(self, data):
        global rgb_image
        try:
            # Isaac Sim sometimes sends memory that trips up cv_bridge's imgmsg_to_cv2.
            # Convert directly via numpy for stability.
            img_arr = np.frombuffer(data.data, dtype=np.uint8).reshape((data.height, data.width, -1))
            
            # If Isaac sends RGB, convert to BGR for OpenCV
            if data.encoding.lower() == 'rgb8':
                img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
            elif data.encoding.lower() == 'rgba8':
                img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2BGR)
                
            rgb_image = cv2.resize(img_arr, (vid_W, vid_H), cv2.INTER_LINEAR)
        except Exception as e:
            pass

class Wrist_Camera_Subscriber(Node):
    """Subscribe to wrist camera topic (placeholder for backward compatibility)"""
    
    def __init__(self):
        super().__init__('wrist_camera_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/wrist_camera',
            self.camera_callback,
            10)
        self.subscription
    
    def camera_callback(self, data):
        global wrist_camera_image
        try:
            img_arr = np.frombuffer(data.data, dtype=np.uint8).reshape((data.height, data.width, -1))
            if data.encoding.lower() == 'rgb8':
                img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
            elif data.encoding.lower() == 'rgba8':
                img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2BGR)
            wrist_camera_image = cv2.resize(img_arr, (vid_W, vid_H), cv2.INTER_LINEAR)
        except:
            pass

class Top_Camera_Subscriber(Node):
    """Subscribe to top camera topic (placeholder for backward compatibility)"""
    
    def __init__(self):
        super().__init__('top_camera_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/top_camera',
            self.camera_callback,
            10)
        self.subscription
    
    def camera_callback(self, data):
        global top_camera_image
        try:
            img_arr = np.frombuffer(data.data, dtype=np.uint8).reshape((data.height, data.width, -1))
            if data.encoding.lower() == 'rgb8':
                img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
            elif data.encoding.lower() == 'rgba8':
                img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2BGR)
            top_camera_image = cv2.resize(img_arr, (vid_W, vid_H), cv2.INTER_LINEAR)
        except:
            pass


class Data_Recorder(Node):

    def __init__(self):
        super().__init__('Data_Recorder')
        self.Hz = 10 # bridge data frequency (dt = 0.1 s)
        self.prev_ee_pose = np.array([0, 0, 0], float)
        self.timer = self.create_timer(1/self.Hz, self.timer_callback)
        self.start_recording = False
        self.data_recorded = False

        #### log files for multiple runs are NOT overwritten
        # Output directory (relative to script location, not cwd)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = os.path.join(script_dir, "output")
        self.log_dir = os.path.join(self.base_dir, "data/chunk-000/")
        os.makedirs(self.log_dir, exist_ok=True)

        # Meta directory for LeRobot v3.0 format
        self.meta_dir = os.path.join(self.base_dir, "meta/")
        os.makedirs(self.meta_dir, exist_ok=True)
        self.episodes_meta_dir = os.path.join(self.meta_dir, "episodes/chunk-000/")
        os.makedirs(self.episodes_meta_dir, exist_ok=True)

        # Single RGB video output from /rgb topic
        base_vid_dir = os.path.join(self.base_dir, 'videos/observation.images.rgb/chunk-000/')
        self.rgb_vid_dir = base_vid_dir
        os.makedirs(self.rgb_vid_dir, exist_ok=True)

        # Detect existing episodes and set next episode index
        self.episode_index, self.index = self._detect_last_episode()
        print(f"📊 Current episode count: {self.episode_index}, Next index: {self.index}")

        self.df = pd.DataFrame(columns=['observation.state', 'action', 'episode_index', 'frame_index', 'timestamp', 'index', 'task_index'])
        self.frame_index = 0
        self.previous_observation_state = None
        self.previous_ee_state = None
        self.time_stamp = 0.0
        self.column_index = 0

        self.rgb_image_array = []
        
        # Episode statistics for LeRobot meta
        self.all_episodes_meta = []
        self.total_frames = 0
        
        # Load existing meta if available
        self._load_existing_meta()

        # Initialize Pinocchio for EE conversion
        try:
            import pinocchio as pin
            urdf_path = "/home/beable/IsaacLab/IsaacLab/r1-new/urdf/r1-new.urdf"
            self.model = pin.buildModelFromUrdf(urdf_path)
            self.data = self.model.createData()
            if self.model.existJointName("wrist_pitch_joint_r"):
                self.ee_id = self.model.getJointId("wrist_pitch_joint_r")
            else:
                self.ee_id = self.model.getFrameId("wrist_pitch_joint_r")
            self.use_ee = True
        except Exception as e:
            self.get_logger().error(f"Pinocchio Init failed in Data_Recorder: {e}")
            self.use_ee = False

    def get_ee_pose_and_gripper(self, joint_positions, joint_names):
        import pinocchio as pin
        import numpy as np
        if not getattr(self, "use_ee", False):
            return np.array(joint_positions).tolist()
        
        q = pin.neutral(self.model)
        for name, p in zip(joint_names, joint_positions):
            if self.model.existJointName(name):
                j_id = self.model.getJointId(name)
                idx_q = self.model.joints[j_id].idx_q
                if idx_q < self.model.nq:
                    q[idx_q] = p
                    
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        
        se3 = self.data.oMi[self.ee_id]
        pos = se3.translation
        rpy = pin.rpy.matrixToRpy(se3.rotation)
        
        # Gripper joints are from index 6 onwards
        gripper_joints = joint_positions[6:]
        
        return np.concatenate([pos, rpy, gripper_joints]).tolist()

    def _detect_last_episode(self):
        """Detect last episode number from existing parquet files"""
        import glob
        
        # Find parquet files in data/chunk-000/
        parquet_files = glob.glob(os.path.join(self.log_dir, "*.parquet"))
        
        if not parquet_files:
            # Old format check (chunk_000 folder)
            old_log_dir = os.path.join(self.base_dir, "data/chunk_000/")
            parquet_files = glob.glob(os.path.join(old_log_dir, "*.parquet"))
        
        if not parquet_files:
            return 0, 0  # No episodes, start from 0
        
        max_episode = -1
        max_index = 0
        
        for pf in parquet_files:
            try:
                df = pd.read_parquet(pf)
                if 'episode_index' in df.columns:
                    ep_max = df['episode_index'].max()
                    if ep_max > max_episode:
                        max_episode = ep_max
                if 'index' in df.columns:
                    idx_max = df['index'].max()
                    if idx_max > max_index:
                        max_index = idx_max
            except Exception as e:
                print(f"⚠ Parquet read error: {pf} - {e}")
                continue
        
        next_episode = max_episode + 1 if max_episode >= 0 else 0
        next_index = max_index + 1 if max_index >= 0 else 0
        
        return next_episode, next_index

    def _load_existing_meta(self):
        """Load existing meta files"""
        info_path = os.path.join(self.meta_dir, "info.json")
        if os.path.exists(info_path):
            try:
                import json
                with open(info_path, 'r') as f:
                    info = json.load(f)
                    self.total_frames = info.get('total_frames', 0)
                    print(f"📂 Existing meta loaded: {info.get('total_episodes', 0)} episodes, {self.total_frames} frames")
            except Exception as e:
                print(f"⚠ Meta yükleme hatası: {e}")
        
        # Load existing episodes metadata
        episodes_parquet = os.path.join(self.episodes_meta_dir, "file-000.parquet")
        if os.path.exists(episodes_parquet):
            try:
                episodes_df = pd.read_parquet(episodes_parquet)
                self.all_episodes_meta = episodes_df.to_dict('records')
                # Calculate total frames from episodes to ensure exact sync
                self.total_frames = sum([ep.get('length', 0) for ep in self.all_episodes_meta])
                print(f"📂 Existing episodes meta loaded: {len(self.all_episodes_meta)} episodes, recalculated {self.total_frames} total frames")
            except Exception as e:
                print(f"⚠ Episodes meta load error: {e}")

    def timer_callback(self):
        global action, wrist_camera_image, top_camera_image, rgb_image, joint_states, record_data

        if record_data:
            print('\033[32m'+f'RECORDING episode:{self.episode_index}, frame:{self.frame_index}, index:{self.index}'+'\033[0m')

            # Observation.state: joint positions from /joint_states (filtered for right arm only)
            # Right arm joint names to filter
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
            
            # Filter joint states to get only right arm joints in correct order
            observation_state = []
            if len(joint_states['positions']) > 0 and len(joint_states['names']) > 0:
                joint_dict = dict(zip(joint_states['names'], joint_states['positions']))
                for joint_name in right_arm_joint_names:
                    if joint_name in joint_dict:
                        observation_state.append(float(joint_dict[joint_name]))
                    else:
                        observation_state.append(0.0)  # Missing joint defaults to 0
            else:
                observation_state = [0.0] * len(right_arm_joint_names)
            
            # Action: joint positions from /joint_command (sent by control_right_arm.py)
            action_to_save = copy.copy(action.tolist() if len(action) > 0 else [])

            # CONVERT TO EE SPACE
            current_ee_state = self.get_ee_pose_and_gripper(observation_state, right_arm_joint_names)
            target_ee_state = self.get_ee_pose_and_gripper(action_to_save, right_arm_joint_names) if len(action_to_save) > 0 else [0.0]*len(current_ee_state)

            # OVERRIDE FOR DELTA ACTION: target_ee - current_ee
            if len(current_ee_state) == len(target_ee_state):
                action_to_save = [t - c for t, c in zip(target_ee_state, current_ee_state)]
                # Wrap rpy differences
                for i in range(3, 6):
                    action_to_save[i] = np.arctan2(np.sin(action_to_save[i]), np.cos(action_to_save[i]))
            else:
                action_to_save = [0.0] * len(current_ee_state)
            
            # OVERRIDE FOR DELTA STATE: Calculate delta state from previous frame
            if getattr(self, "previous_ee_state", None) is None or len(self.previous_ee_state) != len(current_ee_state):
                delta_observation_state = [0.0] * len(current_ee_state)
            else:
                delta_observation_state = [o - p for o, p in zip(current_ee_state, self.previous_ee_state)]
                # Wrap rpy differences
                for i in range(3, 6):
                    delta_observation_state[i] = np.arctan2(np.sin(delta_observation_state[i]), np.cos(delta_observation_state[i]))
            
            self.previous_ee_state = copy.copy(current_ee_state)
            observation_state = delta_observation_state
            
            # DEBUG: Check if joint states are being received
            if self.frame_index == 0:  # First frame
                print('\n' + '='*70)
                print('🔍 DATA RECORDING DEBUG (First Frame)')
                print('='*70)
                
                # Check observation_state
                if len(observation_state) == 0:
                    print('\033[33m'+'⚠️  WARNING: observation_state EMPTY! /joint_states topic might not be sending data!'+'\033[0m')
                elif np.allclose(observation_state, 0.0):
                    print('\033[33m'+'⚠️  WARNING: observation_state all 0.0! Joint states might not be coming correctly!'+'\033[0m')
                    print(f'   observation_state: {observation_state}')
                else:
                    print('\033[32m'+f'✅ observation_state OK: {len(observation_state)} joints'+'\033[0m')
                    print(f'   First 5 values: {[f"{v:.4f}" for v in observation_state[:5]]}')
                    non_zero_count = sum(1 for v in observation_state if abs(v) > 0.001)
                    print(f'   Non-zero joint count: {non_zero_count}/{len(observation_state)}')
                
                # Check action
                if len(action_to_save) == 0:
                    print('\033[33m'+'⚠️  WARNING: action EMPTY!'+'\033[0m')
                else:
                    print('\033[32m'+f'✅ action OK: {len(action_to_save)} joints'+'\033[0m')
                    print(f'   First 5 values: {[f"{v:.4f}" for v in action_to_save[:5]]}')
                    non_zero_count = sum(1 for v in action_to_save if abs(v) > 0.001)
                    print(f'   Non-zero joint count: {non_zero_count}/{len(action_to_save)}')
                
                # Check raw joint_states data
                if len(joint_states['names']) > 0:
                    print(f'\n📊 Raw /joint_states topic:')
                    print(f'   Total joint count: {len(joint_states["names"])}')
                    print(f'   First 3 joints: {joint_states["names"][:3]}')
                    print(f'   First 3 positions: {[f"{v:.4f}" for v in joint_states["positions"][:3]]}')
                
                print('='*70 + '\n')
            
            self.df.loc[self.column_index] = [observation_state, action_to_save, self.episode_index, self.frame_index, self.time_stamp, self.index, 0]
            self.column_index += 1
            self.frame_index += 1
            self.time_stamp += 1/self.Hz
            self.index += 1

            self.start_recording = True

            # Use RGB image from /rgb topic (single output)
            self.rgb_image_array.append(rgb_image)

        else:
            if(self.start_recording and self.data_recorded == False):
                print('\033[31m'+'WRITING A PARQUET FILE'+'\033[0m')

                # LeRobot v3.0 format: file-{file_index:03d}.parquet
                data_file_name = f'file-{self.episode_index:03d}.parquet'
                video_file_name = f'file-{self.episode_index:03d}.mp4'

                # Save data parquet
                table = pa.Table.from_pandas(self.df)
                pq.write_table(table, self.log_dir + data_file_name)
                print("The parquet file is generated!")

                # Save video with H.264 using ffmpeg directly (browser compatible)
                video_path = self.rgb_vid_dir + video_file_name
                temp_video_path = self.rgb_vid_dir + f"temp_{video_file_name}"
                
                # First save with OpenCV (any codec that works)
                print("📹 Saving temporary video...")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_rgb = cv2.VideoWriter(temp_video_path, fourcc, self.Hz, (vid_W, vid_H))
                
                if not out_rgb.isOpened():
                    print("⚠ VideoWriter failed to open!")
                else:
                    for frame_rgb in self.rgb_image_array:
                        out_rgb.write(frame_rgb)
                    out_rgb.release()
                    print("✓ Temporary video saved")
                    
                    # Convert to H.264 using ffmpeg
                    print("🔄 Converting to H.264 (browser compatible)...")
                    try:
                        ffmpeg_cmd = [
                            'ffmpeg',
                            '-y',  # Overwrite output file
                            '-i', temp_video_path,  # Input
                            '-c:v', 'libx264',  # H.264 codec
                            '-preset', 'medium',  # Encoding speed
                            '-crf', '23',  # Quality (lower = better, 18-28 range)
                            '-pix_fmt', 'yuv420p',  # Pixel format (browser compatible)
                            '-movflags', '+faststart',  # Enable streaming
                            video_path  # Output
                        ]
                        
                        result = subprocess.run(
                            ffmpeg_cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True
                        )
                        
                        if result.returncode == 0:
                            print("✓ RGB video generated with H.264 (browser compatible)!")
                            # Remove temporary file
                            os.remove(temp_video_path)
                        else:
                            print(f"⚠ ffmpeg conversion failed: {result.stderr[:200]}")
                            print("⚠ Using temporary mp4v video instead")
                            os.rename(temp_video_path, video_path)
                    
                    except FileNotFoundError:
                        print("⚠ ffmpeg not found, using mp4v codec")
                        os.rename(temp_video_path, video_path)
                    except Exception as e:
                        print(f"⚠ Conversion error: {e}")
                        if os.path.exists(temp_video_path):
                            os.rename(temp_video_path, video_path)

                # Update episode metadata for LeRobot v3.0
                episode_length = self.frame_index
                self._save_episode_metadata(episode_length)
                
                # Generate meta files
                self._generate_meta_files()

                # Reset for next episode - exactly like original
                self.data_recorded = True
                self.episode_index += 1  # Increment episode for next recording
                self.frame_index = 0
                self.previous_observation_state = None
                self.previous_ee_state = None
                self.time_stamp = 0.0
                self.column_index = 0
                self.start_recording = False
                
                # Clear arrays for next episode
                self.rgb_image_array.clear()
                
                # Reset dataframe for next episode
                self.df = self.df.iloc[0:0]  # Clear dataframe
                
                # AUTO RESET ROBOT TO HOME POSITION after episode completion
                self.reset_robot_after_episode()
                
                print(f"🔄 Ready for episode {self.episode_index}")
                self.data_recorded = False

    def _save_episode_metadata(self, episode_length):
        """Episode metadata'sını kaydet (LeRobot v3.0 format)"""
        dataset_from_index = self.total_frames
        dataset_to_index = self.total_frames + episode_length
        
        episode_meta = {
            "episode_index": self.episode_index,
            "tasks": ["Pick the yellow box and put it in the empty box"],
            "length": episode_length,
            "dataset_from_index": dataset_from_index,
            "dataset_to_index": dataset_to_index,
            "data/chunk_index": 0,
            "data/file_index": self.episode_index,
            "videos/observation.images.rgb/chunk_index": 0,
            "videos/observation.images.rgb/file_index": self.episode_index,
        }
        
        self.all_episodes_meta.append(episode_meta)
        self.total_frames += episode_length
        
        print(f"📝 Episode {self.episode_index} metadata saved: {episode_length} frames")

    def _generate_meta_files(self):
        """Generate all meta files in LeRobot v3.0 format"""
        import json
        
        # YENI format: End effector delta pozisyon ve oryantasyonu + parmaklar
        ee_names = [
            "x", "y", "z", "roll", "pitch", "yaw",
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
        
        # Ensure total_frames is correct before generating info.json
        self.total_frames = sum([ep.get('length', 0) for ep in self.all_episodes_meta])
        
        # 1. info.json
        info = {
            "codebase_version": "v3.0",
            "robot_type": "r1_humanoid",
            "total_episodes": len(self.all_episodes_meta),
            "total_frames": self.total_frames,
            "total_tasks": 1,
            "fps": self.Hz,
            "splits": {"train": f"0:{len(self.all_episodes_meta)}"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": {
                "observation.state": {
                    "dtype": "float32",
                    "shape": [len(ee_names)],
                    "names": ee_names
                },
                "action": {
                    "dtype": "float32",
                    "shape": [len(ee_names)],
                    "names": ee_names
                },
                "observation.images.rgb": {
                    "dtype": "video",
                    "shape": [vid_H, vid_W, 3],
                    "names": ["height", "width", "channels"],
                    "info": {
                        "video.fps": self.Hz,
                        "video.height": vid_H,
                        "video.width": vid_W,
                        "video.codec": "h264",
                        "video.pix_fmt": "yuv420p"
                    }
                },
                "timestamp": {
                    "dtype": "float32",
                    "shape": [1],
                    "names": None
                },
                "frame_index": {
                    "dtype": "int64",
                    "shape": [1],
                    "names": None
                },
                "episode_index": {
                    "dtype": "int64",
                    "shape": [1],
                    "names": None
                },
                "index": {
                    "dtype": "int64",
                    "shape": [1],
                    "names": None
                },
                "task_index": {
                    "dtype": "int64",
                    "shape": [1],
                    "names": None
                }
            }
        }
        
        print(f"ℹ️  Dataset format: observation.state + action (no reward/done/success)")
        
        info_path = os.path.join(self.meta_dir, "info.json")
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4, ensure_ascii=False)
        print(f"✅ info.json created")
        
        # 2. tasks.parquet
        tasks_df = pd.DataFrame({
            "task_index": [1],
        }, index=["Pick the red box and put it in the empty box"])
        tasks_path = os.path.join(self.meta_dir, "tasks.parquet")
        tasks_df.to_parquet(tasks_path)
        print(f"✅ tasks.parquet created")
        
        # 3. episodes/chunk-000/file-000.parquet
        if self.all_episodes_meta:
            episodes_df = pd.DataFrame(self.all_episodes_meta)
            episodes_path = os.path.join(self.episodes_meta_dir, "file-000.parquet")
            episodes_df.to_parquet(episodes_path, index=False)
            print(f"✅ episodes meta parquet created: {len(self.all_episodes_meta)} episodes")
        
        # 4. stats.json (basit istatistikler - ee isimleriyle)
        stats = {
            "observation.state": {
                "min": {name: -3.14 for name in ee_names},
                "max": {name: 3.14 for name in ee_names},
                "mean": {name: 0.0 for name in ee_names},
                "std": {name: 1.0 for name in ee_names},
                "count": self.total_frames
            },
            "action": {
                "min": {name: -3.14 for name in ee_names},
                "max": {name: 3.14 for name in ee_names},
                "mean": {name: 0.0 for name in ee_names},
                "std": {name: 1.0 for name in ee_names},
                "count": self.total_frames
            }
        }
        stats_path = os.path.join(self.meta_dir, "stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)
        print(f"✅ stats.json created")
        
        print(f"🎉 All meta files created! Total: {len(self.all_episodes_meta)} episodes, {self.total_frames} frames")

    def reset_robot_after_episode(self):
        """Placeholder: post-episode reset removed (no /joint_trajectory topic)."""
        print("ℹ Post-episode home reset skipped (no /joint_trajectory controller).")


def interactive_control(controller):
    """Interactive keyboard + mouse control"""
    global record_data, mouse_middle_button
    
    print("\n" + "="*70)
    print("RIGHT ARM & FINGERS CONTROL + DATA RECORDING")
    print("="*70)
    print(f"\nROS2 Topic: /joint_command")
    print(f"Controlled joints: {len(controller.joint_names)}")
    print("\n⌨️  KEYBOARD CONTROL:")
    print("  (IK Coordinate System if IK is active, otherwise Joints):")
    print("  Movement:")
    print("    a / d  -> Arm Forward / Backward (+/- X axis)")
    print("    s / w  -> Arm Left / Right (+/- Y axis)")
    print("    e / r  -> Arm Up / Down (+/- Z axis)")
    print("  Rotation:")
    print("    f / t  -> Pitch (+/- Y axis rot)")
    print("    q / y  -> Yaw (+/- Z axis rot)")
    print("    z / x  -> Roll (+/- X axis rot)")
    print("    c / v  -> thumb_joint_roll_r (+/- 0.01 rad)")
    print("    b      -> thumb_joint_roll_r toggle open/close")
    print("  Fingers (3-step cycle):")
    print("    g -> Cycle: Full open -> Half closed -> Full closed")
    print("\n🖱️  MOUSE CONTROL:")
    if MOUSE_AVAILABLE:
        print("    m      -> Toggle mouse control")
        print("    n      -> Change mouse mode (Shoulder/Elbow/Wrist)")
        print("    (When mouse active:)")
        print("      Movement   -> Control selected joints")
        print("      Left Click -> Close fingers")
        print("      Right Click-> Open fingers")
        print("      Middle Click-> Change mode")
        print("      Scroll     -> Adjust sensitivity")
    else:
        print("    ⚠ pynput module not installed. Mouse control unavailable.")
        print("    To install: pip install pynput")
    print("\n  Recording:")
    print("    SPACE -> Start/Stop Recording")
    print("\n  Other:")
    print("    h -> Home (slow/stepwise)")
    print("    p -> Show positions")
    print("    ESC or Ctrl+C -> Exit")
    print("\n" + "-"*70)
    print("Press any key to start...\n")
    
    # Test ROS2 connection
    time.sleep(0.5)
    try:
        test_msg = JointState()
        test_msg.header.stamp = controller.get_clock().now().to_msg()
        test_msg.header.frame_id = 'base_link'
        test_msg.name = controller.joint_names[:1]
        test_msg.position = [0.0]
        test_msg.velocity = [0.0]
        test_msg.effort = [0.0]
        controller.publisher.publish(test_msg)
        print("✓ ROS2 connection successful\n")
    except Exception as e:
        print(f"⚠ ROS2 connection warning: {e}")
    
    push_time = 0
    prev_push_time = 0
    
    while rclpy.ok():
        try:
            # Get all keys pressed instantaneously
            keys = get_all_keys()
            
            for key in keys:
                # Exit with ESC
                if ord(key) == 27:
                    print("\nExiting...")
                    rclpy.shutdown()
                    return
                
                # Start/Stop recording with SPACE
                if key == ' ':
                    push_time = time.time()
                    dif = push_time - prev_push_time
                    if dif > 1:  # Debounce
                        if record_data == False:
                            record_data = True
                            print('\033[32m'+'🔴 START RECORDING'+'\033[0m')
                        else:
                            record_data = False
                            print('\033[31m'+'⏹️  END RECORDING'+'\033[0m')
                    prev_push_time = push_time
                    continue
                
                key_lower = key.lower()
                
                # Special commands
                if key_lower == 'h':
                    controller.home()
                    continue
                
                if key_lower == 'p':
                    controller.show_positions()
                    continue
                
                # Finger controls (3 stages)
                if key_lower == 'g':
                    controller.cycle_fingers()
                    continue

                # Thumb-roll toggle open/close
                if key_lower == 'b':
                    controller.toggle_thumb_roll()
                    continue
                
                # Toggle mouse control
                if key_lower == 'm' and MOUSE_AVAILABLE:
                    controller.toggle_mouse_control()
                    continue
                
                # Change mouse mode
                if key_lower == 'n' and MOUSE_AVAILABLE and controller.mouse_enabled:
                    controller.cycle_mouse_mode()
                    continue
                
                # Change mode with Middle click (when mouse active)
                if mouse_middle_button and MOUSE_AVAILABLE and controller.mouse_enabled:
                    mouse_middle_button = False  # Reset
                    controller.cycle_mouse_mode()
                    continue
                
                # Control from keyboard mapping (arm controls + IK cartesian)
                if key_lower in controller.key_mappings and not controller.use_ik:
                    joint_name, delta = controller.key_mappings[key_lower]
                    controller.update_position(joint_name, delta)
                    continue
                
                if controller.use_ik:
                    # override keyboard
                    if key_lower == 'a': controller.update_cartesian([0.01, 0, 0]); continue   # mapped to +X (forward)
                    if key_lower == 'd': controller.update_cartesian([-0.01, 0, 0]); continue  # mapped to -X (backward)
                    if key_lower == 's': controller.update_cartesian([0, 0.01, 0]); continue   # mapped to +Y (left)
                    if key_lower == 'w': controller.update_cartesian([0, -0.01, 0]); continue  # mapped to -Y (right)
                    if key_lower == 'e': controller.update_cartesian([0, 0, 0.01]); continue
                    if key_lower == 'r': controller.update_cartesian([0, 0, -0.01]); continue
                    
                    if key_lower == 'f': controller.update_cartesian(None, [0, 0.05, 0]); continue
                    if key_lower == 't': controller.update_cartesian(None, [0, -0.05, 0]); continue
                    if key_lower == 'q': controller.update_cartesian(None, [0, 0, 0.05]); continue
                    if key_lower == 'y': controller.update_cartesian(None, [0, 0, -0.05]); continue
                    if key_lower == 'z': controller.update_cartesian(None, [0.05, 0, 0]); continue
                    if key_lower == 'x': controller.update_cartesian(None, [-0.05, 0, 0]); continue
                    
                    if key_lower == 'c': controller.update_position("thumb_joint_roll_r", +controller.step_size * 3); continue
                    if key_lower == 'v': controller.update_position("thumb_joint_roll_r", -controller.step_size * 3); continue
        
        except KeyboardInterrupt:
            print("\n\nStopped.")
            break
        except Exception as e:
            print(f"Hata: {e}")


if __name__ == '__main__':
    rclpy.init(args=None)

    joint_states_subscriber = JointStates_Subscriber()
    joint_command_subscriber = JointCommand_Subscriber()
    rgb_camera_subscriber = RGB_Camera_Subscriber()
    robot_keyboard_controller = Robot_Keyboard_Controller()
    wrist_camera_subscriber = Wrist_Camera_Subscriber()  # Keep for backward compatibility
    top_camera_subscriber = Top_Camera_Subscriber()  # Keep for backward compatibility
    data_recorder = Data_Recorder()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(joint_states_subscriber)
    executor.add_node(joint_command_subscriber)
    executor.add_node(rgb_camera_subscriber)
    executor.add_node(robot_keyboard_controller)
    executor.add_node(wrist_camera_subscriber)
    executor.add_node(top_camera_subscriber)
    executor.add_node(data_recorder)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    # Start interactive control (like control_right_arm.py)
    try:
        interactive_control(robot_keyboard_controller)
    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except RuntimeError:
            pass
        executor_thread.join()