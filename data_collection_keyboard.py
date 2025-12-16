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
from tf2_msgs.msg import TFMessage
from scipy.spatial.transform import Rotation as R
import numpy as np
from cv_bridge import CvBridge
from math import sin, cos, pi
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sys
import termios
import tty

bridge = CvBridge()

record_data = False
tool_pose_xy = [0.0, 0.0] # tool(end effector) pose
tbar_pose_xyw = [0.0, 0.0, 0.0]
vid_H = 360
vid_W = 640
wrist_camera_image = np.zeros((vid_H, vid_W, 3), np.uint8)
top_camera_image = np.zeros((vid_H, vid_W, 3), np.uint8)
rgb_image = np.zeros((vid_H, vid_W, 3), np.uint8)  # RGB image from /rgb topic
action = np.array([0.0] * 17, float)  # Joint positions as action (17 joints for right arm)
joint_states = {'names': [], 'positions': np.array([], float)}  # Joint states from /joint_states topic


class Get_Poses_Subscriber(Node):

    def __init__(self):
        super().__init__('get_modelstate')
        self.subscription = self.create_subscription(
            TFMessage,
            '/isaac_tf',
            self.listener_callback,
            10)
        self.subscription

        self.euler_angles = np.array([0.0, 0.0, 0.0], float)

    def listener_callback(self, data):
        global tool_pose_xy, tbar_pose_xyw

        # 0:tool
        tool_pose = data.transforms[0].transform.translation
        tool_pose_xy[0] = tool_pose.y
        tool_pose_xy[1] = tool_pose.x

        # 1:tbar
        tbar_translation  = data.transforms[1].transform.translation       
        tbar_rotation = data.transforms[1].transform.rotation 
        tbar_pose_xyw[0] = tbar_translation.y
        tbar_pose_xyw[1] = tbar_translation.x
        self.euler_angles[:] = R.from_quat([tbar_rotation.x, tbar_rotation.y, tbar_rotation.z, tbar_rotation.w]).as_euler('xyz', degrees=False)
        tbar_pose_xyw[2] = self.euler_angles[2]

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
        
        # Robot joint isimleri - rod_graph.usda'daki JointNameArray ile eşleşmeli
        # Toplam 43 joint
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
        
        self.get_logger().info(f'Robot Arm Controller başlatıldı')
        self.get_logger().info(f'Topic: {topic_name}')
        self.get_logger().info(f'Joint sayısı: {len(self.joint_names)}')
        
    def send_command(self, positions, velocities=None, duration_sec=3.0):
        """
        Joint komutunu gönder.
        
        Args:
            positions: List veya dict. Joint pozisyonları (radyan)
            velocities: List veya dict. Joint hızları (radyan/s) - opsiyonel
            duration_sec: Hareket süresi (saniye)
        """
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.name = self.joint_names
        
        # Pozisyonları hazırla
        if isinstance(positions, dict):
            pos_list = [positions.get(name, 0.0) for name in self.joint_names]
        else:
            pos_list = list(positions) if len(positions) >= len(self.joint_names) else list(positions) + [0.0] * (len(self.joint_names) - len(positions))
            pos_list = pos_list[:len(self.joint_names)]
        
        # Hızlar
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
        
        # Yayınla
        self.publisher.publish(msg)
        self.get_logger().info(f'Komut gönderildi: {pos_list[:5]}... (ilk 5 joint)')
        return msg
    
    def move_joint(self, joint_name, position, duration_sec=2.0):
        """Belirli bir joint'i hareket ettir"""
        if joint_name not in self.joint_names:
            self.get_logger().error(f'Joint bulunamadı: {joint_name}')
            self.get_logger().info(f'Mevcut jointler: {self.joint_names}')
            return False
        
        positions = {joint_name: float(position)}
        self.send_command(positions, duration_sec=duration_sec)
        return True
    
    def move_all(self, positions, duration_sec=3.0):
        """Tüm joint'leri hareket ettir"""
        self.send_command(positions, duration_sec=duration_sec)
    
    def home(self, duration_sec=3.0):
        """Tüm joint'leri sıfır pozisyonuna al"""
        positions = [0.0] * len(self.joint_names)
        self.send_command(positions, duration_sec=duration_sec)
        self.get_logger().info('Home pozisyonuna gidiliyor...')

def interactive_control(controller):
    """İnteraktif kontrol döngüsü - from control_robot_arms.py - unchanged"""
    print("\n" + "="*60)
    print("ROBOT KOL KONTROLÜ")
    print("="*60)
    print("\nKomutlar:")
    print("  set <pos1> <pos2> ...     -> Tüm joint'leri belirtilen pozisyonlara al")
    print("  move <joint_name> <pos>   -> Belirli bir joint'i hareket ettir")
    print("  home                     -> Tüm joint'leri sıfır pozisyonuna al")
    print("  list                     -> Joint isimlerini listele")
    print("  quit                     -> Çıkış")
    print("\n" + "-"*60 + "\n")
    
    while rclpy.ok():
        try:
            command = input("Komut> ").strip()
            
            if not command:
                continue
            
            parts = command.split()
            cmd = parts[0].lower()
            
            if cmd == 'quit' or cmd == 'exit' or cmd == 'q':
                print("Çıkılıyor...")
                break
            
            elif cmd == 'home':
                controller.home()
            
            elif cmd == 'list':
                print("\nJoint İsimleri:")
                for i, name in enumerate(controller.joint_names):
                    print(f"  [{i}] {name}")
                print()
            
            elif cmd == 'set':
                if len(parts) < 2:
                    print("Kullanım: set <pos1> <pos2> ...")
                    continue
                try:
                    positions = [float(p) for p in parts[1:]]
                    controller.move_all(positions)
                except ValueError as e:
                    print(f"Hata: Geçersiz pozisyon değeri - {e}")
            
            elif cmd == 'move':
                if len(parts) < 3:
                    print("Kullanım: move <joint_name> <position>")
                    continue
                try:
                    joint_name = parts[1]
                    position = float(parts[2])
                    controller.move_joint(joint_name, position)
                except ValueError as e:
                    print(f"Hata: Geçersiz pozisyon değeri - {e}")
            
            else:
                print(f"Bilinmeyen komut: {cmd}")
                print("Komutlar: set, move, home, list, quit")
        
        except EOFError:
            print("\nÇıkılıyor...")
            break
        except Exception as e:
            print(f"Hata: {e}")


def get_key():
    """Tek bir tuş basımını al (non-blocking)"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


class Robot_Keyboard_Controller(Node):

    def __init__(self):
        super().__init__('robot_keyboard_controller')
        
        # Robot control publisher - Same as control_right_arm.py
        topic_name = self.declare_parameter('topic_name', '/joint_command').value
        self.publisher = self.create_publisher(JointState, topic_name, 10)
        
        # Sağ kol ve parmak joint'leri (control_right_arm.py ile aynı)
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
        
        # Home pozisyonları (robotun doğal duruş pozisyonu)
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
        
        # Mevcut pozisyonlar (sıfırdan başla - control_right_arm.py gibi)
        self.current_positions = {name: 0.0 for name in self.joint_names}

        # Hedef pozisyonlar: non-blocking yumuşak hareket için (tüm eklemler)
        self.target_positions = self.current_positions.copy()
        
        # Adım boyutu (radyan) - tuş başına hareket miktarı
        self.step_size = 0.01
        
        # Maksimum hız (radyan/saniye) - yavaş ve güvenli hareket için
        self.max_velocity = 0.2  # 0.2 rad/s
        self.control_rate = 50   # Hz - kontrol döngüsü frekansı
        self.smooth_step = self.max_velocity / self.control_rate  # Her adımda maksimum hareket

        # Parmaklar biraz daha hızlı olabilir: ayrı hız limiti
        self.finger_max_velocity = 0.35  # rad/s (arm: 0.2 rad/s)
        self.finger_smooth_step = self.finger_max_velocity / self.control_rate

        # Effort/Kuvvet limitleri (Newton-metre)
        # Parmaklar için düşük tork -> objeyi ezmez, fırlatmaz
        self.finger_max_effort = 5.0   # Nm - parmaklar için (düşük tutun!)
        self.arm_max_effort = 50.0     # Nm - kol joint'leri için

        # Non-blocking hareket güncelleyici (50Hz): current_positions -> target_positions
        self._motion_timer = self.create_timer(1.0 / self.control_rate, self._motion_update)
        
        # Parmak joint'leri
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
        
        # Parmak pozisyonları (3 aşama)
        self.finger_full_open = 0.1
        self.finger_half_closed = 1.0
        self.finger_full_closed = 3.0
        self.finger_state = 0

        # Thumb roll toggle (tek tuşla aç/kapat)
        # Not: thumb_joint_roll_r aynı zamanda (c/v) ile manuel +/- hareket ediyor.
        # Toggle, non-blocking hedefe gider.
        self.thumb_roll_open = -0.3
        self.thumb_roll_closed = 0.3
        self.thumb_roll_is_closed = False

        # Thumb-first / gating için durum
        self._finger_gate_active = False
        self._finger_gate_target = None
        self._finger_gate_other_joints = []
        
        # Klavye mapping (control_right_arm.py ile aynı)
        self.key_mappings = {
            # Omuz
            'a': ("right_shoulder_link_joint", +self.step_size),
            'w': ("right_shoulder_link_joint", -self.step_size),
            's': ("right_arm_top_link_joint", +self.step_size),
            'd': ("right_arm_top_link_joint", -self.step_size),
            
            # Dirsek
            'e': ("right_arm_bottom_link_joint", +self.step_size),
            'r': ("right_arm_bottom_link_joint", -self.step_size),
            'f': ("right_forearm_link_joint", +self.step_size),
            't': ("right_forearm_link_joint", -self.step_size),
            
            # Bilek
            'q': ("wrist_pitch_joint_r", +self.step_size),
            'y': ("wrist_pitch_joint_r", -self.step_size),
            'z': ("wrist_roll_joint_r", +self.step_size),
            'x': ("wrist_roll_joint_r", -self.step_size),

            # Baş parmak roll
            'c': ("thumb_joint_roll_r", +self.step_size),
            'v': ("thumb_joint_roll_r", -self.step_size),
        }
        
        self.get_logger().info(f'Robot Keyboard Controller başlatıldı')
        self.get_logger().info(f'Topic: {topic_name}')
        self.get_logger().info(f'Joint sayısı: {len(self.joint_names)}')
        
        # Başlangıçta home pozisyonuna git
        self.send_command()

    def _motion_update(self):
        """Non-blocking hareket: hedefe max_velocity ile yaklaş ve komut gönder."""
        moved = False
        for joint_name in self.joint_names:
            current = self.current_positions[joint_name]
            target = self.target_positions.get(joint_name, current)
            diff = target - current

            if abs(diff) > 0.001:
                # Parmaklarda ayrı hız limiti uygula
                max_step = self.finger_smooth_step if joint_name in self.finger_joints else self.smooth_step
                step = max_step if abs(diff) > max_step else abs(diff)
                self.current_positions[joint_name] = current + (step if diff > 0 else -step)
                moved = True
            else:
                # Snap to exact target if very close
                self.current_positions[joint_name] = target

        # Thumb hedefe ulaştıysa diğer parmak hedeflerini devreye al
        if self._finger_gate_active and self._finger_gate_target is not None:
            if self._thumb_at_target(self._finger_gate_target):
                if self._finger_gate_other_joints:
                    self.move_smooth({j: self._finger_gate_target for j in self._finger_gate_other_joints})
                self._finger_gate_active = False
                self._finger_gate_target = None
                self._finger_gate_other_joints = []

        # Sürekli publish etmeyelim: sadece hareket varken gönder
        if moved:
            self.send_command()

    def _thumb_at_target(self, target, tol=0.03):
        """Thumb eklemleri hedef pozisyona yeterince yaklaştı mı?"""
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
        
    def update_position(self, joint_name, delta):
        """Joint pozisyonunu güncelle ve komut gönder"""
        global action
        if joint_name not in self.current_positions:
            print(f'HATA: Bilinmeyen joint: {joint_name}')
            return
        
        # Pozisyonu güncelle (tuş başına step_size kadar) -> hedefi güncelle (non-blocking)
        self.target_positions[joint_name] = self.target_positions.get(joint_name, self.current_positions[joint_name]) + delta

    def move_smooth(self, target_positions_dict):
        """(Non-blocking) hedef pozisyonları güncelle. Hareket timer ile gerçekleşir."""
        for joint_name, target in target_positions_dict.items():
            if joint_name in self.target_positions:
                self.target_positions[joint_name] = float(target)
    
    def send_command(self):
        """Mevcut pozisyonları komut olarak gönder"""
        global action
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        
        msg.name = list(self.joint_names)
        msg.position = [self.current_positions[name] for name in self.joint_names]
        msg.velocity = [0.0] * len(self.joint_names)
        msg.effort = [0.0] * len(self.joint_names)  # Effort sıfır - position control
        
        # Action'ı güncelle (veri kaydı için)
        action = np.array(msg.position, dtype=float)
        
        self.publisher.publish(msg)
    
    def cycle_fingers(self):
        """Parmakları 3 aşamada döngüye al"""
        self.finger_state = (self.finger_state + 1) % 3
        
        if self.finger_state == 0:
            position = self.finger_full_open
        elif self.finger_state == 1:
            position = self.finger_half_closed
        else:
            position = self.finger_full_closed
        
        # Non-blocking: hedefleri ayarla, hareket timer ile gerçekleşir.
        # İstek: thumb kapanmadan diğer parmaklar kapanmasın => gating.
        thumb_joints = [
            'thumb_joint_roll_r',
            'thumb_proximal_joint_r',
            'thumb_proximal_joint_r_1',
        ]
        thumb_joints = [j for j in thumb_joints if j in self.finger_joints]
        other_finger_joints = [j for j in self.finger_joints if j not in thumb_joints]

        closing = position > self.finger_full_open + 1e-6
        if closing and thumb_joints and other_finger_joints:
            # Önce thumb hedefleri; diğerlerini, thumb hedefe gelince devreye al
            self.move_smooth({j: position for j in thumb_joints})
            self._finger_gate_active = True
            self._finger_gate_target = float(position)
            self._finger_gate_other_joints = other_finger_joints
        else:
            # Açarken veya thumb yoksa: hepsi birlikte
            self._finger_gate_active = False
            self._finger_gate_target = None
            self._finger_gate_other_joints = []
            self.move_smooth({j: position for j in self.finger_joints})

    def toggle_thumb_roll(self):
        """Thumb roll jointini tek tuşla aç/kapat (non-blocking)."""
        self.thumb_roll_is_closed = not self.thumb_roll_is_closed
        target = self.thumb_roll_closed if self.thumb_roll_is_closed else self.thumb_roll_open
        self.move_smooth({'thumb_joint_roll_r': target})
        state = 'CLOSED' if self.thumb_roll_is_closed else 'OPEN'
        print(f"👍 Thumb roll toggle -> {state} (target={target:.3f} rad)")
    
    def home(self):
        """Tüm joint'leri home pozisyonuna yavaşça al (max 0.2 rad/s)"""
        print("🏠 Home pozisyonuna gidiliyor...")
        # Non-blocking: hedefleri ayarla, hareket timer ile gerçekleşecek
        self.move_smooth(self.home_positions)
        self.finger_state = 0
        print("🏠 Home hedefi set edildi (non-blocking)")
    
    def show_positions(self):
        """Mevcut pozisyonları göster"""
        print("\n" + "="*60)
        print("MEVCUT JOINT POZİSYONLARI")
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
    
    def joint_states_callback(self, data):
        global joint_states
        joint_states['names'] = list(data.name)
        joint_states['positions'] = np.array(data.position, dtype=float)

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
        rgb_image = cv2.resize(bridge.imgmsg_to_cv2(data, "bgr8"), (vid_W, vid_H), cv2.INTER_LINEAR)

# Keep old camera subscribers for backward compatibility (if needed)
class Wrist_Camera_Subscriber(Node):

    def __init__(self):
        super().__init__('wrist_camera_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/rgb_wrist',
            self.camera_callback,
            10)
        self.subscription 

    def camera_callback(self, data):
        global wrist_camera_image
        wrist_camera_image = cv2.resize(bridge.imgmsg_to_cv2(data, "bgr8"), (vid_W, vid_H), cv2.INTER_LINEAR)

class Top_Camera_Subscriber(Node):

    def __init__(self):
        super().__init__('top_camera_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/rgb_top',
            self.camera_callback,
            10)
        self.subscription 

    def camera_callback(self, data):
        global top_camera_image
        top_camera_image = cv2.resize(bridge.imgmsg_to_cv2(data, "bgr8"), (vid_W, vid_H), cv2.INTER_LINEAR)

class Data_Recorder(Node):

    def __init__(self):
        super().__init__('Data_Recorder')
        self.Hz = 10 # bridge data frequency (dt = 0.1 s)
        self.prev_ee_pose = np.array([0, 0, 0], float)
        self.timer = self.create_timer(1/self.Hz, self.timer_callback)
        self.start_recording = False
        self.data_recorded = False

        #### log files for multiple runs are NOT overwritten
        # Output directory (current project ./output)
        self.base_dir = os.path.join(os.getcwd(), "output")
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
        print(f"📊 Mevcut episode sayısı: {self.episode_index}, Sonraki index: {self.index}")

        # image of a T shape on the table
        try:
            self.initial_image = cv2.imread(os.environ['HOME'] + "/ur5_simulation/images/stand_top_plane.png")
            if self.initial_image is not None:
                self.initial_image = cv2.rotate(self.initial_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                self.initial_image = np.zeros((vid_H, vid_W, 3), np.uint8)
        except:
            self.initial_image = np.zeros((vid_H, vid_W, 3), np.uint8)
 
        # for reward calculation
        self.Tbar_region = np.zeros((self.initial_image.shape[0], self.initial_image.shape[1]), np.uint8)

        # filled image of T shape on the table
        try:
            self.T_image = cv2.imread(os.environ['HOME'] + "/ur5_simulation/images/stand_top_plane_filled.png")
            if self.T_image is not None:
                self.T_image = cv2.rotate(self.T_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img_gray = cv2.cvtColor(self.T_image, cv2.COLOR_BGR2GRAY)
                thr, img_th = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)
                self.blue_region = cv2.bitwise_not(img_th)
                self.blue_region_sum = cv2.countNonZero(self.blue_region)
            else:
                self.blue_region = np.zeros((vid_H, vid_W), np.uint8)
                self.blue_region_sum = 1
        except:
            self.blue_region = np.zeros((vid_H, vid_W), np.uint8)
            self.blue_region_sum = 1
        
        self.pub_img = self.create_publisher(Image, '/pushT_image', 10)
        self.tool_radius = 10 # millimeters
        self.scale = 1.639344 # mm/pix
        self.C_W = 182 # pix
        self.C_H = 152 # pix
        self.OBL1 = int(150/self.scale)
        self.OBL2 = int(120/self.scale)
        self.OBW = int(30/self.scale)
        # radius of the tool
        self.radius = int(10/self.scale)

        self.df = pd.DataFrame(columns=['observation.state', 'action', 'episode_index', 'frame_index', 'timestamp', 'next.reward', 'next.done', 'next.success', 'index', 'task_index'])
        self.frame_index = 0
        self.time_stamp = 0.0
        self.success = False
        self.done = False
        self.column_index = 0
        self.prev_sum = 0.0

        self.rgb_image_array = []
        
        # Episode statistics for LeRobot meta
        self.all_episodes_meta = []
        self.total_frames = 0
        
        # Load existing meta if available
        self._load_existing_meta()

    def _detect_last_episode(self):
        """Mevcut parquet dosyalarından son episode numarasını tespit et"""
        import glob
        
        # data/chunk-000/ içindeki parquet dosyalarını bul
        parquet_files = glob.glob(os.path.join(self.log_dir, "*.parquet"))
        
        if not parquet_files:
            # Eski format kontrolü (chunk_000 klasörü)
            old_log_dir = os.path.join(self.base_dir, "data/chunk_000/")
            parquet_files = glob.glob(os.path.join(old_log_dir, "*.parquet"))
        
        if not parquet_files:
            return 0, 0  # Hiç episode yok, 0'dan başla
        
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
                print(f"⚠ Parquet okuma hatası: {pf} - {e}")
                continue
        
        next_episode = max_episode + 1 if max_episode >= 0 else 0
        next_index = max_index + 1 if max_index >= 0 else 0
        
        return next_episode, next_index

    def _load_existing_meta(self):
        """Mevcut meta dosyalarını yükle"""
        info_path = os.path.join(self.meta_dir, "info.json")
        if os.path.exists(info_path):
            try:
                import json
                with open(info_path, 'r') as f:
                    info = json.load(f)
                    self.total_frames = info.get('total_frames', 0)
                    print(f"📂 Mevcut meta yüklendi: {info.get('total_episodes', 0)} episode, {self.total_frames} frame")
            except Exception as e:
                print(f"⚠ Meta yükleme hatası: {e}")
        
        # Load existing episodes metadata
        episodes_parquet = os.path.join(self.episodes_meta_dir, "file-000.parquet")
        if os.path.exists(episodes_parquet):
            try:
                episodes_df = pd.read_parquet(episodes_parquet)
                self.all_episodes_meta = episodes_df.to_dict('records')
                print(f"📂 Mevcut episodes meta yüklendi: {len(self.all_episodes_meta)} episode")
            except Exception as e:
                print(f"⚠ Episodes meta yükleme hatası: {e}")

    def timer_callback(self):
        global tool_pose_xy, tbar_pose_xyw, action, wrist_camera_image, top_camera_image, rgb_image, joint_states, record_data
        
        image = copy.copy(self.initial_image)
        self.Tbar_region[:] = 0

        x = int((tool_pose_xy[0]*1000 + 300)/self.scale)
        y = int((tool_pose_xy[1]*1000 - 320)/self.scale)

        cv2.circle(image, center=(x, y), radius=self.radius, color=(100, 100, 100), thickness=cv2.FILLED)        
        
        # horizontal part of T
        x1 = tbar_pose_xyw[0]
        y1 = tbar_pose_xyw[1]
        th1 = -tbar_pose_xyw[2] - pi/2
        dx1 = -self.OBW/2*cos(th1 - pi/2)
        dy1 = -self.OBW/2*sin(th1 - pi/2)
        self.tbar1_ob = [[int(cos(th1)*self.OBL1/2     - sin(th1)*self.OBW/2   + dx1 + self.C_W + 1000*x1/self.scale), int(sin(th1)*self.OBL1/2    + cos(th1)*self.OBW/2   + dy1 + (1000*y1-320)/self.scale)],
                          [int(cos(th1)*self.OBL1/2    - sin(th1)*(-self.OBW/2)+ dx1 + self.C_W + 1000*x1/self.scale), int(sin(th1)*self.OBL1/2    + cos(th1)*(-self.OBW/2)+ dy1 + (1000*y1-320)/self.scale)],
                          [int(cos(th1)*(-self.OBL1/2) - sin(th1)*(-self.OBW/2)+ dx1 + self.C_W + 1000*x1/self.scale), int(sin(th1)*(-self.OBL1/2) + cos(th1)*(-self.OBW/2)+ dy1 + (1000*y1-320)/self.scale)],
                          [int(cos(th1)*(-self.OBL1/2) - sin(th1)*self.OBW/2   + dx1 + self.C_W + 1000*x1/self.scale), int(sin(th1)*(-self.OBL1/2) + cos(th1)*self.OBW/2   + dy1 + (1000*y1-320)/self.scale)]]  
        pts1_ob = np.array(self.tbar1_ob, np.int32)
        cv2.fillPoly(image, [pts1_ob], (0, 0, 180))
        cv2.fillPoly(self.Tbar_region, [pts1_ob], 255)
        
        #vertical part of T
        th2 = -tbar_pose_xyw[2] - pi
        dx2 = self.OBL2/2*cos(th2)
        dy2 = self.OBL2/2*sin(th2)
        self.tbar2_ob = [[int(cos(th2)*self.OBL2/2    - sin(th2)*self.OBW/2    + dx2 + self.C_W + 1000*x1/self.scale), int(sin(th2)*self.OBL2/2    + cos(th2)*self.OBW/2   + dy2 + (1000*y1-320)/self.scale)],
                          [int(cos(th2)*self.OBL2/2    - sin(th2)*(-self.OBW/2)+ dx2 + self.C_W + 1000*x1/self.scale), int(sin(th2)*self.OBL2/2    + cos(th2)*(-self.OBW/2)+ dy2 + (1000*y1-320)/self.scale)],
                          [int(cos(th2)*(-self.OBL2/2) - sin(th2)*(-self.OBW/2)+ dx2 + self.C_W + 1000*x1/self.scale), int(sin(th2)*(-self.OBL2/2) + cos(th2)*(-self.OBW/2)+ dy2 + (1000*y1-320)/self.scale)],
                          [int(cos(th2)*(-self.OBL2/2) - sin(th2)*self.OBW/2   + dx2 + self.C_W + 1000*x1/self.scale), int(sin(th2)*(-self.OBL2/2) + cos(th2)*self.OBW/2   + dy2 + (1000*y1-320)/self.scale)]]  
        pts2_ob = np.array(self.tbar2_ob, np.int32)
        cv2.fillPoly(image, [pts2_ob], (0, 0, 180))
        cv2.fillPoly(self.Tbar_region, [pts2_ob], 255)

        common_part = cv2.bitwise_and(self.blue_region, self.Tbar_region)
        common_part_sum = cv2.countNonZero(common_part)
        sum = common_part_sum/self.blue_region_sum
        sum_dif = sum - self.prev_sum
        self.prev_sum = sum

        cv2.circle(image, center=(int(self.C_W + 1000*x1/self.scale), int((1000*y1-320)/self.scale)), radius=2, color=(0, 200, 0), thickness=cv2.FILLED)  

        img_msg = bridge.cv2_to_imgmsg(image)  
        self.pub_img.publish(img_msg) 

        if record_data:
            print('\033[32m'+f'RECORDING episode:{self.episode_index}, index:{self.index} sum:{sum}'+'\033[0m')

            if sum >= 0.90:
                self.success = True
                self.done = True
                record_data = False
                print('\033[31m'+'SUCCESS!'+f': {sum}'+'\033[0m')
            else:
                self.success = False

            # Observation.state: joint positions from /joint_states
            observation_state = copy.copy(joint_states['positions'].tolist() if len(joint_states['positions']) > 0 else [])
            
            # Action: joint positions from /joint_command (sent by control_right_arm.py)
            action_to_save = copy.copy(action.tolist() if len(action) > 0 else [])
            
            self.df.loc[self.column_index] = [observation_state, action_to_save, self.episode_index, self.frame_index, self.time_stamp, sum, self.done, self.success, self.index, 0]
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

                # Save video with H.264 codec (better compatibility)
                # Try 'avc1' first (macOS), then 'x264', then fallback to 'mp4v'
                video_path = self.rgb_vid_dir + video_file_name
                
                # H.264 codec - avc1 for better compatibility
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                out_rgb = cv2.VideoWriter(video_path, fourcc, self.Hz, (vid_W, vid_H))
                
                # Fallback to mp4v if avc1 doesn't work
                if not out_rgb.isOpened():
                    print("⚠ avc1 codec failed, trying mp4v...")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out_rgb = cv2.VideoWriter(video_path, fourcc, self.Hz, (vid_W, vid_H))
                
                for frame_rgb in self.rgb_image_array:
                    out_rgb.write(frame_rgb)
                out_rgb.release()
                print("The RGB video is generated (H.264/avc1)!")

                # Update episode metadata for LeRobot v3.0
                episode_length = self.frame_index
                self._save_episode_metadata(episode_length)
                
                # Generate meta files
                self._generate_meta_files()

                # Reset for next episode - exactly like original
                self.data_recorded = True
                self.episode_index += 1  # Increment episode for next recording
                self.frame_index = 0
                self.time_stamp = 0.0
                self.column_index = 0
                self.start_recording = False
                self.success = False
                self.done = False
                
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
            "tasks": ["robot_arm_control"],
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
        
        print(f"📝 Episode {self.episode_index} metadata kaydedildi: {episode_length} frame")

    def _generate_meta_files(self):
        """LeRobot v3.0 formatında tüm meta dosyalarını oluştur"""
        import json
        
        # Joint isimleri (observation.state ve action için)
        joint_names = [
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
                    "shape": [len(joint_names)],
                    "names": joint_names
                },
                "action": {
                    "dtype": "float32",
                    "shape": [len(joint_names)],
                    "names": joint_names
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
        
        info_path = os.path.join(self.meta_dir, "info.json")
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4, ensure_ascii=False)
        print(f"✅ info.json oluşturuldu")
        
        # 2. tasks.parquet
        tasks_df = pd.DataFrame({
            "task_index": [0],
        }, index=["robot_arm_control"])
        tasks_path = os.path.join(self.meta_dir, "tasks.parquet")
        tasks_df.to_parquet(tasks_path)
        print(f"✅ tasks.parquet oluşturuldu")
        
        # 3. episodes/chunk-000/file-000.parquet
        if self.all_episodes_meta:
            episodes_df = pd.DataFrame(self.all_episodes_meta)
            episodes_path = os.path.join(self.episodes_meta_dir, "file-000.parquet")
            episodes_df.to_parquet(episodes_path, index=False)
            print(f"✅ episodes meta parquet oluşturuldu: {len(self.all_episodes_meta)} episode")
        
        # 4. stats.json (basit istatistikler - joint isimleriyle)
        stats = {
            "observation.state": {
                "min": {name: -3.14 for name in joint_names},
                "max": {name: 3.14 for name in joint_names},
                "mean": {name: 0.0 for name in joint_names},
                "std": {name: 1.0 for name in joint_names},
                "count": self.total_frames
            },
            "action": {
                "min": {name: -3.14 for name in joint_names},
                "max": {name: 3.14 for name in joint_names},
                "mean": {name: 0.0 for name in joint_names},
                "std": {name: 1.0 for name in joint_names},
                "count": self.total_frames
            }
        }
        stats_path = os.path.join(self.meta_dir, "stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)
        print(f"✅ stats.json oluşturuldu")
        
        print(f"🎉 Tüm meta dosyaları oluşturuldu! Toplam: {len(self.all_episodes_meta)} episode, {self.total_frames} frame")

    def reset_robot_after_episode(self):
        """Placeholder: post-episode reset removed (no /joint_trajectory topic)."""
        print("ℹ Post-episode home reset skipped (no /joint_trajectory controller).")


def interactive_control(controller):
    """İnteraktif klavye kontrolü - control_right_arm.py ile aynı"""
    global record_data
    
    print("\n" + "="*70)
    print("RIGHT ARM & FINGERS CONTROL + DATA RECORDING")
    print("="*70)
    print(f"\nROS2 Topic: /joint_command")
    print(f"Controlled joints: {len(controller.joint_names)}")
    print("\nControls:")
    print("  Shoulder:")
    print("    a / w  -> right_shoulder_link_joint (+/- 0.01 rad)")
    print("    s / d  -> right_arm_top_link_joint (+/- 0.01 rad)")
    print("  Elbow:")
    print("    e / r  -> right_arm_bottom_link_joint (+/- 0.01 rad)")
    print("    f / t  -> right_forearm_link_joint (+/- 0.01 rad)")
    print("  Wrist:")
    print("    q / y  -> wrist_pitch_joint_r (+/- 0.01 rad)")
    print("    z / x  -> wrist_roll_joint_r (+/- 0.01 rad)")
    print("    c / v  -> thumb_joint_roll_r (+/- 0.01 rad)")
    print("    b      -> thumb_joint_roll_r toggle open/close")
    print("  Fingers (3-step cycle):")
    print("    g -> Cycle: Full open -> Half closed -> Full closed")
    print("\n  Recording:")
    print("    SPACE -> Start/Stop Recording")
    print("\n  Other:")
    print("    h -> Home (slow/stepwise)")
    print("    p -> Show positions")
    print("    ESC or Ctrl+C -> Exit")
    print("\n" + "-"*70)
    print("Press any key to start...\n")
    
    # ROS2 bağlantısını test et
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
            key = get_key()
            
            # ESC ile çıkış
            if ord(key) == 27:
                print("\nÇıkılıyor...")
                break
            
            # SPACE ile kayıt başlat/durdur
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
            
            # Özel komutlar
            if key_lower == 'h':
                controller.home()
                continue
            
            if key_lower == 'p':
                controller.show_positions()
                continue
            
            # Parmak kontrolleri (3 aşama)
            if key_lower == 'g':
                controller.cycle_fingers()
                continue

            # Thumb-roll aç/kapat tek tuş
            if key_lower == 'b':
                controller.toggle_thumb_roll()
                continue
            
            # Klavye mapping'den kontrol (kol kontrolleri)
            if key_lower in controller.key_mappings:
                joint_name, delta = controller.key_mappings[key_lower]
                controller.update_position(joint_name, delta)
        
        except KeyboardInterrupt:
            print("\n\nDurduruldu.")
            break
        except Exception as e:
            print(f"Hata: {e}")


if __name__ == '__main__':
    rclpy.init(args=None)

    get_poses_subscriber = Get_Poses_Subscriber()
    joint_states_subscriber = JointStates_Subscriber()  # NEW: Joint states subscriber
    joint_command_subscriber = JointCommand_Subscriber()  # NEW: Joint command subscriber (for action)
    rgb_camera_subscriber = RGB_Camera_Subscriber()  # NEW: RGB camera subscriber
    robot_keyboard_controller = Robot_Keyboard_Controller()
    wrist_camera_subscriber = Wrist_Camera_Subscriber()  # Keep for backward compatibility
    top_camera_subscriber = Top_Camera_Subscriber()  # Keep for backward compatibility
    data_recorder = Data_Recorder()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(get_poses_subscriber)
    executor.add_node(joint_states_subscriber)  # NEW
    executor.add_node(joint_command_subscriber)  # NEW
    executor.add_node(rgb_camera_subscriber)  # NEW
    executor.add_node(robot_keyboard_controller)
    executor.add_node(wrist_camera_subscriber)
    executor.add_node(top_camera_subscriber)
    executor.add_node(data_recorder)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    # İnteraktif kontrolü başlat (control_right_arm.py gibi)
    try:
        interactive_control(robot_keyboard_controller)
    except KeyboardInterrupt:
        print("\n\nDurduruldu.")
    finally:
        rclpy.shutdown()
        executor_thread.join()