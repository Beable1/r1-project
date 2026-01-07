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

# Mouse control
try:
    from pynput import mouse
    from pynput.mouse import Controller as MouseController
    MOUSE_AVAILABLE = True
    mouse_controller = MouseController()
except ImportError:
    print("⚠ pynput modülü bulunamadı. Mouse kontrolü devre dışı. Yüklemek için: pip install pynput")
    MOUSE_AVAILABLE = False
    mouse_controller = None

# Ekran boyutunu al (mouse kilitleme için)
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

# Mouse kontrol değişkenleri
mouse_control_enabled = False
mouse_sensitivity = 0.002  # Mouse hareket hassasiyeti (radyan/pixel) - daha hızlı
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

def get_all_keys():
    """Aynı anda basılan tüm tuşları al (non-blocking, çoklu tuş desteği)"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    keys = []
    try:
        tty.setraw(sys.stdin.fileno())
        # İlk tuşu bekle (blocking)
        ch = sys.stdin.read(1)
        keys.append(ch)
        # Kısa süre içinde gelen diğer tuşları da oku (non-blocking)
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
        
        # Maksimum hız (radyan/saniye) - mouse için hızlı tepki
        self.max_velocity = 4.0  # 3.0 rad/s - daha hızlı!
        self.control_rate = 200  # Hz - 200Hz kontrol döngüsü (düşük gecikme)
        self.smooth_step = self.max_velocity / self.control_rate  # Her adımda maksimum hareket

        # Parmaklar biraz daha hızlı olabilir: ayrı hız limiti
        self.finger_max_velocity = 1.5  # rad/s - daha hızlı parmak hareketi
        self.finger_smooth_step = self.finger_max_velocity / self.control_rate

        # Effort/Kuvvet limitleri (Newton-metre)
        # Parmaklar için düşük tork -> objeyi ezmez, fırlatmaz
        self.finger_max_effort = 5.0   # Nm - parmaklar için (düşük tutun!)
        self.arm_max_effort = 50.0     # Nm - kol joint'leri için

        # Non-blocking hareket güncelleyici (200Hz): current_positions -> target_positions
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

            # Baş parmak roll (3x daha hızlı)
            'c': ("thumb_joint_roll_r", +self.step_size * 3),
            'v': ("thumb_joint_roll_r", -self.step_size * 3),
        }
        
        # Mouse kontrol ayarları
        self.mouse_enabled = False
        self.mouse_sensitivity = 0.002  # radyan/pixel - daha hızlı kontrol
        self.mouse_mode = 0  # 0: shoulder, 1: elbow, 2: wrist
        self.mouse_mode_names = ['Shoulder (Omuz)', 'Elbow (Dirsek)', 'Wrist (Bilek)']
        
        # Mouse ile kontrol edilecek joint'ler (mod başına)
        self.mouse_joint_mappings = {
            0: {  # Shoulder mode
                'x': 'right_shoulder_link_joint',  # Yatay hareket
                'y': 'right_arm_top_link_joint',   # Dikey hareket
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
        
        # Mouse listener başlat (eğer pynput varsa)
        self.mouse_listener = None
        if MOUSE_AVAILABLE:
            self._setup_mouse_listener()
        
        self.get_logger().info(f'Robot Keyboard Controller başlatıldı')
        self.get_logger().info(f'Topic: {topic_name}')
        self.get_logger().info(f'Joint sayısı: {len(self.joint_names)}')
        if MOUSE_AVAILABLE:
            self.get_logger().info('🖱️  Mouse kontrolü hazır (M tuşu ile aç/kapat)')
        
        # Başlangıçta home pozisyonuna git
        self.send_command()
    
    def _setup_mouse_listener(self):
        """Mouse listener'ı başlat"""
        global mouse_delta_x, mouse_delta_y, mouse_left_button, mouse_right_button, mouse_middle_button
        
        def on_move(x, y):
            global mouse_last_x, mouse_last_y, mouse_delta_x, mouse_delta_y
            if self.mouse_enabled:
                # Kilitleme pozisyonundan itibaren delta hesapla (birikimli DEĞİL)
                if hasattr(self, 'lock_position') and self.lock_position and mouse_controller:
                    # Delta = mevcut pozisyon - kilit pozisyonu (doğrudan, biriktirme yok)
                    mouse_delta_x = x - self.lock_position[0]
                    mouse_delta_y = y - self.lock_position[1]
                    # İmleci geri gönder
                    mouse_controller.position = self.lock_position
                    # Last pozisyonu kilit pozisyonuna ayarla
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
            # Scroll ile hassasiyet ayarla
            global mouse_sensitivity
            if self.mouse_enabled:
                if dy > 0:
                    self.mouse_sensitivity = min(0.005, self.mouse_sensitivity * 1.2)
                else:
                    self.mouse_sensitivity = max(0.00005, self.mouse_sensitivity / 1.2)
                print(f"🖱️  Mouse hassasiyeti: {self.mouse_sensitivity:.5f}")
        
        try:
            self.mouse_listener = mouse.Listener(
                on_move=on_move,
                on_click=on_click,
                on_scroll=on_scroll
            )
            self.mouse_listener.start()
            self.get_logger().info('🖱️  Mouse listener başlatıldı')
        except Exception as e:
            self.get_logger().warning(f'Mouse listener başlatılamadı: {e}')
            self.mouse_listener = None
    
    def toggle_mouse_control(self):
        """Mouse kontrolünü aç/kapat - aktifken imleç sağ alt köşeye kilitlenir"""
        global mouse_delta_x, mouse_delta_y
        self.mouse_enabled = not self.mouse_enabled
        mouse_delta_x = 0
        mouse_delta_y = 0
        
        if self.mouse_enabled:
            # İmleci sağ alt köşeye taşı
            if mouse_controller:
                self.lock_position = (SCREEN_WIDTH - 50, SCREEN_HEIGHT - 50)
                mouse_controller.position = self.lock_position
            print(f"\n🖱️  MOUSE KONTROL AKTİF - Mod: {self.mouse_mode_names[self.mouse_mode]}")
            print(f"    🔒 İmleç sağ alt köşeye kilitlendi ({SCREEN_WIDTH-50}, {SCREEN_HEIGHT-50})")
            print("    Sol tık: Parmak kapat | Sağ tık: Parmak aç | Orta tık: Mod değiştir")
            print("    Scroll: Hassasiyet ayarla | M: Mouse kapat")
        else:
            self.lock_position = None
            print("\n⌨️  Klavye kontrolüne dönüldü - İmleç serbest")
    
    def cycle_mouse_mode(self):
        """Mouse kontrol modunu değiştir (Shoulder -> Elbow -> Wrist)"""
        self.mouse_mode = (self.mouse_mode + 1) % 3
        print(f"🖱️  Mouse modu: {self.mouse_mode_names[self.mouse_mode]}")
    
    def process_mouse_input(self):
        """FPS TARZI Mouse kontrolü - SİMÜLASYONDAKİ GERÇEK POZİSYONU KULLAN
        
        Birikme YOK çünkü:
        - Her komutta simülasyonun GERÇEK pozisyonunu oku (/joint_states)
        - O pozisyona delta ekle
        - Simülasyon yetişemediyse önemli değil, hep güncel pozisyondan başlıyoruz
        """
        global mouse_delta_x, mouse_delta_y, mouse_left_button, mouse_right_button, mouse_middle_button
        global joint_states
        
        if not self.mouse_enabled:
            return
        
        # Delta'yı AL ve HEMEN SIFIRLA
        dx = mouse_delta_x
        dy = mouse_delta_y
        mouse_delta_x = 0
        mouse_delta_y = 0
        
        # Simülasyondan GERÇEK pozisyonları al
        actual_positions = {}
        if len(joint_states['names']) > 0 and len(joint_states['positions']) > 0:
            actual_positions = dict(zip(joint_states['names'], joint_states['positions']))
        
        # Hareket mapping
        mapping = self.mouse_joint_mappings.get(self.mouse_mode, {})
        
        # X ekseni
        if 'x' in mapping and abs(dx) > 0:
            joint_name = mapping['x']
            # GERÇEK pozisyonu al (simülasyondan)
            if joint_name in actual_positions:
                real_pos = actual_positions[joint_name]
            else:
                real_pos = self.current_positions.get(joint_name, 0)
            
            # Delta hesapla ve sınırla (FPS sensitivity)
            max_move = 0.1  # radyan - tek seferde maksimum
            delta = dx * self.mouse_sensitivity
            delta = max(-max_move, min(max_move, delta))
            
            # Yeni hedef = GERÇEK pozisyon + delta
            new_pos = real_pos + delta
            self.current_positions[joint_name] = new_pos
            self.target_positions[joint_name] = new_pos
        
        # Y ekseni
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
        
        # Sol tık: parmakları kapat
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
        
        # Sağ tık: parmakları aç
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
        
        # Komut gönder
        self.send_command()
    
    def toggle_mouse_control(self):
        """Mouse kontrolünü aç/kapat - aktifken imleç sağ alt köşeye kilitlenir"""
        global mouse_delta_x, mouse_delta_y
        self.mouse_enabled = not self.mouse_enabled
        mouse_delta_x = 0
        mouse_delta_y = 0
        
        if self.mouse_enabled:
            # Baz pozisyonları sıfırla (yeni oturum)
            if hasattr(self, '_mouse_base_positions'):
                del self._mouse_base_positions
            
            # İmleci sağ alt köşeye taşı
            if mouse_controller:
                self.lock_position = (SCREEN_WIDTH - 50, SCREEN_HEIGHT - 50)
                mouse_controller.position = self.lock_position
            print(f"\n🖱️  MOUSE KONTROL AKTİF - Mod: {self.mouse_mode_names[self.mouse_mode]}")
            print(f"    🔒 İmleç sağ alt köşeye kilitlendi ({SCREEN_WIDTH-50}, {SCREEN_HEIGHT-50})")
            print("    Sol tık: Parmak kapat | Sağ tık: Parmak aç | Orta tık: Mod değiştir")
            print("    Scroll: Hassasiyet ayarla | M: Mouse kapat")
        else:
            self.lock_position = None
            print("\n⌨️  Klavye kontrolüne dönüldü - İmleç serbest")
    
    def _motion_update(self):
        """Non-blocking hareket: hedefe max_velocity ile yaklaş ve komut gönder."""
        # Mouse input'u işle
        self.process_mouse_input()
        
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
        self.get_logger().info('🔍 Joint States Subscriber başlatıldı - /joint_states topic dinleniyor...')
    
    def joint_states_callback(self, data):
        global joint_states
        joint_states['names'] = list(data.name)
        joint_states['positions'] = np.array(data.position, dtype=float)
        # Debug: İlk veri geldiğinde log
        if len(joint_states['positions']) > 0:
            self.get_logger().info(f'✅ Joint states alındı: {len(joint_states["positions"])} joint, ilk 3: {joint_states["positions"][:3]}', once=True)

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
            wrist_camera_image = cv2.resize(bridge.imgmsg_to_cv2(data, "bgr8"), (vid_W, vid_H), cv2.INTER_LINEAR)
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
            top_camera_image = cv2.resize(bridge.imgmsg_to_cv2(data, "bgr8"), (vid_W, vid_H), cv2.INTER_LINEAR)
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
        print(f"📊 Mevcut episode sayısı: {self.episode_index}, Sonraki index: {self.index}")

        self.df = pd.DataFrame(columns=['observation.state', 'action', 'episode_index', 'frame_index', 'timestamp', 'index', 'task_index'])
        self.frame_index = 0
        self.time_stamp = 0.0
        self.column_index = 0

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
            
            # DEBUG: Check if joint states are being received
            if self.frame_index == 0:  # First frame
                print('\n' + '='*70)
                print('🔍 DATA RECORDING DEBUG (First Frame)')
                print('='*70)
                
                # Check observation_state
                if len(observation_state) == 0:
                    print('\033[33m'+'⚠️  WARNING: observation_state BOŞ! /joint_states topic\'i veri göndermiyor olabilir!'+'\033[0m')
                elif np.allclose(observation_state, 0.0):
                    print('\033[33m'+'⚠️  WARNING: observation_state tümü 0.0! Joint states doğru gelmiyor olabilir!'+'\033[0m')
                    print(f'   observation_state: {observation_state}')
                else:
                    print('\033[32m'+f'✅ observation_state OK: {len(observation_state)} joint'+'\033[0m')
                    print(f'   İlk 5 değer: {[f"{v:.4f}" for v in observation_state[:5]]}')
                    non_zero_count = sum(1 for v in observation_state if abs(v) > 0.001)
                    print(f'   Sıfır olmayan joint sayısı: {non_zero_count}/{len(observation_state)}')
                
                # Check action
                if len(action_to_save) == 0:
                    print('\033[33m'+'⚠️  WARNING: action BOŞ!'+'\033[0m')
                else:
                    print('\033[32m'+f'✅ action OK: {len(action_to_save)} joint'+'\033[0m')
                    print(f'   İlk 5 değer: {[f"{v:.4f}" for v in action_to_save[:5]]}')
                    non_zero_count = sum(1 for v in action_to_save if abs(v) > 0.001)
                    print(f'   Sıfır olmayan joint sayısı: {non_zero_count}/{len(action_to_save)}')
                
                # Check raw joint_states data
                if len(joint_states['names']) > 0:
                    print(f'\n📊 Raw /joint_states topic:')
                    print(f'   Toplam joint sayısı: {len(joint_states["names"])}')
                    print(f'   İlk 3 joint: {joint_states["names"][:3]}')
                    print(f'   İlk 3 pozisyon: {[f"{v:.4f}" for v in joint_states["positions"][:3]]}')
                
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
        """LeRobot v2.1 formatında tüm meta dosyalarını oluştur (JSONL based)"""
        import json
        import glob
        
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
        
        # Compute real statistics from all parquet files
        all_obs_states = []
        all_actions = []
        episode_stats_list = []
        
        parquet_files = sorted(glob.glob(os.path.join(self.log_dir, "*.parquet")))
        for pf in parquet_files:
            try:
                df = pd.read_parquet(pf)
                ep_idx = df['episode_index'].iloc[0] if 'episode_index' in df.columns else 0
                
                # Extract observation.state and action arrays
                obs_data = np.array([np.array(x) for x in df['observation.state'].values])
                act_data = np.array([np.array(x) for x in df['action'].values])
                
                all_obs_states.append(obs_data)
                all_actions.append(act_data)
                
                # Per-episode stats
                ep_stats = {
                    "episode_index": int(ep_idx),
                    "stats": {
                        "observation.state": {
                            "min": obs_data.min(axis=0).tolist(),
                            "max": obs_data.max(axis=0).tolist(),
                            "mean": obs_data.mean(axis=0).tolist(),
                            "std": obs_data.std(axis=0).tolist()
                        },
                        "action": {
                            "min": act_data.min(axis=0).tolist(),
                            "max": act_data.max(axis=0).tolist(),
                            "mean": act_data.mean(axis=0).tolist(),
                            "std": act_data.std(axis=0).tolist()
                        }
                    }
                }
                episode_stats_list.append(ep_stats)
            except Exception as e:
                print(f"⚠ Stats hesaplama hatası {pf}: {e}")
        
        # Global stats from all data
        if all_obs_states:
            all_obs = np.vstack(all_obs_states)
            all_act = np.vstack(all_actions)
            global_stats = {
                "observation.state": {
                    "min": all_obs.min(axis=0).tolist(),
                    "max": all_obs.max(axis=0).tolist(),
                    "mean": all_obs.mean(axis=0).tolist(),
                    "std": all_obs.std(axis=0).tolist(),
                    "count": len(all_obs)
                },
                "action": {
                    "min": all_act.min(axis=0).tolist(),
                    "max": all_act.max(axis=0).tolist(),
                    "mean": all_act.mean(axis=0).tolist(),
                    "std": all_act.std(axis=0).tolist(),
                    "count": len(all_act)
                }
            }
        else:
            # Fallback if no data
            global_stats = {
                "observation.state": {
                    "min": [-3.14] * len(joint_names),
                    "max": [3.14] * len(joint_names),
                    "mean": [0.0] * len(joint_names),
                    "std": [1.0] * len(joint_names),
                    "count": self.total_frames
                },
                "action": {
                    "min": [-3.14] * len(joint_names),
                    "max": [3.14] * len(joint_names),
                    "mean": [0.0] * len(joint_names),
                    "std": [1.0] * len(joint_names),
                    "count": self.total_frames
                }
            }
        
        # 1. info.json (v2.1 format)
        info = {
            "codebase_version": "v2.1",
            "robot_type": "r1_humanoid",
            "total_episodes": len(self.all_episodes_meta),
            "total_frames": self.total_frames,
            "total_tasks": 1,
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": self.Hz,
            "splits": {"train": f"0:{len(self.all_episodes_meta)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
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
        
        print(f"ℹ️  Dataset format: LeRobot v2.1 (JSONL based)")
        
        info_path = os.path.join(self.meta_dir, "info.json")
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4, ensure_ascii=False)
        print(f"✅ info.json oluşturuldu (v2.1)")
        
        # 2. tasks.jsonl (v2.1 format - JSONL instead of parquet)
        tasks_path = os.path.join(self.meta_dir, "tasks.jsonl")
        with open(tasks_path, 'w') as f:
            task_entry = {"task_index": 0, "task": "robot_arm_control"}
            f.write(json.dumps(task_entry) + "\n")
        print(f"✅ tasks.jsonl oluşturuldu")
        
        # 3. episodes.jsonl (v2.1 format - JSONL instead of parquet)
        episodes_path = os.path.join(self.meta_dir, "episodes.jsonl")
        with open(episodes_path, 'w') as f:
            for ep_meta in self.all_episodes_meta:
                episode_entry = {
                    "episode_index": ep_meta["episode_index"],
                    "tasks": [0],  # task_index as integer
                    "length": ep_meta["length"]
                }
                f.write(json.dumps(episode_entry) + "\n")
        print(f"✅ episodes.jsonl oluşturuldu: {len(self.all_episodes_meta)} episode")
        
        # 4. episodes_stats.jsonl (v2.1 - per-episode statistics)
        episodes_stats_path = os.path.join(self.meta_dir, "episodes_stats.jsonl")
        with open(episodes_stats_path, 'w') as f:
            for ep_stats in episode_stats_list:
                f.write(json.dumps(ep_stats) + "\n")
        print(f"✅ episodes_stats.jsonl oluşturuldu")
        
        # 5. stats.json (global statistics - computed from real data)
        stats_path = os.path.join(self.meta_dir, "stats.json")
        with open(stats_path, 'w') as f:
            json.dump(global_stats, f, indent=4, ensure_ascii=False)
        print(f"✅ stats.json oluşturuldu (gerçek veriden hesaplandı)")
        
        # Cleanup old parquet meta files if they exist
        old_tasks_parquet = os.path.join(self.meta_dir, "tasks.parquet")
        if os.path.exists(old_tasks_parquet):
            os.remove(old_tasks_parquet)
            print(f"🗑️  Eski tasks.parquet silindi")
        
        print(f"🎉 Tüm meta dosyaları oluşturuldu! Toplam: {len(self.all_episodes_meta)} episode, {self.total_frames} frame")

    def reset_robot_after_episode(self):
        """Placeholder: post-episode reset removed (no /joint_trajectory topic)."""
        print("ℹ Post-episode home reset skipped (no /joint_trajectory controller).")


def interactive_control(controller):
    """İnteraktif klavye + mouse kontrolü"""
    global record_data, mouse_middle_button
    
    print("\n" + "="*70)
    print("RIGHT ARM & FINGERS CONTROL + DATA RECORDING")
    print("="*70)
    print(f"\nROS2 Topic: /joint_command")
    print(f"Controlled joints: {len(controller.joint_names)}")
    print("\n⌨️  KLAVYE KONTROL:")
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
    print("\n🖱️  MOUSE KONTROL:")
    if MOUSE_AVAILABLE:
        print("    m      -> Mouse kontrolü aç/kapat")
        print("    n      -> Mouse modu değiştir (Shoulder/Elbow/Wrist)")
        print("    (Mouse aktifken:)")
        print("      Hareket    -> Seçili joint'leri kontrol et")
        print("      Sol tık    -> Parmakları kapat")
        print("      Sağ tık    -> Parmakları aç")
        print("      Orta tık   -> Mod değiştir")
        print("      Scroll     -> Hassasiyet ayarla")
    else:
        print("    ⚠ pynput modülü yüklü değil. Mouse kontrolü kullanılamaz.")
        print("    Yüklemek için: pip install pynput")
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
            # Aynı anda basılan tüm tuşları al
            keys = get_all_keys()
            
            for key in keys:
                # ESC ile çıkış
                if ord(key) == 27:
                    print("\nÇıkılıyor...")
                    rclpy.shutdown()
                    return
                
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
                
                # Mouse kontrolü aç/kapat
                if key_lower == 'm' and MOUSE_AVAILABLE:
                    controller.toggle_mouse_control()
                    continue
                
                # Mouse modu değiştir
                if key_lower == 'n' and MOUSE_AVAILABLE and controller.mouse_enabled:
                    controller.cycle_mouse_mode()
                    continue
                
                # Orta tık ile mouse modu değiştir (mouse aktifken)
                if mouse_middle_button and MOUSE_AVAILABLE and controller.mouse_enabled:
                    mouse_middle_button = False  # Reset
                    controller.cycle_mouse_mode()
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

    # İnteraktif kontrolü başlat (control_right_arm.py gibi)
    try:
        interactive_control(robot_keyboard_controller)
    except KeyboardInterrupt:
        print("\n\nDurduruldu.")
    finally:
        rclpy.shutdown()
        executor_thread.join()