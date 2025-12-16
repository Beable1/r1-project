#!/usr/bin/env python3
"""
Robot Kol Kontrol Scripti

Bu script robotun kollarını interaktif olarak kontrol etmenizi sağlar.
ROS2 /joint_command topic'ine komut gönderir.

Kullanım:
    python control_robot_arms.py

Örnek Komutlar:
    set 0 0.5 0.3    -> İlk 3 joint'i belirtilen pozisyonlara al
    move Chest_link_joint 0.5  -> Belirli bir joint'i hareket ettir
    home              -> Tüm joint'leri sıfır pozisyonuna al
    quit              -> Çıkış
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import sys
import time
import threading


class RobotArmController(Node):
    def __init__(self):
        super().__init__('robot_arm_controller')
        
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
        
        # Hızları hazırla
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
    """İnteraktif kontrol döngüsü"""
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


def main(args=None):
    rclpy.init(args=args)
    
    controller = RobotArmController()
    
    # ROS2 spin'i ayrı thread'de çalıştır
    spin_thread = threading.Thread(target=rclpy.spin, args=(controller,), daemon=True)
    spin_thread.start()
    
    # İnteraktif kontrolü başlat
    try:
        interactive_control(controller)
    except KeyboardInterrupt:
        print("\n\nDurduruldu.")
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

