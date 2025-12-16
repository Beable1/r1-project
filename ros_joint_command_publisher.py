#!/usr/bin/env python3
"""
ROS2 Joint Command Publisher

Bu script robotun kollarını kontrol etmek için ROS2 /joint_command topic'ine
pozisyon komutları gönderir.

Kullanım:
    python ros_joint_command_publisher.py

Ortam Değişkenleri:
    ROS_DOMAIN_ID=0          ROS2 domain ID (varsayılan: 0)
    JOINT_COMMAND_TOPIC     Topic adı (varsayılan: /joint_command)
    PUBLISH_RATE            Yayınlama hızı Hz (varsayılan: 10)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import sys
import time


class JointCommandPublisher(Node):
    def __init__(self):
        super().__init__('joint_command_publisher')
        
        # Topic adını ortam değişkeninden al veya varsayılan kullan
        topic_name = self.declare_parameter('topic_name', '/joint_command').value
        publish_rate = self.declare_parameter('publish_rate', 10.0).value
        
        # Publisher oluştur
        self.publisher = self.create_publisher(
            JointState,
            topic_name,
            10
        )
        
        # Timer oluştur (periyodik yayınlama için)
        timer_period = 1.0 / publish_rate
        self.timer = self.create_timer(timer_period, self.publish_command)
        
        self.get_logger().info(f'Joint Command Publisher başlatıldı')
        self.get_logger().info(f'Topic: {topic_name}')
        self.get_logger().info(f'Yayınlama hızı: {publish_rate} Hz')
        
        # Robotun joint isimleri - rod_graph.usda'daki JointNameArray ile eşleşmeli
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
        
        # Başlangıç pozisyonları (her joint için radyan cinsinden)
        self.target_positions = [0.0] * len(self.joint_names)
        
    def set_joint_positions(self, positions):
        """
        Joint pozisyonlarını ayarla.
        
        Args:
            positions: List veya dict. List ise joint_names sırasına göre,
                      dict ise {joint_name: position} formatında
        """
        if isinstance(positions, dict):
            # Dict formatında gelirse
            for i, joint_name in enumerate(self.joint_names):
                if joint_name in positions:
                    self.target_positions[i] = float(positions[joint_name])
        elif isinstance(positions, (list, tuple)):
            # List formatında gelirse
            if len(positions) != len(self.joint_names):
                self.get_logger().warn(
                    f'Pozisyon sayısı ({len(positions)}) joint sayısına ({len(self.joint_names)}) eşit değil!'
                )
            for i, pos in enumerate(positions[:len(self.joint_names)]):
                self.target_positions[i] = float(pos)
        else:
            self.get_logger().error('Geçersiz pozisyon formatı! List veya dict olmalı.')
            
    def publish_command(self):
        """JointState komutunu yayınla"""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = list(self.target_positions)
        msg.velocity = [0.0] * len(self.joint_names)
        msg.effort = [0.0] * len(self.joint_names)

        self.publisher.publish(msg)
        self.get_logger().debug(f'Komut yayınlandı: {self.target_positions}')
    
    def move_to_position(self, positions, duration_sec=3.0):
        """
        Robotu belirtilen pozisyona hareket ettir (JointState).
        
        Args:
            positions: Joint pozisyonları (list veya dict)
            duration_sec: Hareket süresi (saniye) - JointState için bilgilendirici
        """
        self.set_joint_positions(positions)

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = list(self.target_positions)
        msg.velocity = [0.0] * len(self.joint_names)
        msg.effort = [0.0] * len(self.joint_names)

        self.publisher.publish(msg)
        self.get_logger().info(f'Pozisyon komutu gönderildi: {positions} (target in {duration_sec}s)')

        # Hedef pozisyonları güncelle
        self.target_positions = list(self.target_positions)


def main(args=None):
    rclpy.init(args=args)
    
    publisher = JointCommandPublisher()
    
    # Örnek kullanım: Robotu belirli pozisyonlara hareket ettir
    try:
        # İlk pozisyon (örnek)
        print("\n=== Robot Kontrolü ===")
        print("Komut göndermek için aşağıdaki örnekleri kullanabilirsiniz:\n")
        
        # Örnek 1: Tüm joint'leri 0.5 radyan pozisyonuna al
        print("Örnek 1: Tüm joint'leri 0.5 radyan pozisyonuna al")
        positions = [0.5] * len(publisher.joint_names)
        publisher.move_to_position(positions, duration_sec=3.0)
        time.sleep(3.5)
        
        # Örnek 2: Sıfır pozisyonuna dön
        print("Örnek 2: Sıfır pozisyonuna dön")
        positions = [0.0] * len(publisher.joint_names)
        publisher.move_to_position(positions, duration_sec=3.0)
        time.sleep(3.5)
        
        # Örnek 3: Belirli joint'leri kontrol et (dict formatında)
        if len(publisher.joint_names) > 0:
            print(f"Örnek 3: {publisher.joint_names[0]} joint'ini 0.3 radyan pozisyonuna al")
            positions = {publisher.joint_names[0]: 0.3}
            publisher.move_to_position(positions, duration_sec=2.0)
            time.sleep(2.5)
        
        print("\nSürekli yayınlama modu başlatıldı. Çıkmak için CTRL+C'ye basın.")
        print("Pozisyonları değiştirmek için publisher.set_joint_positions() metodunu kullanın.\n")
        
        # Sürekli yayınlama (timer ile)
        rclpy.spin(publisher)
        
    except KeyboardInterrupt:
        print("\n\nDurduruldu.")
    finally:
        publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

