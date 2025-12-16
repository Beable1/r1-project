#!/usr/bin/env python3
"""
Right Arm & Fingers Control Script

Controls the robot’s right arm and fingers via keyboard.
Sends sensor_msgs/JointState to ROS2 /joint_command.

Usage:
    python control_right_arm.py

Controls:
    Shoulder:
        a / w  -> right_shoulder_link_joint (+/- 0.01 rad)
        s / d  -> right_arm_top_link_joint (+/- 0.01 rad)
    
    Elbow:
        e / r  -> right_arm_bottom_link_joint (+/- 0.01 rad)
        f / t  -> right_forearm_link_joint (+/- 0.01 rad)
    
    Wrist:
        q / y  -> wrist_pitch_joint_r (+/- 0.01 rad)
        z / x  -> wrist_roll_joint_r (+/- 0.01 rad)
        c / v  -> thumb_joint_roll_r (+/- 0.01 rad)
    
    Fingers (3-step cycle):
        g      -> Cycle fingers: Full open -> Half closed -> Full closed
    
    Other:
        h      -> Home (slow, stepwise)
        p      -> Show current positions
        ESC    -> Exit
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import sys
import time
import threading
import termios
import tty


class RightArmController(Node):
    def __init__(self):
        super().__init__('right_arm_controller')
        
        topic_name = self.declare_parameter('topic_name', '/joint_command').value
        self.publisher = self.create_publisher(JointState, topic_name, 10)
        
        # Sadece sağ kol ve parmak joint'leri (orijinal listeden)
        # Mapping: kullanıcı isimleri -> gerçek joint isimleri
        self.joint_mapping = {
            "right_shoulder_pitch": "right_shoulder_link_joint",
            "right_shoulder_roll": "right_arm_top_link_joint", 
            "right_elbow": "right_arm_bottom_link_joint",
            "wrist_pitch_r": "wrist_pitch_joint_r",
            "wrist_roll_r": "wrist_roll_joint_r",
        }
        
        # Tüm sağ kol ve parmak joint'leri (gerçek isimlerle)
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
        
        # Mevcut pozisyonlar (başlangıçta sıfır)
        self.current_positions = {name: 0.0 for name in self.joint_names}
        
        # Adım boyutu (radyan) - daha yumuşak hareket için küçültüldü
        self.step_size = 0.01
        
        # Parmak joint'leri (tüm parmak eklemleri)
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
            "thumb_joint_roll_r",  # Baş parmak roll (g tuşu ile birlikte döngüye dahil)
        ]
        
        # Parmak pozisyonları (3 aşama)
        self.finger_full_open = 0.1  # Full açık pozisyon (radyan)
        self.finger_half_closed = 1.0  # Orta kapalı pozisyon (radyan)
        self.finger_full_closed = 3.0  # Full kapalı pozisyon (radyan)
        self.finger_state = 0  # 0=full açık, 1=orta kapalı, 2=full kapalı
        
        # Klavye mapping (a w s d e r f t g + bilek için q/y/z/x + baş parmak roll için c/v)
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
        
        self.get_logger().info(f'Right Arm Controller başlatıldı')
        self.get_logger().info(f'Topic: {topic_name}')
        self.get_logger().info(f'Kontrol edilen joint sayısı: {len(self.joint_names)}')
        print(f'\n[INFO] Right Arm Controller başlatıldı')
        print(f'[INFO] Topic: {topic_name}')
        print(f'[INFO] Joint sayısı: {len(self.joint_names)}')
        print(f'[INFO] İlk 5 joint: {self.joint_names[:5]}')
        
    def update_position(self, joint_name, delta):
        """Joint pozisyonunu güncelle ve komut gönder"""
        if joint_name not in self.current_positions:
            print(f'HATA: Bilinmeyen joint: {joint_name}')
            print(f'Mevcut jointler: {self.joint_names}')
            return
        
        # Pozisyonu güncelle
        old_pos = self.current_positions[joint_name]
        self.current_positions[joint_name] += delta
        new_pos = self.current_positions[joint_name]
        
        # Komut gönder
        try:
            self.send_command()
        except Exception as e:
            print(f'HATA: Komut gönderilemedi: {e}')
            # Geri al
            self.current_positions[joint_name] = old_pos
    
    def send_command(self):
        """Mevcut pozisyonları komut olarak gönder (sensor_msgs/JointState formatında)"""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        
        # Joint isimleri ve pozisyonları
        msg.name = list(self.joint_names)
        msg.position = [self.current_positions[name] for name in self.joint_names]
        
        # Hızlar ve effort'lar (opsiyonel, sıfır olarak ayarla)
        msg.velocity = [0.0] * len(self.joint_names)
        msg.effort = [0.0] * len(self.joint_names)
        
        # Yayınla
        self.publisher.publish(msg)
    
    def cycle_fingers(self):
        """Parmakları 3 aşamada döngüye al: full açık -> orta kapalı -> full kapalı -> full açık"""
        # State'i artır ve mod 3 al
        self.finger_state = (self.finger_state + 1) % 3
        
        # State'e göre pozisyonu belirle
        if self.finger_state == 0:  # Full açık
            position = self.finger_full_open
        elif self.finger_state == 1:  # Orta kapalı
            position = self.finger_half_closed
        else:  # Full kapalı (state == 2)
            position = self.finger_full_closed
        
        # Tüm parmak joint'lerini ayarla
        for joint_name in self.finger_joints:
            self.current_positions[joint_name] = position
        
        self.send_command()
    
    def home(self):
        """Tüm sağ kol joint'lerini sıfır pozisyonuna yavaşça al (adım adım)"""
        # Her joint için mevcut pozisyonu kaydet
        target_positions = {name: 0.0 for name in self.joint_names}
        
        # Parmakları da sıfırla
        for finger_name in self.finger_joints:
            target_positions[finger_name] = 0.0
        
        # Adım adım home pozisyonuna git
        max_steps = 100  # Maksimum adım sayısı
        for step in range(max_steps):
            all_reached = True
            for joint_name in self.joint_names:
                current = self.current_positions[joint_name]
                target = target_positions[joint_name]
                
                # Eğer hedefe ulaşılmadıysa adım at
                if abs(current - target) > 0.001:  # 0.001 radyan tolerans
                    all_reached = False
                    # Hedefe doğru adım at (step_size kadar)
                    if current > target:
                        self.current_positions[joint_name] = max(target, current - self.step_size)
                    else:
                        self.current_positions[joint_name] = min(target, current + self.step_size)
            
            # Komut gönder
            self.send_command()
            time.sleep(0.01)  # Her adım arasında kısa bekleme
            
            # Tüm joint'ler hedefe ulaştıysa dur
            if all_reached:
                break
        
        # Son durumu ayarla (tam olarak 0.0)
        for joint_name in self.joint_names:
            self.current_positions[joint_name] = 0.0
        self.finger_state = 0  # Full açık durumuna dön
        self.send_command()
    
    def show_positions(self):
        """current pozitions"""
        print("\n" + "="*60)
        print("current joint pozitions")
        print("="*60)
        for joint_name in self.joint_names:
            print(f"  {joint_name:30s}: {self.current_positions[joint_name]:7.3f} rad")
        print("="*60 + "\n")


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


def interactive_control(controller):
    """İnteraktif klavye kontrolü"""
    print("\n" + "="*70)
    print("RIGHT ARM & FINGERS CONTROL")
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
    print("  Fingers (3-step cycle):")
    print("    g -> Cycle: Full open -> Half closed -> Full closed")
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
        test_msg.name = controller.joint_names[:1]  # Sadece test için
        test_msg.position = [0.0]
        test_msg.velocity = [0.0]
        test_msg.effort = [0.0]
        controller.publisher.publish(test_msg)
        print("✓ ROS2 connection succesfuln")
    except Exception as e:
        print(f"⚠ ROS2 connection warning: {e}")
       
    
    while rclpy.ok():
        try:
            key = get_key()
            
            # ESC veya Ctrl+C ile çıkış
            if ord(key) == 27:  # ESC
                print("\nÇıkılıyor...")
                break
            
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
            
            # Klavye mapping'den kontrol (kol kontrolleri)
            if key_lower in controller.key_mappings:
                joint_name, delta = controller.key_mappings[key_lower]
                controller.update_position(joint_name, delta)
            else:
                # Bilinmeyen tuş - sessizce yoksay
                pass
        
        except KeyboardInterrupt:
            print("\n\nDurduruldu.")
            break
        except Exception as e:
            print(f"Hata: {e}")


def main(args=None):
    rclpy.init(args=args)
    
    controller = RightArmController()
    
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

