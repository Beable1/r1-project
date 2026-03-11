#!/usr/bin/env python3
"""
Diffusion Policy Delta EE Controller for R1 Humanoid Robot

Trained on delta EE space:
  observation.state = delta EE state (current_ee - prev_ee)
  action = delta EE action (target_ee - current_ee)
  Both: [x, y, z, roll, pitch, yaw, gripper_joints x 11] = 17 dims

Inference flow (10 Hz):
  1. /joint_states -> FK -> current EE state
  2. delta_state = current_ee - prev_ee
  3. Model(delta_state, image) -> delta_action
  4. target_ee = current_ee + delta_action
  5. IK(target_ee[:6]) -> arm joints (6)
  6. gripper = current_gripper + delta_action[6:] (direct)
  7. Send /joint_command
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
import numpy as np
import torch
import threading
import time
import cv2
import copy

import pinocchio as pin

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.processor.pipeline import DataProcessorPipeline


# ---------------------------------------------------------------------------
# IKSolver – same as data_collection_keyboard_delta.py (with custom limits)
# ---------------------------------------------------------------------------
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
            # If anything goes wrong, fall back to URDF limits
            pass

    def solve(self, q_current_full, target_pos, target_rot=None, max_iter=50, eps=1e-3):
        q = np.copy(q_current_full)
        target_se3 = pin.SE3(
            target_rot if target_rot is not None else np.eye(3), target_pos
        )
        damp = 1e-4
        dt = 0.5

        for _ in range(max_iter):
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

            v = J_masked.T @ np.linalg.solve(
                J_masked @ J_masked.T + damp * np.eye(6), err
            )
            q = pin.integrate(self.model, q, v * dt)
            q = np.clip(q, self.model.lowerPositionLimit, self.model.upperPositionLimit)

        return q


# ---------------------------------------------------------------------------
# Controller Node
# ---------------------------------------------------------------------------
class DiffusionPolicyDeltaController(Node):

    URDF_PATH = "/home/beable/IsaacLab/IsaacLab/r1-new/urdf/r1-new.urdf"
    EE_JOINT = "wrist_pitch_joint_r"

    RIGHT_ARM_JOINTS = [
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

    IK_CONTROLLED_JOINTS = [
        "right_shoulder_link_joint",
        "right_arm_top_link_joint",
        "right_arm_bottom_link_joint",
        "right_forearm_link_joint",
        "wrist_roll_joint_r",
        "wrist_pitch_joint_r",
    ]

    HOME_POSITIONS = {
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

    def __init__(self, model_path: str, device: str = "cuda"):
        super().__init__("diffusion_policy_delta_controller")

        self.device = device
        self.vid_H = 360
        self.vid_W = 640
        self.state_dim = 17

        # ---- Pinocchio FK model (same as Data_Recorder) ----
        self.pin_model = pin.buildModelFromUrdf(self.URDF_PATH)
        self.pin_data = self.pin_model.createData()
        if self.pin_model.existJointName(self.EE_JOINT):
            self.ee_joint_id = self.pin_model.getJointId(self.EE_JOINT)
        else:
            self.ee_joint_id = self.pin_model.getFrameId(self.EE_JOINT)

        # ---- IK solver (same config as Robot_Keyboard_Controller) ----
        self.ik_solver = IKSolver(
            self.URDF_PATH, self.EE_JOINT, self.IK_CONTROLLED_JOINTS
        )

        # ---- Sensor state ----
        self.current_joint_positions = {n: 0.0 for n in self.RIGHT_ARM_JOINTS}
        self.current_rgb_image = np.zeros(
            (self.vid_H, self.vid_W, 3), dtype=np.uint8
        )
        self.previous_ee_state: np.ndarray | None = None

        # ---- Load model + processor pipelines ----
        self.get_logger().info(f"Loading model from {model_path}")
        self.policy = DiffusionPolicy.from_pretrained(model_path)
        self.policy.to(self.device)
        self.policy.eval()
        self.policy.reset()

        self.preprocessor = DataProcessorPipeline.from_pretrained(
            model_path, config_filename="policy_preprocessor.json"
        )
        self.postprocessor = DataProcessorPipeline.from_pretrained(
            model_path, config_filename="policy_postprocessor.json"
        )

        n_obs = self.policy.config.n_obs_steps
        n_act = self.policy.config.n_action_steps
        self.get_logger().info(
            f"Model loaded  n_obs_steps={n_obs}  n_action_steps={n_act}  "
            f"horizon={self.policy.config.horizon}"
        )

        # ---- ROS2 subscribers / publisher ----
        self.create_subscription(
            JointState, "/joint_states", self._joint_states_cb, 10
        )
        self.create_subscription(Image, "/rgb", self._rgb_cb, 10)
        self.joint_cmd_pub = self.create_publisher(JointState, "/joint_command", 10)

        # ---- Control loop ----
        self.control_rate = 10.0
        self.create_timer(1.0 / self.control_rate, self._control_loop)
        self.is_running = False
        self.inference_count = 0

        self.get_logger().info("Controller ready  (ENTER = toggle, h = home, q = quit)")

    # ------------------------------------------------------------------
    # FK  (identical to Data_Recorder.get_ee_pose_and_gripper)
    # ------------------------------------------------------------------
    def _joint_positions_to_ee(self, joint_positions: list[float]) -> np.ndarray:
        """joint_positions (17) -> [x,y,z,roll,pitch,yaw, gripper x 11] (17)"""
        q = pin.neutral(self.pin_model)
        for name, p in zip(self.RIGHT_ARM_JOINTS, joint_positions):
            if self.pin_model.existJointName(name):
                j_id = self.pin_model.getJointId(name)
                idx_q = self.pin_model.joints[j_id].idx_q
                if idx_q < self.pin_model.nq:
                    q[idx_q] = p

        pin.forwardKinematics(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)

        se3 = self.pin_data.oMi[self.ee_joint_id]
        pos = se3.translation.copy()
        rpy = pin.rpy.matrixToRpy(se3.rotation)
        gripper = np.array(joint_positions[6:], dtype=np.float64)
        return np.concatenate([pos, rpy, gripper])

    # ------------------------------------------------------------------
    # IK  – delta_action (17) -> joint command (17)
    # ------------------------------------------------------------------
    def _apply_delta_action(self, delta_action: np.ndarray) -> list[float]:
        current_joints = [
            self.current_joint_positions[n] for n in self.RIGHT_ARM_JOINTS
        ]
        current_ee = self._joint_positions_to_ee(current_joints)

        # target EE position
        target_pos = current_ee[:3] + delta_action[:3]

        # target EE orientation (wrap rpy)
        target_rpy = current_ee[3:6] + delta_action[3:6]
        for i in range(3):
            target_rpy[i] = np.arctan2(np.sin(target_rpy[i]), np.cos(target_rpy[i]))
        target_rot = pin.rpy.rpyToMatrix(
            float(target_rpy[0]), float(target_rpy[1]), float(target_rpy[2])
        )

        # IK for arm joints (6)
        q_init = pin.neutral(self.ik_solver.model)
        for name, p in zip(self.RIGHT_ARM_JOINTS, current_joints):
            if self.ik_solver.model.existJointName(name):
                j_id = self.ik_solver.model.getJointId(name)
                idx_q = self.ik_solver.model.joints[j_id].idx_q
                if idx_q < self.ik_solver.model.nq:
                    q_init[idx_q] = p

        q_result = self.ik_solver.solve(q_init, target_pos, target_rot)

        arm_joints: list[float] = []
        for name in self.RIGHT_ARM_JOINTS[:6]:
            j_id = self.ik_solver.model.getJointId(name)
            idx_q = self.ik_solver.model.joints[j_id].idx_q
            arm_joints.append(float(q_result[idx_q]))

        # gripper: current + delta (direct, no IK)
        current_gripper = np.array(current_joints[6:])
        target_gripper = current_gripper + delta_action[6:]

        return arm_joints + target_gripper.tolist()

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------
    def _joint_states_cb(self, msg: JointState):
        for i, name in enumerate(msg.name):
            if name in self.current_joint_positions:
                self.current_joint_positions[name] = msg.position[i]

    def _rgb_cb(self, msg: Image):
        try:
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                (msg.height, msg.width, -1)
            )
            enc = msg.encoding.lower()
            if enc == "rgb8":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif enc == "rgba8":
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            self.current_rgb_image = cv2.resize(
                img, (self.vid_W, self.vid_H), cv2.INTER_LINEAR
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Observation  (same delta logic as Data_Recorder.timer_callback)
    # ------------------------------------------------------------------
    def _prepare_observation(self) -> dict[str, torch.Tensor]:
        current_joints = [
            self.current_joint_positions[n] for n in self.RIGHT_ARM_JOINTS
        ]
        current_ee = self._joint_positions_to_ee(current_joints)

        # delta state
        if self.previous_ee_state is None:
            delta_state = np.zeros(self.state_dim, dtype=np.float32)
        else:
            delta_state = (current_ee - self.previous_ee_state).astype(np.float32)
            for i in range(3, 6):
                delta_state[i] = np.arctan2(
                    np.sin(delta_state[i]), np.cos(delta_state[i])
                )
        self.previous_ee_state = current_ee.copy()

        state_tensor = torch.from_numpy(delta_state).unsqueeze(0)  # (1, 17)

        # image: BGR -> RGB, HWC -> CHW, [0, 1]
        image = self.current_rgb_image[:, :, ::-1].copy()
        image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image).unsqueeze(0)  # (1, 3, H, W)

        return {
            "observation.state": state_tensor,
            "observation.images.rgb": image_tensor,
        }

    # ------------------------------------------------------------------
    # Control loop  (10 Hz)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _control_loop(self):
        if not self.is_running:
            return
        try:
            batch = self._prepare_observation()

            # --- DEBUG: raw observation (first 3 steps + every 50th) ---
            is_debug_step = self.inference_count < 3 or self.inference_count % 50 == 0
            if is_debug_step:
                raw_state = batch["observation.state"].squeeze(0).numpy()
                raw_img = batch["observation.images.rgb"].squeeze(0)
                print(f"\n===== STEP {self.inference_count} =====")
                print(f"  [RAW]  state  min={raw_state.min():.5f}  max={raw_state.max():.5f}  "
                      f"xyz=[{raw_state[0]:.5f},{raw_state[1]:.5f},{raw_state[2]:.5f}]  "
                      f"rpy=[{raw_state[3]:.5f},{raw_state[4]:.5f},{raw_state[5]:.5f}]")
                print(f"  [RAW]  image  shape={list(raw_img.shape)}  "
                      f"min={raw_img.min():.3f}  max={raw_img.max():.3f}  "
                      f"mean={raw_img.mean():.3f}")

            # preprocessor: normalization + device
            batch = self.preprocessor(batch)

            if is_debug_step:
                norm_state = batch["observation.state"].squeeze(0).cpu().numpy()
                norm_img = batch["observation.images.rgb"].squeeze(0).cpu()
                print(f"  [NORM] state  min={norm_state.min():.5f}  max={norm_state.max():.5f}  "
                      f"first6={[f'{v:.4f}' for v in norm_state[:6]]}")
                print(f"  [NORM] image  shape={list(norm_img.shape)}  "
                      f"min={norm_img.min():.3f}  max={norm_img.max():.3f}  "
                      f"mean={norm_img.mean():.3f}")

            # select_action handles obs queuing + action chunking internally
            action_tensor = self.policy.select_action(batch)

            if is_debug_step:
                raw_act = action_tensor.cpu().numpy().flatten()
                print(f"  [MODEL OUT] normalized action  min={raw_act.min():.5f}  "
                      f"max={raw_act.max():.5f}  first6={[f'{v:.4f}' for v in raw_act[:6]]}")

            # postprocessor: unnormalize + move to cpu
            action_tensor = self.postprocessor({"action": action_tensor})["action"]

            delta_action = action_tensor.squeeze(0).numpy()

            if is_debug_step:
                print(f"  [UNNORM] delta_action  "
                      f"xyz=[{delta_action[0]:.5f},{delta_action[1]:.5f},{delta_action[2]:.5f}]  "
                      f"rpy=[{delta_action[3]:.5f},{delta_action[4]:.5f},{delta_action[5]:.5f}]  "
                      f"grip_sum={sum(abs(delta_action[6:])):.5f}")

            # delta EE -> joint positions via IK
            joint_cmd = self._apply_delta_action(delta_action)
            self._send_joint_command(joint_cmd)

            if is_debug_step:
                print(f"  [CMD]  joints[:6]={[f'{v:.3f}' for v in joint_cmd[:6]]}")
                print(f"  =====")

            self.inference_count += 1
            if self.inference_count % 10 == 0 and not is_debug_step:
                self.get_logger().info(
                    f"#{self.inference_count}  delta_xyz=[{delta_action[0]:.4f}, "
                    f"{delta_action[1]:.4f}, {delta_action[2]:.4f}]"
                )

        except Exception as e:
            self.get_logger().error(f"Inference error: {e}")
            import traceback
            traceback.print_exc()
            self.stop_control()

    # ------------------------------------------------------------------
    # Joint command publisher
    # ------------------------------------------------------------------
    def _send_joint_command(self, positions: list[float]):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.name = list(self.RIGHT_ARM_JOINTS)
        msg.position = positions
        msg.velocity = [0.0] * len(self.RIGHT_ARM_JOINTS)
        msg.effort = [0.0] * len(self.RIGHT_ARM_JOINTS)
        self.joint_cmd_pub.publish(msg)

    # ------------------------------------------------------------------
    # Start / stop / home
    # ------------------------------------------------------------------
    def go_home(self, duration_sec: float = 3.0):
        self.get_logger().info("Moving to home position...")
        home = [self.HOME_POSITIONS[n] for n in self.RIGHT_ARM_JOINTS]
        steps = int(duration_sec * self.control_rate)
        for _ in range(steps):
            self._send_joint_command(home)
            time.sleep(1.0 / self.control_rate)
        self.get_logger().info("Home position reached")

    def start_control(self):
        self.go_home(duration_sec=3.0)
        time.sleep(0.5)

        # reset policy observation / action queues
        self.policy.reset()
        self.previous_ee_state = None
        self.inference_count = 0

        # warm up FK so first delta_state is zeros
        current_joints = [
            self.current_joint_positions[n] for n in self.RIGHT_ARM_JOINTS
        ]
        self.previous_ee_state = self._joint_positions_to_ee(current_joints).copy()

        self.is_running = True
        self.get_logger().info("Control STARTED")

    def stop_control(self):
        self.is_running = False
        self.get_logger().info("Control STOPPED")

    def toggle_control(self):
        if self.is_running:
            self.stop_control()
        else:
            self.start_control()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    MODEL_PATH = (
        "/home/beable/lerobot/outputs/train/2026-03-11/"
        "06-35-18_diffusion/checkpoints/015000/pretrained_model"
    )

    rclpy.init()
    controller = DiffusionPolicyDeltaController(model_path=MODEL_PATH)

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(controller)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    print("\n" + "=" * 60)
    print("DIFFUSION POLICY DELTA EE CONTROLLER")
    print("=" * 60)
    print("\nCommands:")
    print("  h      - Go to home position")
    print("  ENTER  - Start/stop control (goes home first)")
    print("  q      - Quit")
    print("-" * 60)

    time.sleep(1.0)

    try:
        while rclpy.ok():
            cmd = input("\n> ").strip().lower()
            if cmd == "q":
                print("Exiting...")
                break
            elif cmd == "h":
                controller.go_home()
            else:
                controller.toggle_control()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        controller.stop_control()
        try:
            rclpy.shutdown()
        except Exception:
            pass
        executor_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
