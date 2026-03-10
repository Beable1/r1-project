import os

file_path = "/home/beable/Desktop/r1-project/data_collection_keyboard_delta.py"
with open(file_path, "r") as f:
    lines = f.readlines()

new_lines = []
ik_class_code = """
import setuptools # maybe needed
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
            
        return q
"""

for i, line in enumerate(lines):
    if line.startswith("import select"):
        new_lines.append(line)
        new_lines.append(ik_class_code)
        continue
    
    # In Robot_Keyboard_Controller.__init__
    if "self.current_positions = {name: 0.0 for name in self.joint_names}" in line:
        new_lines.append(line)
        new_lines.append("""
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
""")
        continue
        
    if "self.key_mappings = {" in line and "Keyboard mapping" in lines[i-1]:
        # we will rewrite interactive control directly so we skip mapping changes here.
        new_lines.append(line)
        continue

    if "def update_position(self, joint_name, delta):" in line:
        # inject update_cartesian right before it
        new_lines.append("""
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

""")
        new_lines.append(line)
        continue

    # In timer_callback
    if "action_to_save = copy.copy(action.tolist() if len(action) > 0 else [])" in line:
        new_lines.append(line)
        new_lines.append("""
            # OVERRIDE FOR DELTA: Calculate joint delta between target action and current state
            if len(observation_state) == len(action_to_save):
                action_to_save = [a - o for a, o in zip(action_to_save, observation_state)]
            else:
                action_to_save = [0.0] * len(observation_state)
""")
        continue
        
    # In interactive_control mapping block
    if "                # Control from keyboard mapping (arm controls)" in line:
        new_lines.append("""                # Control from keyboard mapping (arm controls + IK cartesian)
                if key_lower in controller.key_mappings and not controller.use_ik:
                    joint_name, delta = controller.key_mappings[key_lower]
                    controller.update_position(joint_name, delta)
                    continue
                
                if controller.use_ik:
                    # override keyboard
                    if key_lower == 'w': controller.update_cartesian([0.01, 0, 0]); continue
                    if key_lower == 's': controller.update_cartesian([-0.01, 0, 0]); continue
                    if key_lower == 'a': controller.update_cartesian([0, 0.01, 0]); continue
                    if key_lower == 'd': controller.update_cartesian([0, -0.01, 0]); continue
                    if key_lower == 'e': controller.update_cartesian([0, 0, 0.01]); continue
                    if key_lower == 'r': controller.update_cartesian([0, 0, -0.01]); continue
                    
                    if key_lower == 'q': controller.update_cartesian(None, [0, 0, 0.05]); continue
                    if key_lower == 'y': controller.update_cartesian(None, [0, 0, -0.05]); continue
                    if key_lower == 'z': controller.update_cartesian(None, [0.05, 0, 0]); continue
                    if key_lower == 'x': controller.update_cartesian(None, [-0.05, 0, 0]); continue
                    
                    if key_lower == 'c': controller.update_position("thumb_joint_roll_r", +controller.step_size * 3); continue
                    if key_lower == 'v': controller.update_position("thumb_joint_roll_r", -controller.step_size * 3); continue
""")
        continue
    if "if key_lower in controller.key_mappings:" in line:
        continue # skip
    if "joint_name, delta = controller.key_mappings[key_lower]" in line:
        continue # skip
    if "controller.update_position(joint_name, delta)" in line:
        continue # skip
        
    new_lines.append(line)

with open(file_path, "w") as f:
    f.writelines(new_lines)
print("File patched.")
