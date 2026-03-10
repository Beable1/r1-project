import pinocchio as pin
import numpy as np
import time

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
                
    def solve(self, q_current_full, target_pos, target_rot=None, max_iter=1000, eps=1e-4):
        q = np.copy(q_current_full)
        target_se3 = pin.SE3(target_rot if target_rot is not None else np.eye(3), target_pos)
        damp = 1e-4
        dt = 0.5
        
        for i in range(max_iter):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            
            # Error in LOCAL frame of the EE
            iMd = self.data.oMi[self.ee_id].actInv(target_se3)
            err = pin.log(iMd).vector
            
            if np.linalg.norm(err) < eps:
                break
            
            J = pin.computeJointJacobian(self.model, self.data, q, self.ee_id)
            # J is in LOCAL frame
            J_masked = np.zeros_like(J)
            for vj in self.v_idx:
                J_masked[:, vj] = J[:, vj]
                
            v = J_masked.T @ np.linalg.solve(J_masked @ J_masked.T + damp * np.eye(6), err)
            q = pin.integrate(self.model, q, v * dt)
            
        return q

urdf_path = "/home/beable/IsaacLab/IsaacLab/r1-new/urdf/r1-new.urdf"
solver = IKSolver(urdf_path, "wrist_pitch_joint_r", [
    "right_shoulder_link_joint",
    "right_arm_top_link_joint",
    "right_arm_bottom_link_joint",
    "right_forearm_link_joint",
    "wrist_roll_joint_r",
    "wrist_pitch_joint_r"
])

q0 = pin.neutral(solver.model)
# try to see if neutral pos works or if we should use the actual home pos
pin.forwardKinematics(solver.model, solver.data, q0)
ee_initial_pos = solver.data.oMi[solver.ee_id].translation.copy()
ee_initial_rot = solver.data.oMi[solver.ee_id].rotation.copy()

print("Initial EE:", ee_initial_pos)
target_pos = ee_initial_pos + np.array([0.1, 0, 0])

t1 = time.time()
q_res = solver.solve(q0, target_pos, ee_initial_rot)
t2 = time.time()

print("Solved in", t2-t1, "s")
pin.forwardKinematics(solver.model, solver.data, q_res)
pin.updateFramePlacements(solver.model, solver.data)
print("Final EE:", solver.data.oMi[solver.ee_id].translation)
print("Expected:", target_pos)
