import pinocchio as pin
import numpy as np

urdf_path = "/home/beable/IsaacLab/IsaacLab/r1-new/urdf/r1-new.urdf"
model = pin.buildModelFromUrdf(urdf_path)
data = model.createData()

JOINT_ID = model.getJointId("wrist_pitch_joint_r")
print("Joint ID:", JOINT_ID)

q = pin.neutral(model)
pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)

print("EE pos:", data.oMi[JOINT_ID].translation)
