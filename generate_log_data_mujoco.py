import mujoco
import mujoco.viewer
import numpy as np

# model = mujoco.MjModel.from_xml_path(f"assets/go1/xml/go1.xml")
model = mujoco.MjModel.from_xml_path(f"assets/esh/right_hand_mjcf_torque.xml")
data = mujoco.MjData(model)
                
viewer = mujoco.viewer.launch_passive(model, data)
viewer._hide_overlay = True
viewer._render_every_frame = True

def apply_command(kp, kd, q_des, dq_des, tau_ff):
    
    for _ in range(10):
        # # Floating base robots
        # joint_pos = data.qpos[7:] # Skip the position and rotation of the base
        # joint_vel = data.qvel[6:] # Skip the linear and angular velocity of the base

        # Fixed base robots
        joint_pos = data.qpos
        joint_vel = data.qvel

        # Apply PD control
        torques = kp * (q_des - joint_pos) + kd * (dq_des - joint_vel) + tau_ff
        data.ctrl = torques

        mujoco.mj_step(model, data)

    if viewer is not None:
        viewer.sync()

num_dof = 9
random_scale = 0.0

kp = [300] * num_dof
kd = [4] * num_dof
dq_des = [0] * num_dof
tau_ff = [0] * num_dof

log_data = {
    "joint_pos": [],
    "joint_pos_targets": [],
    "joint_vel": [],
    "torques": [],
}

while True:
    q_des = np.random.uniform(-random_scale, random_scale, num_dof) #- 0.5
    apply_command(kp, kd, q_des, dq_des, tau_ff)
    
