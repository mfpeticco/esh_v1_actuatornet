import mujoco
import mujoco.viewer
import pickle
import datetime
import os
import numpy as np
import scipy.interpolate as interp

model = mujoco.MjModel.from_xml_path(f"assets/esh/right_hand_mjcf_torque.xml") # load mujoco model
data = mujoco.MjData(model) # create empty data object
                
viewer = mujoco.viewer.launch_passive(model, data) # launch viewer
viewer._hide_overlay = True
viewer._render_every_frame = True


def generate_smooth_positions(joint_ranges, num_timesteps, control_point_count):
    """
    Generate a smooth trajectory of joint positions with a fixed time step, 
    ensuring consistent speed regardless of the number of time steps.
    
    Returns:
    - joint_positions: numpy array of shape (num_timesteps, n_joints) containing joint positions.
    """
    n_joints = joint_ranges.shape[0]
    total_duration = num_timesteps

    control_point_count = max(5, control_point_count) # Generate a proportional number of control points 
    control_times = np.linspace(0, total_duration, control_point_count) # Control times should span the entire duration
    time = np.linspace(0, total_duration, num_timesteps)

    joint_positions = np.zeros((num_timesteps, n_joints))
    for i in range(n_joints):
        # Generate random control points within the joint range for smooth interpolation
        random_points = np.random.uniform(joint_ranges[i, 0], joint_ranges[i, 1], size=control_point_count)
        random_points[0] = 0  # Start with all joints at position 0

        # Use cubic interpolation to generate a smooth trajectory
        cubic_spline = interp.CubicSpline(control_times, random_points, bc_type='clamped')

        # Evaluate the spline at each time step for joint positions
        joint_positions[:, i] = cubic_spline(time)

        # Clamp the joint positions to stay within joint limits
        joint_positions[:, i] = np.clip(joint_positions[:, i], joint_ranges[i, 0], joint_ranges[i, 1])

    return joint_positions


def apply_command(kp, kd, q_des, dq_des, tau_ff, steps_window=10):
    '''
    Applies a PD control command to the robot and steps the simulation forward by steps_window time steps.

    Returns:
    - q_0: numpy array of shape (n_joints,) containing the initial positions before applying the command
    - dq_0: numpy array of shape (n_joints,) containing the initial velocities before applying the command
    - cmd_torques: numpy array of shape (n_joints,) containing the torques applied to the joints
    - q_f: numpy array of shape (n_joints,) containing the final positions after applying the command
    - torque_f: numpy array of shape (n_joints,) containing the torques applied to the joints after stepping the simulation forward
    '''
    # Read current position and velocity from mujoco data object
    q_0 = np.array(data.qpos)
    dq_0 = np.array(data.qvel)
    # Step the simulation forward by steps_window time steps
    for _ in range(steps_window):
        # # Floating base robots
        # joint_pos = data.qpos[7:] # Skip the position and rotation of the base
        # joint_vel = data.qvel[6:] # Skip the linear and angular velocity of the base

        # Fixed base robots
        joint_pos = data.qpos
        joint_vel = data.qvel

        # Apply PD control
        cmd_torques = kp * (q_des - joint_pos) + kd * (dq_des - joint_vel) + tau_ff
         # set the control signals for the "motor" actuators specified in the model file, which take in torque values for control
        data.ctrl = np.array(cmd_torques)
        mujoco.mj_step(model, data)

    if viewer is not None:
        viewer.sync()

    # record relevant quantities after stepping the simulation
    
    q_f = np.array(data.qpos)
    dq_f = np.array(data.qvel)
    torque_cmd = cmd_torques # final commanded torques
    
    return (q_0, dq_0, torque_cmd, q_f, dq_f)


num_dof = 9
random_scale = 1.0
joint_ranges = np.array([
    [0, 1.48353],    # base_l00
    [0, 1.0472],     # l00_l01
    [0, 0.785398],   # l01_l02
    [-0.349066, 0.349066],   # base_l10
    [0, 1.5708],     # l10_l11
    [0, 1.8326],     # l11_l12
    [-0.349066, 0.349066],   # base_l20
    [0, 1.5708],     # l20_l21
    [0, 1.8326]      # l21_l22
])

kp = [10] * num_dof
kd = [4] * num_dof
dq_des = [0] * num_dof # desired velocity
tau_ff = [0] * num_dof # feed-forward torque

num_steps = 20000
example_trajectory = generate_smooth_positions(joint_ranges, num_steps, 20)
joint_data = []
for t in range(num_steps):
    # q_des = np.random.uniform(joint_ranges[:, 0], joint_ranges[:, 1])
    q_des = example_trajectory[t,:]
    q_0, dq_0, torque_cmd, q_f, dq_f = apply_command(kp, kd, q_des, dq_des, tau_ff)
    cur_log_data = {
    "joint_pos": q_0,
    "joint_vel": dq_0,
    "joint_torque_cmd": torque_cmd,
    "joint_pos_next": q_f,
    "joint_vel_next": dq_f
    }
    joint_data.append(cur_log_data)
joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
log_data = (joint_names, joint_data)


# Create the directory if it doesn't exist
output_dir = "logs/esh_v2"
os.makedirs(output_dir, exist_ok=True)

# Get today's date as a string (formatted as YYYY-MM-DD)
today = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

# # Save the dictionary to a .pkl file in the specified directory
# filename = os.path.join(output_dir, f"joint_values_{today}.pkl")
# with open(filename, 'wb') as f:
#     pickle.dump(log_data, f)

# print(f"Data saved to {filename}")

filename = os.path.join(output_dir, "logtrain.pkl")
with open(filename, 'wb') as f:
    pickle.dump(log_data, f)

print(f"Data saved to {filename}")