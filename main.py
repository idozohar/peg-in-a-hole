import os
import numpy as np
import mujoco_py as mj
from mujoco_py import (MjSim, load_model_from_path)
import motion_plan
import impedance
import filters
import contacts
import print_results
import sensor_reading as sr
import UR5_kinematics as UR5_kin
from scipy.spatial.transform import Rotation as R

# ---------------  Run the simulation  ---------------- #
control_law = 'Impedance'  # Choose Control law: 'PID' , 'Impedance'
hybrid = True  # if true and chosen 'Impedance', Movement to cylinder with PID and afterwards impedance
need_render = True  # To print results choose False

# ---------------  Set trajectory (homogeneous transformation matrix)  ---------------- #
start = np.array([[1, 0, 0, -0.7],
                  [0, 0, -1, -0.21],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1]])

mid = np.array([[1, 0, 0, -0.26],
                [0, 0, -1, -0.5],
                [0, 1, 0, 0.14],
                [0, 0, 0, 1]])

target = np.array([[1, 0, 0, -0.26],
                   [0, 0, -1, -0.65],
                   [0, 1, 0, 0.14],
                   [0, 0, 0, 1]])

up_from_target = np.array([[1, 0, 0, -0.26],
                           [0, 0, -1, -0.65],
                           [0, 1, 0, 0.3],
                           [0, 0, 0, 1]])

left = np.array([[1, 0, 0, 0.02],
                 [0, 0, -1, -0.65],
                 [0, 1, 0, 0.3],
                 [0, 0, 0, 1]])
# path 1
down = np.array([[1, 0, 0, 0.02],
                 [0, 0, -1, -0.6575],
                 [0, 1, 0, 0.27],
                 [0, 0, 0, 1]])
x_down = np.array([0.02, -0.6575, 0.27])

# ---------------  Add error to final position  ---------------- #
dx = -0.01
dy = -0.01
angle_err = np.deg2rad([0, 0, 0])
rotation_mat = np.identity(3)
temp_mat = R.from_rotvec(angle_err)
rotation_mat = temp_mat.as_matrix()
trans_mat = np.identity(4)
trans_mat[0:3, 0:3] = rotation_mat
down2 = np.array([[1, 0, 0, 0.02 + dx],
                  [0, 0, -1, -0.6575 + dy],
                  [0, 1, 0, 0.20],
                  [0, 0, 0, 1]])
down2 = down2 @ trans_mat
# ----------------------------------------------------------------- #
# --- End of points --- #

# add all points by order to path
path2cylinder_points = np.array([start, mid, target])  # straight path
path2target_points = np.array([target, up_from_target, left, down, down2])  # straight path with grip

# build simulation from xml file
model = load_model_from_path("./UR5/UR5gripper_box.xml")
sim = MjSim(model)

# Define start Position
state = sim.get_state()
state.qpos[7:13] = np.array([0.02644, -0.51929, 1.07047, -0.55118, 0.02646, 0])
sim.set_state(state)



# ---------------  Set Simulation Parameters  ---------------- #
move_speed = 0.5  # [m/s] Set the desired Speed
sim_time = 12  # [sec] Length of Simulation
is_gravity_on = True
controller = 'PID'  # initial Controller: 'PID' , 'impedance' , 'bias'
if not hybrid and control_law == 'Impedance':
    controller = 'impedance'
elif control_law == 'PID':
    hybrid = False


# Control
# PID
kp = np.diag(np.array([100, 100, 100, 50, 50, 0.1]))
ki = np.diag(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.001])*0)  # 0.5
kd = 0.3*0  # 0.3

# impedance
kp_im = np.diag(np.array([100, 100, 100, 50, 50, 0.1])*1)
ki_im = np.diag(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.001])*0)  # 0.5
kd_im = 0.3*0  # 0.3

K = np.array([[1000, 0, 0, 0, 0, 0],
              [0, 1000, 0, 0, 0, 0],
              [0, 0, 2000, 0, 0, 0],
              [0, 0, 0, 2500, 0, 0],
              [0, 0, 0, 0, 500, 0],
              [0, 0, 500, 0, 0, 500]])

B = np.identity(6) * 400  # 200

M = np.identity(6) * 0.2   # 0.2


# Camera Settings
cam_position = sim.data.get_camera_xpos('main1')
cam_position2 = model.stat.center

# Video Settings
if need_render:
    viewer = mj.MjViewer(sim)
    viewer._record_video = False
    viewer._video_idx = 0
    viewer._video_path = "/home/boxmaster/Project/local/videos/video_1.mp4"

    # view Settings
    viewer._paused = True
    viewer._hide_overlay = True
    viewer._run_speed = 1
    # ---------------  End of Simulation Parameters  --------------- #
    viewer.render()

# set Gravity value
model.opt.gravity[-1] = is_gravity_on * -9.81
# get time for each step of sim
DT = model.opt.timestep

# ---- build paths in Cartesian space ---- #
path2cylinder, p_dot_2cylinder, p_2dot_2cylinder = motion_plan.build_cartesian_path(path2cylinder_points, move_speed, DT)
path2target, p_dot_2target, p_2dot_2target = motion_plan.build_cartesian_path(path2target_points, move_speed, DT)

# apply inverse kinematics to get path in joint space
q_path2cylinder = motion_plan.build_trajectory(path2cylinder)
q_path2target = motion_plan.build_trajectory(path2target)

# get q velocity by:  q_dot = Jacobian.T * cartesian_velocity
q_dot_2cylinder = motion_plan.q_velocity(sim, p_dot_2cylinder, q_path2cylinder)
q_dot_2target = motion_plan.q_velocity(sim, p_dot_2target, q_path2target)

# initialize joint position and velocity vectors
q_r = np.zeros((6,))
q_r_dot = np.zeros((6,))

# init flags and counters
k = 0
sim_loop_num = 0
delay_time = 1 / DT  # [sec] delay time for Grip 
delay = delay_time
delay_flag = 0
move_flag = 1
advance_q_path = 1
cylinder_flag = 1
i_control = np.zeros((6,))
d_control = 0
pos_error = 0
last_forces_avg = [0, 0, 0]
Griper_on = True
flag_norm = False

# init log vars
actual_q = np.zeros((6, 1))
actual_q_dot = np.zeros((6, 1))
pos_error_log = np.zeros((6, 1))
f_in_log = np.zeros(1)
u_log = np.zeros((6, 1))
force_log = np.zeros((6, 1))
force = np.zeros((3, 1))
x_r = np.zeros((3, 1))
x_0 = np.zeros((3, 1))
x_m = np.ones((6, 1)) * 0
empty_vec = np.ones((6, 1)) * 0
gripper_norm = 0.0

joints = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

# init PID path (in joint space)
q_path = q_path2cylinder
q_dot = q_dot_2cylinder

# init Impedance Path (in Cartesian space)
p = path2cylinder
p_dot = p_dot_2cylinder
p_2dot = p_2dot_2cylinder

while True:
    # Read current joint angels and velocity
    q_r, q_r_dot = motion_plan.get_joints(sim, q_r, q_r_dot)
    x_r_mat = UR5_kin.forward(q_r)
    x_r_temp = x_r_mat[0:3, 3]
    x_r = np.append(x_r, np.reshape(x_r_temp, (3, 1)), axis=1)
    # Controller choice
    if controller == 'PID':
        bias = is_gravity_on * sim.data.qfrc_bias[6:12] / 101  # 101 is Gear ratio
        pos_error = q_path[:, k] - q_r[:]
        p_control = kp @ pos_error                 
        i_control = i_control + ki @ pos_error
        d_control = kd * (q_dot[:, k] - q_r_dot[:])
        sim.data.ctrl[0:6] = p_control + i_control + d_control + bias
        x_m = np.append(x_m, empty_vec, axis=1)

    elif controller == 'impedance':
        q_d, q_d_dot, x_im = impedance.imic(sim, K, B, M, q_r, q_r_dot, p[k], p_dot[:, k], DT, Tau)
        bias = is_gravity_on * sim.data.qfrc_bias[6:12] / 101  # 101 is Gear ratio
        pos_error = q_d - q_r[:]
        p_control = kp_im @ pos_error
        d_control = kd_im * (q_d_dot - q_r_dot[:])
        sim.data.ctrl[0:6] = p_control + d_control + bias
        x_m = np.append(x_m, np.reshape(x_im, (6, 1)), axis=1)

    elif controller == 'bias':
        pos_error = q_path[:, k] - q_r[:]
        bias = is_gravity_on * sim.data.qfrc_bias[6:12] / 101  # 101 is Gear ratio
        sim.data.ctrl[0:6] = bias

    else:
        print('no control')

	# log Position Error
    pos_error_log = np.append(pos_error_log, np.reshape(pos_error, (6, 1)), axis=1)
	# log Control Effort
    u_log = np.append(u_log, np.reshape(sim.data.ctrl[0:6], (6, 1)), axis=1)

	# Execute Grip and Change Path
    if k == len(q_path[1, :]) - 1 and advance_q_path == 1:
        delay_flag = 1
        move_flag = 0
        cylinder_flag = 0
        if Griper_on:
            # Close Gripper
            grip_torque = 0.4
            sim.data.ctrl[6] = grip_torque
            sim.data.ctrl[7] = grip_torque
            sim.data.ctrl[8] = grip_torque
            Griper_on = False
        # change path
        p = path2target
        p_dot = p_dot_2target
        p_2dot = p_2dot_2target
        if controller == 'PID':
            q_path = q_path2target
            q_dot = q_dot_2target
        k = 0
        delay = delay_time  # delay in sec
        advance_q_path = 0
        # log data
        actual_q = np.append(actual_q, np.reshape(q_r, (6, 1)), axis=1)
        actual_q_dot = np.append(actual_q_dot, np.reshape(q_r_dot, (6, 1)), axis=1)

    x_0_pos_temp = p[k, 0:3, 3]
    x_0 = np.append(x_0, np.reshape(x_0_pos_temp, (3, 1)), axis=1)

    if k < len(q_path[1, :]) - 1 and move_flag == 1 and controller == 'PID':
        k += 1
        # log data
        if sim_loop_num == 0:
            actual_q = np.reshape(q_r, (6, 1))
            actual_q_dot = np.reshape(q_r_dot, (6, 1))
        else:
            actual_q = np.append(actual_q, np.reshape(q_r, (6, 1)), axis=1)
            actual_q_dot = np.append(actual_q_dot, np.reshape(q_r_dot, (6, 1)), axis=1)

    elif k < len(p_dot[1, :]) - 1 and move_flag == 1 and controller == 'impedance':
        k += 1
        # log data
        if sim_loop_num == 0:
            actual_q = np.reshape(q_r, (6, 1))
            actual_q_dot = np.reshape(q_r_dot, (6, 1))
        else:
            actual_q = np.append(actual_q, np.reshape(q_r, (6, 1)), axis=1)
            actual_q_dot = np.append(actual_q_dot, np.reshape(q_r_dot, (6, 1)), axis=1)

    sim.step()

    # Get Forces on the Cylinder
    Tau_cont = contacts.find_contacts(sim, 1, 'cylinder1')  # from contacts
    # Get Forces from sensor
    gripper_norm = sr.get_reading(sim, "gripperpalm_norm")
    force = sr.get_reading(sim, "wrist_3_link_f")
    torque = sr.get_reading(sim, "wrist_3_link_t")
    # normalize the sensor read when reaching over the hole (down pos)
    epsilon = 0.0002
    if (np.linalg.norm(x_r_temp - x_down) < epsilon) and not flag_norm:
        force_norm = force.copy()
        torque_norm = torque.copy()
        flag_norm = True

    if flag_norm:
        act_force = force.copy()
        act_torque = torque.copy()
        force = act_force - force_norm
        torque = act_torque - torque_norm

        if hybrid and controller == 'PID':  # Start Impedance Controller when sensing obstacle
            if abs(force[2]) > 15:
                controller = 'impedance'

    Tau = np.append(force, torque)
	# filter and log Forces
    force_log = np.append(force_log, np.reshape(Tau, (6, 1)), axis=1)
    force_log = filters.butter_lowpass_filter(force_log, 5, 20, 2)

    if need_render:
        viewer.render()

    if delay_flag == 1:
        delay -= 1
        if delay == 0:
            delay_flag = 0
            move_flag = 1

	# advance the loop counter
    sim_loop_num += 1
    # print(sim_loop_num * DT)
	# Print Graphs and the End of Simulation
    if sim_loop_num+1 > (sim_time/DT):
        if not need_render:
            # viewer._paused = True
            # print desired vs actual (q, q_dot)
            # print(sim_loop_num * DT)
            print_results.print_q_actual(q_path2cylinder, q_path2target, actual_q, DT, 'angle')
            print_results.print_q_actual(q_dot_2cylinder, q_dot_2target, actual_q_dot, DT, 'speed')
            # print Joint Position Error
            print_results.print_pos_error(pos_error_log, DT)
            # print Control effort
            print_results.print_torque(u_log, DT)
            # force print
            print_results.print_force_scope2(force_log, DT)
            print_results.print_xyz_3D(x_r)
            time = np.linspace(0, sim_time, len(x_0[0, :]))
            print_results.print_xyz_time(time[1:], x_0[:, 1:], x_m[:, 1:], x_r[:, 1:])

        break

    if os.getenv('TESTING') is not None:
        break