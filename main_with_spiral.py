import os
import mujoco_py as mj
import numpy as np
from mujoco_py import (MjSim, load_model_from_path)
import our_func
import impedance
import dimanic_mat as dm
import my_filter
import contacts
import UR5_kinematics as kin
import spiral_algorithm as cy
import UR5_kinematics as UR_kin

# Set points here (homogeneous transformation matrix)
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

left = np.array([[1, 0, 0, -0.15],
                 [0, 0, -1, -0.6575],
                 [0, 1, 0, 0.3],
                 [0, 0, 0, 1]])

down = np.array([[1, 0, 0, -0.15],
                 [0, 0, -1, -0.6575],
                 [0, 1, 0, 0.27],
                 [0, 0, 0, 1]])

down2 = np.array([[1, 0, 0, -0.15],
                  [0, 0, -1, -0.6575],
                  [0, 1, 0, 0.20],
                  [0, 0, 0, 1]])

test = np.array([[1, 0, 0, -0.11],
                 [0, 0, -1, -0.65],
                 [0, 1, 0, 0.3],
                 [0, 0, 0, 1]])

target_test = np.array([[1, 0, 0, -0.26],
                        [0, 0, -1, -0.5],
                        [0, 1, 0, 0.3],
                        [0, 0, 0, 1]])

crash_test = np.array([[1, 0, 0, -0.26],
                       [0, 0, -1, -0.65],
                       [0, 1, 0, 0.1],
                       [0, 0, 0, 1]])

downbox = np.array([[1, 0, 0, -0.15],
                    [0, 0, -1, -0.69],
                    [0, 1, 0, 0.27],
                    [0, 0, 0, 1]])

down2box = np.array([[1, 0, 0, -0.14],
                     [0, 0, -1, -0.69],
                     [0, 1, 0, 0.2],
                     [0, 0, 0, 1]])
# --- End of points --- #

# add all points by order to path
path2cylinder_points = np.array([start, mid, target])  # straight path
path2target_points = np.array([target, up_from_target, left, down, down2])  # straight path with grip

# test for calibration, removes contact with cylinder and box
# path2cylinder_points = np.array([start, mid, target_test])
# path2target_points = np.array([target_test, up_from_target, left])

# # test for sensor read, crash to table
# path2cylinder_points = np.array([start, mid, target])
# path2target_points = np.array([target, up_from_target, crash_test])

# test for sensor read, crash to table
# path2cylinder_points = np.array([start, mid, target])
# path2target_points = np.array([target, up_from_target])

# test for contacts, crash to box
path2cylinder_points = np.array([start, mid, target])  # straight path
path2target_points = np.array([target, up_from_target, left, downbox, down2box])

# build simulation from xml file
model = load_model_from_path("./UR5_our/UR5gripper_box.xml")
sim = MjSim(model)

# Define start Position
state = sim.get_state()
state.qpos[7:13] = np.array([0.02644, -0.51929, 1.07047, -0.55118, 0.02646, 0])#7:13     14:20
# state.xpos[7:13] = np.array([0.02644, -0.51929, 1.07047, -0.55118, 0.02646, 0])
# state.qpos[14:20] = np.array([0.02644, -0.51929, 1.07047, -0.55118, 0.02646, 0])
# state.qpos[14:20] = np.array([0.01, -0.51929, 1.07047, -0.55118, 0.5, 0])  # for test
# state.qpos[14:20] = [0, 0, 0, 0, 0, 0]
sim.set_state(state)
viewer = mj.MjViewer(sim)


# ---------------  Set Simulation Parameters  ---------------- #
controller = 'PID'  # 'PID' , 'impedance' , 'bias'
hybrid = True  # if true first part in PID and second part in impedance
move_speed = 0.4  # [m/s] ??  0.5
sim_time = 10  # [sec]

is_gravity_on = True
print_q = False
print_q_dot = False
need_render = True
print_status = True  # Prints Graphs at the end of run
Griper_on = True

# Control
# PID
kp = np.diag(np.array([100, 100, 100, 50, 50, 0.1]))
ki = np.diag(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.001])*0)  # 0.5
kd = 0.3*0  # 0.3

# impedance
kp_im = np.diag(np.array([100, 100, 100, 50, 50, 0.1])*1)
ki_im = np.diag(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.001])*0)  # 0.5
kd_im = 0.3*0  # 0.3

# kp_im = 80  # 1.5
# ki_im = 0.1  # 0.1
# kd_im = 0.3*0  # 0.1

K = np.identity(3) * 4000  # 4000
B = np.identity(3) * 0  # 200
M = np.identity(3) * 0.2   # 0.2

# Camera Settings
cam_position = sim.data.get_camera_xpos('main1')
cam_position2 = model.stat.center

# Video Settings
viewer._record_video = False
viewer._video_idx = 0
viewer._video_path = "/home/Documents/Project/pictur/video_1.mp4"

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
path2cylinder, p_dot_2cylinder, p_2dot_2cylinder = our_func.build_cartesian_path(path2cylinder_points, move_speed, DT)
path2target, p_dot_2target, p_2dot_2target = our_func.build_cartesian_path(path2target_points, move_speed, DT)

# apply inverse kinematics to get path in joint space
q_path2cylinder = our_func.build_trajectory(path2cylinder)
q_path2target = our_func.build_trajectory(path2target)

# print q
if print_q:
    our_func.print_q(q_path2cylinder, q_path2target, DT, 'angle')

# get q velocity by:  q_dot = Jacobian.T * cartesian_velocity
q_dot_2cylinder = our_func.q_velocity(sim, p_dot_2cylinder, q_path2cylinder)
q_dot_2target = our_func.q_velocity(sim, p_dot_2target, q_path2target)

# print q_dot
if print_q_dot:
    our_func.print_q(q_dot_2cylinder, q_dot_2target, DT, 'speed')

# initialize joint position and velocity vectors
q_r = np.zeros((6,))
q_r_dot = np.zeros((6,))

# init flags and counters
k = 0
sim_loop_num = 0
delay_time = 1 / DT  # delay in sec
delay = delay_time
delay_flag = 0
move_flag = 1
advance_q_path = 1
i_control = np.zeros((6,))
d_control = 0
pos_error = 0
last_forces_avg = [0, 0, 0]

# init log vars
actual_q = np.zeros((6, 1))
actual_q_dot = np.zeros((6, 1))
pos_error_log = np.zeros((6, 1))
f_in_log = np.zeros(1)
u_log = np.zeros((6, 1))
force_scope_log = np.zeros((6, 1))
force_log = np.zeros((3, 1))
x_r = np.zeros((3, 1))
x_0 = np.zeros((3, 1))
x_m = np.zeros((3, 1))
x_im = np.zeros((3, 1))
# Sircol= np.zeros((4, 4))
flag_m = 1
pleg = 0
T = 0
h = 0
zpos_old = 0
joints = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

# init PID path
q_path = q_path2cylinder
q_dot = q_dot_2cylinder

# init Impedance Path
p = path2cylinder
p_dot = p_dot_2cylinder
p_2dot = p_2dot_2cylinder

x_r = np.reshape(start[0:3,3],(3, 1))
x_0 = x_r
x_cylinder = x_r
while True:
    # Read current joint angels and velocity
    q_r, q_r_dot = our_func.get_joints(sim, q_r, q_r_dot)
    xpos = sim.data.get_body_xpos('cylinder_play')
    T += DT
    # Controller choice
    if controller == 'PID':
        #### ------------------------- spiral  -----------------------------###
        if (force_log[2, -1] > 0) & (pleg != 2):
            pleg = 1
        if pleg == 1:
            Sircol[0:3, 3] = cy.next_point(P0, theta)
            Sircol[2, 3] = P0[2, 3]-0.00005
            theta = theta + 0.01
            q_d = kin.inv_kin(Sircol,Sircol)
            S = np.append(S, np.reshape(Sircol[0:3, 3], (3, 1)), axis=1)
            pleg = 1

            if (force_log[2, -1] == 0):
                l = 1
            if (-0.22 >= xpos[0]) & (-0.23 <= xpos[0]) & (1.1 <= xpos[1]) & (1.11 >= xpos[1]):  # (xpos[0]+0.29969405)**2+(xpos[1]-1.0070193)**2 < 4e-6:
                ext_forces = sim.data.cfrc_ext[sim.model.body_name2id('OUR_TABLE'), 3:]
                Pinnel = Sircol
                Pinnel[1, 3] -= 0.02
                pleg = 2
        elif pleg == 2:
            if Pinnel[2, 3] > 0.18:
                Pinnel[2, 3] -= 0.01
            q_d = kin.inv_kin(Pinnel,Pinnel)
        else:
            q_d = q_path[:, k]
            P0 = kin.forward(q_d)
            S = np.reshape(P0[0:3, 3], (3, 1))
            Sircol = P0
            theta = 0

        f_in = sim.data.cfrc_ext[sim.model.body_name2id('cylinder_play'), 5]
        print('f_in: ', f_in)
        #### ------------------------- spiral  -----------------------------###

        bias = is_gravity_on * sim.data.qfrc_bias[6:12] / 101  # 101 is Gear ratio
        pos_error = q_d - q_r[:]
        p_control = kp @ pos_error
        i_control = i_control + ki @ pos_error
        d_control = kd * (q_dot[:, k] - q_r_dot[:])
        sim.data.ctrl[0:6] = p_control + i_control + d_control + bias

        # f_in = sim.data.get_sensor('gripperpalm_frc')
        # f_in = sim.data.get_sensor('end')

    elif controller == 'impedance':

        #### ------------------------- spiral  -----------------------------###
        if (xpos[2] > 0.96) :
            h = 1
        if (force_log[2, -1] > 0)  & (pleg != 2): #| ((h == 1) & (xpos[2] <= 0.9598)))
            pleg = 1
        if pleg==1:
            Sircol[0:3, 3] = cy.next_point(P0, theta)
            Sircol[2, 3] = P0[2, 3]
            theta = theta + 0.01
            q_d, q_d_dot, x_im = impedance.imic(sim,K,B,M ,q_r, q_r_dot, Sircol, 0, DT, force_log[2, -1])#, force_scope_log)  # force_log[2, -1]
            S = np.append(S, np.reshape(Sircol[0:3, 3], (3, 1)), axis=1)
            pleg = 1
            ext_forces = sim.data.cfrc_ext[sim.model.body_name2id('cylinder_play'), 3:]
            xpos = sim.data.get_body_xpos('cylinder_play')
            f_in = sim.data.get_sensor('csens')
            ext_forces_t = sim.data.cfrc_ext[sim.model.body_name2id('OUR_TABLE'), 3:]
            if (force_log[2, -1]==0):
                l = 1
            if (-0.15>=xpos[0]) & (-0.16<=xpos[0]) & (1.17 <=xpos[1]) & (1.18>=xpos[1]):#(-0.30413>=xpos[0]) & (-0.31<=xpos[0]) & (1.0073 <=xpos[1]) & (1.008>=xpos[1])
                # ext_forces = sim.data.cfrc_ext[sim.model.body_name2id('OUR_TABLE'), 3:]
                Pinnel = Sircol
                # Pinnel[3, 3] += 0.01
                # q_d, q_d_dot, x_im = impedance.imic(sim, q_r, q_r_dot, Pinnel, 0, DT, force_log[2, -1], force_scope_log)
                pleg = 2
        elif pleg == 2:
            if Pinnel[2, 3] > 0.19:
                Pinnel[2, 3] -= 0.01
            q_d, q_d_dot, x_im = impedance.imic(sim,K,B,M ,q_r, q_r_dot, Pinnel, 0, DT, force_log[2, -1]) #, force_scope_log)
        else:
            q_d, q_d_dot, x_im = impedance.imic(sim,K,B,M ,q_r, q_r_dot, p[k], p_dot[:, k], DT, force_log[2, -1])#, force_scope_log)  # force_log[2, -1]
            P0 = p[k]
            S = np.reshape(P0[0:3, 3], (3, 1))
            Sircol = P0
            theta = 0

        f_in = sim.data.cfrc_ext[sim.model.body_name2id('cylinder_play'), 5]
        print('f_in: ', f_in)
        #### ------------------------- spiral  -----------------------------###
        # q_d, q_d_dot, x_im = impedance.imic(sim, q_r, q_r_dot, p[k], p_dot[:, k], DT, force_log[2, -1], force_scope_log)  # force_log[2, -1]
        if flag_m == 1:
            x_m = np.reshape(x_im, (3, 1))
            flag_m = 0
        bias = is_gravity_on * sim.data.qfrc_bias[6:12]  / 101  # 101 is Gear ratio
        pos_error = q_d - q_r[:]
        p_control = kp_im @ pos_error
        d_control = kd_im * (q_d_dot - q_r_dot[:])
        sim.data.ctrl[0:6] = p_control + d_control + bias
        x_m = np.append(x_m, np.reshape(x_im, (3, 1)) , axis=1)
        zpos_old = xpos[2]

    elif controller == 'bias':
        q_d = q_path[:, k]
        pos_error = q_d - q_r[:]
        bias = is_gravity_on * sim.data.qfrc_bias[6:12] / 101  # 101 is Gear ratio [6:12]  [12:18]
        sim.data.ctrl[0:6] = bias
        ext_forces = sim.data.cfrc_ext[sim.model.body_name2id('cylinder_play'), 3:]

    else:
        print('no control')


    mat_x0 = kin.forward(q_d)
    mat_xr = kin.forward(q_r)
    x_0 = np.append(x_0,  np.reshape(mat_x0[0:3, 3], (3, 1)), axis=1)
    x_r = np.append(x_r, np.reshape(mat_xr[0:3, 3], (3, 1)), axis=1)
    x_cylinder = np.append(x_cylinder, np.reshape(xpos, (3, 1)), axis=1)
    pos_error_log = np.append(pos_error_log, np.reshape(pos_error, (6, 1)), axis=1)
    u_log = np.append(u_log, np.reshape(sim.data.ctrl[0:6], (6, 1)), axis=1)
    # f_in_log = np.append(f_in_log, f_in)
    # force_scope_log = our_func.force_scope(sim, force_scope_log)

    if k == len(q_path[1, :]) - 1 and advance_q_path == 1:
        delay_flag = 1
        move_flag = 0
        if Griper_on:
            # Close Gripper
            grip_torque = 0.4
            sim.data.ctrl[6] = grip_torque
            sim.data.ctrl[7] = grip_torque
            sim.data.ctrl[8] = grip_torque
        # change path
        if controller == 'PID':
            q_path = q_path2target
            q_dot = q_dot_2target
            if hybrid:
                controller = 'impedance'
                p = path2target
                p_dot = p_dot_2target
                p_2dot = p_2dot_2target
        else:
            p = path2target
            p_dot = p_dot_2target
            p_2dot = p_2dot_2target
        k = 0
        delay = delay_time  # delay in sec
        advance_q_path = 0
        # log data
        actual_q = np.append(actual_q, np.reshape(q_r, (6, 1)), axis=1)
        actual_q_dot = np.append(actual_q_dot, np.reshape(q_r_dot, (6, 1)), axis=1)

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

    # our_func.contacts(sim)
    # last_forces_avg, Force = contacts.find_contacts(sim, 1, 'cylinder1') #, last_forces_avg)
    Force = contacts.find_contacts(sim, 1, 'cylinder1')
    # Force = Force/101
    force_log = np.append(force_log, np.reshape(Force, (3, 1)), axis=1)
    force_log = my_filter.butter_lowpass_filter(force_log, 5, 20, 2)

    if need_render:
        viewer.render()

    if delay_flag == 1:
        delay -= 1
        if delay == 0:
            delay_flag = 0
            move_flag = 1

    # Loop number
    # print('Loop number: ', sim_loop_num)
    sim_loop_num += 1

    if print_status and sim_loop_num > sim_time/DT:
        viewer._paused = True
        # print desired vs actual (q, q_dot)
        our_func.print_q_actual(q_path2cylinder, q_path2target, actual_q, DT, 'angle')
        # our_func.print_q_actual(q_dot_2cylinder, q_dot_2target, actual_q_dot, DT, 'speed')
        # print Joint Position Error
        our_func.print_pos_error(pos_error_log, DT)
        # print Control effort
        our_func.print_torque(u_log, DT)
        # force print
        our_func.print_force_scope2(force_log, DT)
        # np.save("a2.npy",force_log)
        # Path print
        our_func.print_xyz(x_r, x_m, x_0, DT)
        our_func.print_xyz_3D(x_cylinder)
        our_func.print_difference(x_r, x_m, x_0, DT)
        print ('Time:' , T)
        # print Force from sensor
        # our_func.print_force(f_in_log, DT)

        # filter and print Force
        # filtered_force = my_filter.butter_lowpass_filter(f_in_log, 5, 500, 5)
        # our_func.print_force(filtered_force, DT)

        # force scope
        # our_func.print_force_scope(force_scope_log, DT)
        break

    if os.getenv('TESTING') is not None:
        break
