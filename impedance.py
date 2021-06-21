import numpy as np
import UR5_kinematics as UR_kin
from mujoco_py import functions
from numpy.linalg import inv
import our_func
import my_filter
from scipy.spatial.transform import Rotation as R


def imic(sim, K, B, M, q_r, q_r_dot, p, p_dot, dt, f_in):
    # Full Instantaneous model - Force & Torque

    # Virtual position & orientation - X_0
    p_im = p.copy()
    start = p.copy()
    p_pos = start[0:3, 3]  # Virtual position
    rot_mat_0 = start[0:3, 0:3]  # Virtual rotation Matrix
    rot_R_0 = R.from_matrix(rot_mat_0)  # Scipy Rotation object
    euler_0 = rot_R_0.as_rotvec()  # Extracting orientation angles - Roll, Pitch, Yaw
    p_0 = np.append(p_pos, euler_0)
    euler_dot_0 = np.zeros((1, 3))  # desired orientation is constant
    p_dot_0 = np.append(p_dot, euler_dot_0)

	# Get current Jacobian
    j_muj, j_l_muj, _ = our_func.jacob(sim)

    # Real position & orientation - X_r
    x_r_mat = UR_kin.forward(q_r)
    x_r_pos = x_r_mat[0:3, 3]
    rot_mat_r = x_r_mat[0:3, 0:3]
    rot_R = R.from_matrix(rot_mat_r)
    euler_r = rot_R.as_rotvec()
    x_r = np.append(x_r_pos, euler_r)
    x_r_dot = j_muj @ q_r_dot
    # print(euler_r)
    # f_in[0:2] = np.array([0, 0])
    # Impedance model equation
    integrand = inv(M) @ (-f_in + B @ (p_dot_0 - x_r_dot) + K @ (p_0 - x_r))  # X_m_2dot
    # Discrete integral for X_m_dot & X_m
    x_im_dot = x_r_dot + dt * integrand
    x_im = x_r + dt * x_im_dot

    # print('position error: ', p_pos - x_r)
    # # print('inv(M): ', inv(M))
    # print('x_im: ', x_im)
    # print('x_im_dot: ', x_im_dot)
    # print('Force: ', f_in)
    # print('--------------------------------------------- \n')

    im_pos = x_im[0:3]
    im_euler = x_im[3:]
    p_im[0:3, 3] = im_pos
    rot_R_im = R.from_rotvec(im_euler)
    im_rot_mat = rot_R_im.as_matrix()
    p_im[0:3,0:3] = im_rot_mat
    try:
        q_d = UR_kin.inv_kin(start, p_im)
    except:
        print('Inverse Kinematics Failed!')
        q_d = q_r  

    try:
        q_d_dot = j_muj.T @ x_im_dot
    except:
        q_d_dot = q_r_dot
        print('Jacobian was singular! ', q_d[4])

    return q_d, q_d_dot, x_im
