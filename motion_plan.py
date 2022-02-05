import numpy as np
from mujoco_py import functions
from numpy.linalg import inv
import UR5_kinematics as UR_kin
import dimanic_mat as dm
import R_construct as rc


def motion_plan(start_point, end_point, time_to_execut, time_diff):
    # get points as homogeneous transformation matrix
    # np.array([[r11, r12, r13, Px],
    #           [r21, r22, r23, Py],
    #           [r31, r32, r33, Pz],
    #           [  0,   0,   0,  1]])
    # time diff determine the number of steps (should be based on the sim time step)

    # convenient, don't really need this
    T = time_to_execut

    # create time vector
    reso = int(T / time_diff)
    t = np.linspace(0, T, reso)

    # --- linear movement of end effector --- #
    # coefficients
    a3 = 10
    a4 = -15 / T
    a5 = 6 / (T ** 2)
    t3 = np.power(t, 3)
    t4 = np.power(t, 4)
    t5 = np.power(t, 5)
    t_coef = a5 * t5 + a4 * t4 + a3 * t3

    # path output init
    p = np.zeros((3, len(t)))

    # get position vectors
    v0 = start_point[0:3, -1]
    vf = end_point[0:3, -1]

    for k in range(3):
        coeff = (vf[k] - v0[k]) / (T ** 3)
        p[k, :] = coeff * t_coef + v0[k]

    # --- Rotary movement of end effector --- #
    # Rotation matrix between the two points (points are given in world cord.)
    R = inv(start_point[0:3, 0:3]) @ end_point[0:3, 0:3]

    # if Rotation is needed find Rotation vector and angle
    if np.any(R - np.identity(3)):
        trace = np.trace(R)
        angle = np.arccos((trace - 1) / 2)
        angle_diff = angle / reso
        n_sub = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        n_vec_diff = (1 / (2 * np.sin(angle))) * n_sub
        R_diff = rc.R_axis_and_angle(n_vec_diff, angle_diff)

    # init first point
    full_path = np.zeros((reso, 4, 4))
    R_start = start_point[0:3, 0:3]
    full_path[0] = build_target(p[:, 0], R_start)
    R_next = R_start

    # connect Linear and Rotary movements for homogeneous transformation matrix
    #  ** Not implemented **
    for j in range(1, reso):
        # if need Rotation, multiply current R with diff Rotation
        if np.any(R-np.identity(3)):
            R_next = R_next @ R_diff

        full_path[j] = build_target(p[:, j], R_next)

    return full_path


def get_joints(sim, q_r, q_r_dot):
    joints = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint',
              'wrist_3_joint']

    for joint in range(len(joints)):
        q_r[joint] = sim.data.get_joint_qpos(joints[joint])
        q_r_dot[joint] = sim.data.get_joint_qvel(joints[joint])

    return q_r, q_r_dot


def execute_time(start_point, end_point, speed):
    # calculate execute time based on distance between the points
    distance = np.linalg.norm(end_point - start_point, 2)
    ex_time = round(distance / speed, 2)
    if ex_time < 1:
        return ex_time * 3
    else:
        return ex_time


def xyz_velocity(start_point, end_point, time_to_execut, time_diff):
    # convenient, don't really need this
    T = time_to_execut

    # create time vector
    reso = int(T / time_diff)
    t = np.linspace(0, T, reso)

    # --- linear movement of end effector --- #
    # coefficients
    a2 = 30
    a3 = -60 / T
    a4 = 30 / (T ** 2)
    t2 = np.power(t, 2)
    t3 = np.power(t, 3)
    t4 = np.power(t, 4)
    t_coef = a4 * t4 + a3 * t3 + a2 * t2

    # path output init
    p_dot = np.zeros((3, len(t)))

    # get position vectors
    v0 = start_point[0:3, -1]
    vf = end_point[0:3, -1]

    for k in range(3):
        coeff = (vf[k] - v0[k]) / (T ** 3)
        p_dot[k, :] = coeff * t_coef

    return p_dot


def xyz_acc(start_point, end_point, time_to_execut, time_diff):
    # convenient, don't really need this
    T = time_to_execut

    # create time vector
    reso = int(T / time_diff)
    t = np.linspace(0, T, reso)

    # --- linear movement of end effector --- #
    # coefficients
    a1 = 60
    a2 = -180 / T
    a3 = 120 / (T ** 2)
    t2 = np.power(t, 2)
    t3 = np.power(t, 3)
    t_coef = a3 * t3 + a2 * t2 + a1 * t

    # path output init
    p_2dot = np.zeros((3, len(t)))

    # get position vectors
    v0 = start_point[0:3, -1]
    vf = end_point[0:3, -1]

    for k in range(3):
        coeff = (vf[k] - v0[k]) / (T ** 3)
        p_2dot[k, :] = coeff * t_coef

    return p_2dot


def build_cartesian_path(path, move_speed, time_diff):
    # First leg of path
    ex_time = execute_time(path[0], path[1], move_speed)
    xyz_path = motion_plan(path[0], path[1], ex_time, time_diff)
    p_dot = xyz_velocity(path[0], path[1], ex_time, time_diff)
    p_2dot = xyz_acc(path[0], path[1], ex_time, time_diff)

    # Rest of the path
    for k in range(1, len(path) - 1):
        ex_time = execute_time(path[k], path[k + 1], move_speed)
        next_part = motion_plan(path[k], path[k + 1], ex_time, time_diff)
        xyz_path = np.concatenate((xyz_path, next_part), axis=0)

        # velocity
        next_p_dot = xyz_velocity(path[k], path[k + 1], ex_time, time_diff)
        p_dot = np.concatenate((p_dot, next_p_dot), axis=1)

        # acceleration
        next_p_2dot = xyz_acc(path[k], path[k + 1], ex_time, time_diff)
        p_2dot = np.concatenate((p_2dot, next_p_2dot), axis=1)
    time = range(len(xyz_path[:, 2, 3]))
    # plt.plot(time, xyz_path[:, 2, 3])
    # plt.show()
    return xyz_path, p_dot, p_2dot


def build_trajectory(xyz_path):
    q_path = np.zeros((6, len(xyz_path[:, 3, 3])))
    for k in range(len(xyz_path[:, 3, 3])):
        q_path[:, k] = UR_kin.inv_kin(xyz_path[k - 1], xyz_path[k])
    return q_path


def build_target(x_desired, r_desired):
    # x_desired = np.array([x, y, z])
    #
    # r_desired = np.array([[1, 0,  0],
    #                       [0, 0,  1],
    #                       [0, -1, 0]])

    v_zero = np.array([[0, 0, 0]])
    r_desired = np.concatenate((r_desired, v_zero),axis=0)
    x_desired = np.append(x_desired, 1)
    x_desired = np.reshape(x_desired, (4, 1))
    target = np.concatenate((r_desired, x_desired), axis=1)
    return target


def jacob(sim):
    target_jacp = np.zeros(3 * sim.model.nv)
    target_jacr = np.zeros(3 * sim.model.nv)
    sim.data.get_site_jacp('gripperpalm', jacp=target_jacp)
    sim.data.get_site_jacr('gripperpalm', jacr=target_jacr)
    J_L = target_jacp.reshape((3, sim.model.nv))
    J_A = target_jacr.reshape((3, sim.model.nv))

    J_L = J_L[:, 6:12]
    J_A = J_A[:, 6:12]
    J = np.concatenate((J_L, J_A), axis=0)

    return J, J_L, J_A


def q_velocity(sim, p_dot, q_path):
    # # initialize joint position and velocity vectors
    # q_r = np.zeros((6,))
    # q_r_dot = np.zeros((6,))
    # q_dot = np.zeros((6, 1))
    # # get joints
    # q_r, q_r_dot = get_joints(sim, q_r, q_r_dot)
    #
    # # find Jacobian
    # j_muj, j_l_muj, _ = jacob(sim)
    # j = dm.Jacobian(q_r)
    #
    # J_diff = j-j_muj
    # print(J_diff)

    # # Need to finish this part
    # j_l = j[0:3, :]
    #
    # # first time
    # q_dot_next = j_l.T @ p_dot[:, 0]
    # q_dot_next = np.reshape(q_dot_next, (6, 1))
    # q_dot = q_dot_next

    for i in range(len(p_dot[0, :])):
        j_muj, j_l_muj, _ = jacob(sim)  # for test

        j = dm.Jacobian(q_path[:, i])
        j_l = j[0:3, :]
        q_dot_next = j_l.T @ p_dot[:, i]
        q_dot_next = np.reshape(q_dot_next, (6, 1))
        if i:
            q_dot = np.concatenate((q_dot, q_dot_next), axis=1)
        else:
            q_dot = q_dot_next

    return q_dot


def H(sim):
    J, J_L, J_A = jacob(sim)
    H = np.zeros(sim.model.nv * sim.model.nv)
    functions.mj_fullM(sim.model, H, sim.data.qM)
    H_L = np.dot(np.linalg.pinv(J_L.T), np.dot(H.reshape(sim.model.nv, sim.model.nv), np.linalg.pinv(J_L)))
    H_all = np.dot(np.linalg.pinv(J.T), np.dot(H.reshape(sim.model.nv, sim.model.nv), np.linalg.pinv(J)))
    return H_all, J