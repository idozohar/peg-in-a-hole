import numpy as np
from numpy.linalg import inv
import our_func


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

    # if Rotation is needed find Rotation vector and angel
    if np.any(R - np.identity(3)):
        trace = np.trace(R)
        angel = np.arccos((trace - 1) / 2)
        angel_diff = angel / reso
        n_sub = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        n_vec_diff = (1 / (2 * np.sin(angel_diff))) * n_sub

    # init first point
    full_path = np.zeros((reso, 4, 4))
    R_start = start_point[0:3, 0:3]
    full_path[0] = our_func.build_target(p[:, 0], R_start)
    R_next = R_start

    # connect Linear and Rotary movements for homogeneous transformation matrix
    #  ** Not implemented **
    for j in range(1, reso):
        # if need Rotation, multiply current R with diff Rotation
        if np.any(R-np.identity(3)):
            R_next = R_next @ n_vec_diff

        full_path[j] = our_func.build_target(p[:, j], R_next)

    return full_path
