import numpy as np
from numpy import cos
from numpy import sin
from numpy.linalg import inv


# UR5 DH parameters
a = np.array([0, -0.425, -0.39225, 0, 0, 0])
d = np.array([0.089159, 0, 0, 0.10915, 0.09465, 0.0823])
alpha = np.array([np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0])

# UR5e DH parameters
# a = np.array([0, -0.425, -0.3922, 0, 0, 0])
# d = np.array([0.1625, 0, 0, 0.1333, 0.0997, 0.0996])
# alpha = np.array([np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0])


# ------------------------ FORWARD KINEMATICS ------------------------ #

def a0_i(i, q):
    # Transformation matrix from i-1 to i (DH convention)
    A_i = np.array([[cos(q[i-1]), -sin(q[i-1])*cos(alpha[i-1]), sin(q[i-1])*sin(alpha[i-1]), a[i-1]*cos(q[i-1])],
                    [sin(q[i-1]),  cos(q[i-1])*cos(alpha[i-1]), -cos(q[i-1])*sin(alpha[i-1]), a[i-1]*sin(q[i-1])],
                    [0,                      sin(alpha[i-1]),           cos(alpha[i-1]),              d[i-1]],
                    [0,                              0,                       0,                         1]])
    return A_i


def forward(q):
    # Transformation matrix from Base to Gripper (DH convention)
    a_i = np.identity(4)
    for i in range(1, len(q) + 1):
        a_i = a_i @ a0_i(i, q)
    return a_i


def a14(theta, target):
    # Transformation matrix from 1 to 4 ( after q1, q5 and q6 are known )
    theta = np.round(theta, 5)
    a_01 = a0_i(1, theta)
    a_06 = target
    a_16 = inv(a_01) @ a_06
    a_45 = a0_i(5, theta)
    a_56 = a0_i(6, theta)
    a_14 = a_16 @ inv((a_45 @ a_56))
    return a_14


# ------------------------ INVERSE KINEMATICS ------------------------ #

def inv_kin(start, target):
    # start and target are homogeneous transformation matrix
    # np.array([[Xx, Yx, Zx, Px],
    #           [Xy, Yy, Zy, Py],
    #           [Xz, Yz, Zz, Pz],
    #           [ 0,  0,  0,  1]])

    theta = np.zeros(6)

    # extract desired position of gripper from target matrix
    p06 = target[:, 3]

    # to find wrist3 origin(P05) move -d6 on gripper Z axis
    # align Z axis with desired Orientation
    p05 = p06 - d[5]*target[:, 2]

    # ---- theta 1 ----
    # p05_xy = sqrt(Px^2 + Py^2)
    p05_xy = np.linalg.norm(p05[0:2], 2)

    # check if the position is in the work space
    if p05_xy < d[5]:
        print('Out of Work-Space ( inside cylinder )')

    # phi = atan2(Py, Px)
    phi1 = np.arctan2(p05[1], p05[0])

    # two possible solutions for phi --> "Left" or "Right" shoulder
    phi2_plus = np.arccos(d[3]/p05_xy)
    # phi2_minus = -phi2_plus

    # <----------------------------------- add choice for shoulder -----

    # found theta1
    theta[0] = phi1 + phi2_plus + np.pi/2

    # ---- theta 5 ----
    # p16_z = Px * sin(theta1) - Py * cos(theta1)
    p16_z = p06[0] * sin(theta[0]) - p06[1] * cos(theta[0])

    # two possible solutions for theta5 --> "Up" or "Down" Wrist
    arg = round((p16_z-d[3]) / d[5], 5)
    theta5_plus = np.arccos(arg)
    # theta5_minus = -theta5_plus

    # <-------------------------------------------- add choice -----
    # found theta5
    theta[4] = theta5_plus

    # ---- theta 6 ----
    if sin(theta[4]) == 0:
        # Rotation matrix between the two points (points are given in world cord.)
        R = inv(start[0:3, 0:3]) @ target[0:3, 0:3]
        trace = np.trace(R)
        angel = np.arccos((trace - 1) / 2)  # Positive angel of Rotation
        if angel == 0:
            theta[5] = angel
        else:
            n_sub = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
            n_vec = (1 / (2 * np.sin(angel))) * n_sub
            n_vec = np.round(n_vec, 5)
            # Here we need to decide the direction of Rotation
            if n_vec[0] <= 0 and n_vec[1] <= 0 and n_vec[2] <= 0:
                angel = -angel
            theta[5] = angel

        print('Singular Point: theta6 is 0 !')

    else:
        a_01 = a0_i(1, theta)
        a_61 = inv(inv(a_01) @ target)
        A = round(-a_61[1, 2] / sin(theta[4]), 5)
        B = round(a_61[0, 2] / sin(theta[4]), 5)
        theta[5] = np.arctan2(A, B)

    # ---- theta 3 ----
    a_14 = a14(theta, target)

    p13 = a_14[:, -1] - d[3]*a_14[:, 1]
    p13_size = np.linalg.norm(p13[0:3], 2)

    A1 = round(p13_size**2 - a[1]**2 - a[2]**2, 9)
    A2 = 2*a[1]*a[2]
    if 0 <= A1 - A2 <= 0.01:
        theta3_plus = np.arccos(1)
    elif abs(A1/A2) > 1:
        print()
        print('Out of Work-Space!! ( Outside of sphere - too far )', end='\n')
        print('please change point...')
        theta3_plus = -999
        quit()
    else:
        theta3_plus = np.arccos(A1/A2)

    # two possible solutions for theta3 --> "Up" or "Down" elbow
    # theta3_minus = -theta3_plus
    theta[2] = theta3_plus

    # ---- theta 2 ----
    at2 = -np.arctan2(p13[1], -p13[0])
    ac2 = np.arcsin(a[2]*sin(theta[2]) / p13_size)
    theta[1] = at2 + ac2

    # ---- theta 4 ----
    a_12 = a0_i(2, theta)
    a_23 = a0_i(3, theta)
    a_34 = inv(a_12 @ a_23) @ a_14
    theta[3] = np.arctan2(a_34[1, 0], a_34[0, 0])

    theta = np.round(theta, 5)
    return theta
