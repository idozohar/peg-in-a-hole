#  this function takes the rotation axis and angle and converts them into the appropriate Rotation matrix.
#  for more info read : https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle

import numpy as np


def R_axis_and_angle(a, theta):  # a is the axis and theta is given in radians
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    v_theta = (1 - np.cos(theta))
    R = np.array([[c_theta + a[0]**2 * v_theta, a[0] * a[1] * v_theta - a[2] * s_theta, a[0] * a[2] * v_theta + a[1] * s_theta],
                 [a[0] * a[1] * v_theta + a[2] * s_theta, c_theta + a[1]**2 * v_theta, a[1] * a[2] * v_theta - a[0] * s_theta],
                 [a[0] * a[2] * v_theta - a[1] * s_theta, a[1] * a[2] * v_theta + a[0] * s_theta, c_theta + a[2]**2 * v_theta]])
    return R

