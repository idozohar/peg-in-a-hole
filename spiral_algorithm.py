import numpy as np

def next_point(P0, theta):
    # position of the end
    P0 = P0[0:3, 3]
    a = 0.0012
    b = 0.0012
    # a = 0.00005
    # b = 0.00005
    # polar cordinate
    r = a + b*theta
    return np.array([P0[0] + r*np.cos(theta), P0[1] + r*np.sin(theta) ,P0[2]])# x,y

