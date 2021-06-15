import numpy as np
import matplotlib
import seaborn as sb
import matplotlib.pyplot as plt
a1 = np.load('a1.npy')
a2 = np.load('a2.npy')
dt = 0.002
t_len = len(a2[0, :]) * dt
t = np.linspace(0, t_len, len(a2[0, :]))
f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='col')
ax1.plot(t, a1[0, :], 'r', t, a2[0, :], '--b')
ax1.grid(axis='both')
ax1.set_ylabel('Force X')
ax2.plot(t, a1[1, :], 'r', t, a2[1, :], '--b')
ax2.grid(axis='both')
ax2.set_ylabel('Force Y')
ax3.plot(t, a1[1, :], 'r', t, a2[1, :], '--b')
ax3.grid(axis='both')
ax3.set_ylabel('Force Z')
ax3.set_xlabel('Time [sec]')
plt.legend(['Impedance', 'PID'])
plt.show()