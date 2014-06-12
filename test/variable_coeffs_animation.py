from wenohj.solver import Solver
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import sys
from time import strftime


def get_alpha(x):
    max_lambda1 = np.abs(8.0 - x)
    max_lambda2 = 2.0

    return np.maximum(max_lambda1, max_lambda2)


def get_system_matrix(x):
    mat = np.zeros((2, 2, len(x)))
    mat[0][0] = 8.0 - x
    mat[0][1] = 0.0
    mat[1][0] = 0.0
    mat[1][1] = 2.0

    return mat


def ic_func(arg):
    return np.exp(-(arg - 5.0)**2) + np.exp(-(arg - 11.0)**2)


def animation_init():
    #line.set_data(x, ic[0])
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


def animation_frame(i):
    global soln_u
    line.set_data(x, soln_u[i])
    time_text.set_text('time = {0:.2f}'.format(i * dt))
    return line, time_text

lb = 0.0
rb = 16.0
ncells = 10000
T = 10.0
NFRAMES = 400
dt = T / NFRAMES

fig = plt.figure()
ax = plt.axes(xlim=(lb, rb), ylim=(-0.1, 1.1))
line, = ax.plot([], [], linewidth=2)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

s = Solver(lb, rb, ncells, get_system_matrix, get_alpha, 'periodic')
x = s.get_x()
u0 = np.zeros_like(x)
v0 = np.zeros_like(x)
u0 = ic_func(x)

ic = np.zeros((2, len(x)))
ic[0] = u0
ic[1] = v0
soln_u = np.zeros((NFRAMES, len(x)))
soln_v = np.zeros((NFRAMES, len(x)))
soln_u[0] = ic[0]
soln_v[0] = ic[1]

for i in range(1, NFRAMES):
    soln_u[i] = ic_func(8.0 - (8.0 - x) * np.exp(i*dt))

anim = animation.FuncAnimation(fig, animation_frame,
                               init_func=animation_init,
                               frames=NFRAMES, interval=200, blit=True)
filename = 'videos/var_coeffs_diag_u.mp4'
anim.save(filename, fps=30, writer=animation.FFMpegFileWriter(), extra_args=['-vcodec', 'libx264'])
#plt.show()
