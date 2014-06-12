from wenohj.solver import Solver
import numpy as np
import matplotlib.pyplot as plt


def get_alpha(x):
    return 3.0


def get_system_matrix(x):
    mat = np.zeros((2, 2, len(x)))
    mat[0][0] = 2.0
    mat[0][1] = 1.0
    mat[1][0] = 1.0
    mat[1][1] = 2.0

    return mat


def ic_func(arg):
    return np.exp(-20 * (arg - 2.0)**2)

lb = -5.0
rb = 8.0
ncells = 1000
T = 1.0

s = Solver(lb, rb, ncells, get_system_matrix, get_alpha, 'periodic')
x = s.get_x()
u0 = np.zeros_like(x)
v0 = np.zeros_like(x)
u0 = ic_func(x)

ic = np.zeros((2, len(x)))
ic[0] = u0
ic[1] = v0
u, v = s.solve(ic, T)

exact_u = 0.5 * (ic_func(x - 3*T) + ic_func(x - T))

plt.plot(x, u)
plt.plot(x, exact_u)
plt.show()
