from wenohj.solver import Solver
import numpy as np
import matplotlib.pyplot as plt
import sys
from time import strftime


def get_alpha(x):
    max_lambda1 = np.abs(1.0 - x)
    max_lambda2 = 2.0

    return np.maximum(max_lambda1, max_lambda2)


def get_system_matrix(x):
    mat = np.zeros((2, 2, len(x)))
    mat[0][0] = 1.0 - x
    mat[0][1] = 0.0
    mat[1][0] = 0.0
    mat[1][1] = 2.0

    return mat


def ic_func(arg):
    return np.exp(-1.0 * (arg - 4.0)**2)

lb = 0.0
rb = 8.0
ncells = 1000
T = 4.0

s = Solver(lb, rb, ncells, get_system_matrix, get_alpha, 'periodic')
x = s.get_x()
u0 = np.zeros_like(x)
v0 = np.zeros_like(x)
u0 = ic_func(x)

ic = np.zeros((2, len(x)))
ic[0] = u0
ic[1] = v0
u, v = s.solve(ic, T)

exact_u = ic_func(1.0 - (1 - x) * np.exp(T))

plt.plot(x, u)
plt.plot(x, exact_u)

if len(sys.argv) > 1 and sys.argv[1] in ['save_fig']:
    dt = strftime('%Y-%m-%dT%H%M%S')
    fig_name = 'images/variable_coeffs_T=' + str(T) + '_' + dt + '.pdf'
    plt.savefig(fig_name)
plt.show()
