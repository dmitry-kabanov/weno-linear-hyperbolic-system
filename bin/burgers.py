from wenohj.solver import Solver
import numpy as np
import matplotlib.pyplot as plt


def get_alpha(x, t, u, u_x_plus, u_x_minus):
    # For Burgers equation derivative of H doesn't depend on u_x.
    max = np.zeros_like(x)

    for i in range(len(x)):
        if np.abs(u_x_plus[i] + 1.0) > np.abs(u_x_minus[i] + 1.0):
            max[i] = np.abs(u_x_plus[i] + 1.0)
        else:
            max[i] = np.abs(u_x_minus[i] + 1.0)

    return max


def flux(x, t, u, u_x):
    return (u_x + 1)**2 / 2.0

lb = -1.0
rb = 1.0
ncells = 80
T = 3.5 / np.pi**2

s = Solver(lb, rb, ncells, flux, get_alpha, 'periodic')
x = s.get_x()
u0 = -np.cos(np.pi * x)
solution = s.solve(u0, T)

plt.plot(x, solution)
plt.show()
