from wenohj.solver import Solver
import numpy as np
import matplotlib.pyplot as plt
import sys
from time import strftime


def get_alpha(x):
    max_lambda1 = np.abs(SONIC_LOCUS_LOCATION - x)
    max_lambda2 = 2.0

    return np.maximum(max_lambda1, max_lambda2)


def get_system_matrix(x):
    mat = np.zeros((2, 2, len(x)))
    mat[0][0] = SONIC_LOCUS_LOCATION - x
    mat[0][1] = 0.0
    mat[1][0] = 0.0
    mat[1][1] = 2.0

    return mat


def ic_func(arg):
    return np.exp(-(arg - 5.0)**2) + np.exp(-(arg - 11.0)**2)

if len(sys.argv) <= 1 or sys.argv[1] not in ['compute', 'plot']:
    print('USAGE: python bin/convergence_variable_coefficients.py <command>\n'
          'where <command> is one of the following:\n\n'
          '  compute - to compute convergence rate and write results to file '
          'in data folder\n'
          '  plot    - to read results from file and plot figure.\n'
          '            Optional flag save_fig can be used to save figure \n'
          '            to file in the images folder.\n')
    sys.exit(1)

filename = 'data/convergence_variable_coeffs.txt'
lb = 0.0
rb = 16.0
T = 1.0
SONIC_LOCUS_LOCATION = 8.0

npoints = [160, 320, 640, 1280, 2560, 5120]
errors_L1 = []
errors_Linf = []

if sys.argv[1] == 'compute':
    for n in npoints:
        # Spatial step. We add 1 because n is the number of internal points.
        dx = (rb - lb) / (n + 1.0)
        cfl_number = dx
        s = Solver(lb, rb, n, get_system_matrix, get_alpha, 'periodic',
                   cfl=cfl_number)
        x = s.get_x()
        u0 = ic_func(x)
        v0 = np.zeros_like(x)
        ic = np.zeros((2, len(x)))
        ic[0] = u0
        ic[1] = v0
        solution_u, solution_v = s.solve(ic, T)
        exact_u = ic_func(
            SONIC_LOCUS_LOCATION - (SONIC_LOCUS_LOCATION - x) * np.exp(T))
        error = dx * np.linalg.norm(exact_u - solution_u, 1)
        errors_L1.append(error)
        error = np.linalg.norm(exact_u - solution_u, np.inf)
        errors_Linf.append(error)

    with open(filename, mode='w', encoding='utf-8') as outfile:
        for i in range(len(errors_L1)):
            line = '{0:4d} {1:22.16e} {2:22.16e}'.format(
                npoints[i], errors_L1[i], errors_Linf[i])
            print(line)
            outfile.write(line + '\n')
else:
    with open(filename, mode='r', encoding='utf-8') as infile:
        for line in infile:
            _, error_L1, error_Linf = line.split()
            errors_L1.append(float(error_L1))
            errors_Linf.append(float(error_Linf))

        assert len(npoints) == len(errors_L1), \
            'Lengths of npoints and errors_L1 mismatch'
        assert len(npoints) == len(errors_Linf), \
            'Lengths of npoints and errors_Linf mismatch'

        print('{0:4d};{1:8.2e};{2:6s};{3:8.2e};{4:6s}'.format(
            npoints[0], errors_L1[0], '-', float(errors_Linf[0]), '-'))
        for i in range(1, len(npoints)):
            err_ratio = np.log(errors_L1[i] / errors_L1[i-1])
            step_ratio = np.log((npoints[i-1] + 1.0) / (npoints[i] + 1.0))
            p_L1 = err_ratio / step_ratio
            err_ratio = np.log(errors_Linf[i] / errors_Linf[i-1])
            step_ratio = np.log((npoints[i-1] + 1.0) / (npoints[i] + 1.0))
            p_Linf = err_ratio / step_ratio
            print('{0:4d};{1:8.2e};{2:4.2f};{3:8.2e};{4:4.2f}'.format(
                npoints[i], errors_L1[i], p_L1, errors_Linf[i], p_Linf))

    plt.loglog(npoints, errors_L1, '-o', label=r'$\|\| E \|\|_1$')
    n_list = [5e8 * n**(-5) for n in npoints]
    plt.loglog(npoints, n_list, '-s', label=r'$N^{-5}$')
    plt.xlabel(r'$N$')
    plt.ylabel(r'$\|\| E \|\|_1$')
    plt.legend(loc='upper right')
    if len(sys.argv) == 3 and sys.argv[2] == 'save_fig':
        dt = strftime('%Y-%m-%dT%H%M%S')
        filename = 'images/convergence_advection_' + dt
        plt.savefig(filename)
    else:
        plt.show()
