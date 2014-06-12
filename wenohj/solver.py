import numpy as np


class Solver(object):
    def __init__(self, lb, rb, ninternal_points, get_system_matrix, get_alpha,
                 bt, cfl=0.6):
        """@todo: Docstring for __init__.

        :ncells: @todo
        :returns: @todo

        """
        self.lb = lb
        self.rb = rb
        self.ninternal_points = ninternal_points
        self.get_system_matrix = get_system_matrix
        self.get_alpha = get_alpha
        self.boundary_type = 'periodic'
        self.cfl = cfl

        # Number of ghost points on each side of the computational domain.
        self.nb = 3

        # Parameter used in WENO algorithm to prevent overflow in weights.
        self.eps = 1.0e-6

        # Spatial step.
        self.dx = (rb - lb) / (self.ninternal_points + 1.0)

        # Total number of the grid points used for the algorithm.
        # This is a scheme of the grid structure:
        # *--*--*--^--o--o--...--o--o--^--*--*--*
        # Legend: * - ghost points, ^ - boundary points, o - internal points
        self.npoints = self.ninternal_points + 2 + 2*self.nb

        # Index of the leftmost point of the physical domain.
        self.left = self.nb
        # Index of the rightmost point of the physical domain.
        self.right = self.nb + self.ninternal_points + 1

        # Grid.
        self.x = np.linspace(lb - self.nb * self.dx, rb + self.nb * self.dx,
                             self.npoints)

        self.c1 = 1.0 / 3.0
        self.c2 = 1.0 / 6.0

    def solve(self, ic, final_time):
        size = len(self.x)
        u0 = np.zeros(size*2)
        u1 = np.zeros(size*2)
        u2 = np.zeros(size*2)

        l1 = self.left
        l2 = self.right + 1
        u0[l1:l2] = ic[0]
        u0[self.npoints + l1:self.npoints + l2] = ic[1]

        self.t = 0
        self.alpha_global = self.get_alpha(self.x[l1:l2])
        alpha = np.max(self.alpha_global)
        self.dt = self.cfl * self.dx / alpha

        while (self.t < final_time):
            if (self.t + self.dt > final_time):
                self.dt = final_time - self.t

            self.t += self.dt

            u1 = u0 + self.dt * self._rhs(u0)
            u2 = (3.0 * u0 + u1 + self.dt * self._rhs(u1)) / 4.0
            u0 = (u0 + 2.0 * u2 + 2.0 * self.dt * self._rhs(u2)) / 3.0

        return u0[l1:l2], u0[self.npoints + l1:self.npoints + l2]

    def get_x(self):
        return self.x[self.left:self.right + 1]

    def _rhs(self, values):
        rhs_values = np.zeros_like(values)
        l1 = self.left
        l2 = self.right + 1

        u = np.zeros_like(self.x)
        v = np.zeros_like(self.x)
        u[l1:l2] = values[l1:l2]
        v[l1:l2] = values[self.npoints + l1:self.npoints + l2]

        u_x_plus, u_x_minus = self._interpolate(u)
        v_x_plus, v_x_minus = self._interpolate(v)
        avg_u = (u_x_plus + u_x_minus) / 2.0
        avg_v = (v_x_plus + v_x_minus) / 2.0

        mat = self.get_system_matrix(self.x[l1:l2])
        flux_u = mat[0][0]*avg_u + mat[0][1]*avg_v
        flux_v = mat[1][0]*avg_u + mat[1][1]*avg_v

        alpha = self.alpha_global

        rhs_values[l1:l2] = flux_u - alpha * (u_x_plus - u_x_minus) / 2.0
        rhs_values[self.npoints+l1:self.npoints+l2] = flux_v - alpha * (
            v_x_plus - v_x_minus) / 2.0
        return -rhs_values

    def _interpolate(self, u):
        self._apply_boundary_conditions(u)
        l1 = self.nb
        l2 = self.nb + self.ninternal_points + 2

        der1 = u[l1-1:l2-1] - u[l1-2:l2-2]
        der2 = u[l1:l2] - u[l1-1:l2-1]
        der3 = u[l1+1:l2+1] - u[l1:l2]
        der4 = u[l1+2:l2+2] - u[l1+1:l2+1]
        numer = -der1 + 7 * der2 + 7 * der3 - der4
        common = numer / (12.0 * self.dx)

        # Compute second derivatives
        der1 = (u[l1+3:l2+3] - 2*u[l1+2:l2+2] + u[l1+1:l2+1]) / self.dx
        der2 = (u[l1+2:l2+2] - 2*u[l1+1:l2+1] + u[l1:l2]) / self.dx
        der3 = (u[l1+1:l2+1] - 2*u[l1:l2] + u[l1-1:l2-1]) / self.dx
        der4 = (u[l1:l2] - 2*u[l1-1:l2-1] + u[l1-2:l2-2]) / self.dx
        der5 = (u[l1-1:l2-1] - 2*u[l1-2:l2-2] + u[l1-3:l2-3]) / self.dx

        weno_plus_flux = self._weno_flux(der1, der2, der3, der4)
        u_x_plus = common + weno_plus_flux
        weno_minus_flux = self._weno_flux(der5, der4, der3, der2)
        u_x_minus = common - weno_minus_flux

        return u_x_plus, u_x_minus

    def _weno_flux(self, a, b, c, d):
        """Calculate WENO approximation of the flux.
        :returns: @todo

        """
        is0 = 13.0*(a - b)**2 + 3.0*(a - 3*b)**2
        is1 = 13.0*(b - c)**2 + 3.0*(b + c)**2
        is2 = 13.0*(c - d)**2 + 3.0*(3*c - d)**2

        alpha0 = 1.0 / (self.eps + is0)**2
        alpha1 = 6.0 / (self.eps + is1)**2
        alpha2 = 3.0 / (self.eps + is2)**2
        sum_alpha = alpha0 + alpha1 + alpha2
        w0 = alpha0 / sum_alpha
        w2 = alpha2 / sum_alpha

        result = self.c1*w0*(a - 2*b + c) + self.c2*(w2 - 0.5)*(b - 2*c + d)
        return result

    def _apply_boundary_conditions(self, u):
        if self.boundary_type == 'periodic':
            for i in range(0, self.nb + 1):
                u[self.left - i] = u[self.right - i]
                u[self.right + i] = u[self.left + i]
