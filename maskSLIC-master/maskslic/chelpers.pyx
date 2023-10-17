
from scipy.optimize import fsolve
import numpy as np
cimport scipy.special.cython_special
cimport numpy as np
from libc.math cimport fabs


cdef class GeneralGamma:
    cdef:
        double[:, :, :, ::1] data
        double c1
        double c2
        double c3
        double[3] solution

    def __init__(self, data, verbose=False, plot=False):
        data = np.asarray(data)
        N = len(data)  # Assuming self.N is the length of data

        # Calculate c1, c2, and c3
        self.c1 = np.log(data).sum() / N
        self.c2 = np.sum((np.log(data) - self.c1) ** 2) / N
        self.c3 = np.sum((np.log(data) - self.c1) ** 3) / N

        self.solve()


    cdef polygamma(self, int n, double x):
        # n, x = np.asarray(n), np.asarray(x)
        fac2 = (-1.0)**(n+1) * scipy.special.cython_special.gamma(n+1) * scipy.special.cython_special.zetac(x)
        return np.where(n == 0, scipy.special.cython_special.psi(x), fac2)

    cdef equations(self, double sigma, double kappa, double nu, double c1, double c2, double c3):
        cdef double eq1 = -self.c1 + np.log(sigma) + (self.polygamma(0, kappa) - np.log(kappa))/nu  
        cdef double eq2 = -self.c2 + self.polygamma(1, kappa)/nu**2
        cdef double eq3 = -self.c3 + self.polygamma(2, kappa)/nu**3

        return [eq1, eq2, eq3]

    cdef solve(self):
        cdef double[3] initial_guess = [1.0, 1.0, 1.0] # Initial guess for sigma, kappa and v
        cdef double[3] solution = fsolve(self.equations, initial_guess)
        self.solutiom = solution
        return solution

    cdef function_value(self, double z):
        cdef double sigma_hat = self.solution[0]
        cdef double kappa_hat = self.solution[1]
        cdef double nu_hat = self.solution[2]
        cdef double part1 = fabs(nu_hat) * (kappa_hat**kappa_hat)
        cdef double part2 = sigma_hat * scipy.special.cython_special.gamma(kappa_hat)
        cdef double part3 = (z/sigma_hat) ** (kappa_hat*nu_hat -1)
        cdef double part4 = np.exp(-kappa_hat*((z/sigma_hat)**nu_hat))

        return (part1/part2) * part3 * part4