import math
from uuid import uuid1

from scipy.optimize import fsolve
from scipy import special
import numpy as np
import matplotlib.pyplot as plt

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

class GeneralizedGamma():

    def __init__(self, data, verbose=False, plot=False):
        self.verbose = verbose
        converted_data = np.asarray(data)
        # if self.verbose:
        #     print(converted_data)
        self.data = converted_data[converted_data != 0]
        self.N = len(self.data)
        if self.verbose:
            print(f'Número de pixels: {self.N}')

        # Assign experimental values using 
        # the method of log cumulants
        self.c1 = sum([math.log(x) for x in self.data])/self.N
        self.c2 = sum([(math.log(x) - self.c1)**2 for x in self.data])/self.N
        self.c3 = sum([(math.log(x) - self.c1)**3 for x in self.data])/self.N

        # Solve for the unknown parameters
        self.solve()
        if plot:
            self.plot()

    def equations(self, vars):
        # Define the system of equations
        sigma, kappa, v = vars
        eq1 = -self.c1 + math.log(sigma) + (special.polygamma(0, kappa) - math.log(kappa))/v  
        eq2 = -self.c2 + special.polygamma(1, kappa)/v**2
        eq3 = -self.c3 + special.polygamma(2, kappa)/v**3
        return [eq1, eq2, eq3]
    
    def solve(self):
        initial_guess = [1.0, 1.0, 1.0] # Initial guess for sigma, kappa and v
        solution = fsolve(self.equations, initial_guess)
        self.sigma_hat, self.kappa_hat, self.v_hat = solution

        if self.verbose:
            print(f"sigma = {self.sigma_hat}")
            print(f"kappa = {self.kappa_hat}")
            print(f"v = {self.v_hat}")

    def function_value(self, z):
        part1 = abs(self.v_hat) * (self.kappa_hat**self.kappa_hat)
        part2 = self.sigma_hat * special.gamma(self.kappa_hat)
        part3 = (z/self.sigma_hat) ** (self.kappa_hat*self.v_hat -1)
        part4 = np.exp(-self.kappa_hat*((z/self.sigma_hat)**self.v_hat))

        return (part1/part2) * part3 * part4
    
    def plot(self):
        x = np.linspace(1, 255, 300)
        y = self.function_value(x) * 4300

        plt.figure(figsize=(8, 6))  # Optional: Set the figure size

        plt.hist(self.data, bins=300)
        plt.plot(x, y, label='General gamma function', color='blue')  # Plot the function
        plt.xlabel('x')  # Label for the x-axis
        plt.ylabel('y')  # Label for the y-axis
        plt.title('Plot of f(x)')  # Title of the plot
        plt.legend()  # Display the legend
        plt.grid(True)  # Enable gridlines

        name = str(uuid1())
        plt.savefig('export/'+name+'.png', dpi=300, bbox_inches='tight') 
    
    def likelihood_distance(self, z, w, spatial_distance):

        # return self.function_value(z)
        # if spatial_distance == 0:
        #     spatial_distance = 0.000001
        probability = self.function_value(z)
        s_f = 1 - np.exp(-probability)
        return s_f
        # s_d = 1 - np.exp(-1/spatial_distance)
        # ml_distance = w*s_f + ((1-w)*s_d)

        # return ml_distance