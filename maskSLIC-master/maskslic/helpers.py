import math
from scipy.optimize import fsolve
from scipy import special
import numpy as np

class GeneralizedGamma():

    def __init__(self, data):
        self.data = data[data != 0]
        self.N = len(self.data)

        # Assign experimental values using 
        # the method of log cumulants
        self.c1 = sum([math.log(x) for x in self.data])/self.N
        self.c2 = sum([(math.log(x) - self.c1)**2 for x in self.data])/self.N
        self.c3 = sum([(math.log(x) - self.c1)**3 for x in self.data])/self.N

        # Solve for the unknown parameters
        self.solve()

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

        print(f"sigma = {self.sigma_hat}")
        print(f"kappa = {self.kappa_hat}")
        print(f"v = {self.v_hat}")

    def function_value(self, z):
        part1 = abs(self.v_hat) * (self.kappa_hat**self.kappa_hat)
        part2 = self.sigma_hat * special.gamma(self.kappa_hat)
        part3 = (z/self.sigma_hat) ** (self.kappa_hat*self.v_hat -1)
        part4 = np.exp(-self.kappa_hat*((z/self.sigma_hat)**self.v_hat))

        return (part1/part2) * part3 * part4