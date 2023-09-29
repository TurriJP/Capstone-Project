import math
from uuid import uuid1

from scipy.optimize import fsolve
from scipy import special
import numpy as np
import matplotlib.pyplot as plt

class GeneralizedGamma():

    def __init__(self, data, verbose=False):
        self.verbose = verbose
        converted_data = np.asarray(data)
        # if self.verbose:
        #     print(converted_data)
        self.data = converted_data[converted_data != 0]
        self.N = len(self.data)
        # if self.verbose:
        print(f'Número de pixels: {self.N}')

        # Assign experimental values using 
        # the method of log cumulants
        self.c1 = sum([math.log(x) for x in self.data])/self.N
        self.c2 = sum([(math.log(x) - self.c1)**2 for x in self.data])/self.N
        self.c3 = sum([(math.log(x) - self.c1)**3 for x in self.data])/self.N

        # Solve for the unknown parameters
        self.solve()
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
        y = self.function_value(x) * 4300000

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

        return self.function_value(z)
        # if spatial_distance == 0:
        #     spatial_distance = 0.000001
        # probability = self.function_value(z)
        # s_f = 1 - np.exp(-probability)
        # s_d = 1 - np.exp(-1/spatial_distance)
        # ml_distance = w*s_f + ((1-w)*s_d)

        # return ml_distance