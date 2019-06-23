import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sci
from scipy.optimize import minimize_scalar, curve_fit

# Defining the Planckian
def planck_lambda(lambd, T):
    x = (sci.h* sci.c )/(sci.k*T * lambd )
    return ((2*sci.h*sci.c**2)/lambd **5)*(1/(np.exp(x)-1))

# Loading data
data = np.loadtxt('solspect.dat')
wavelenght, F_lambda, F2_lambda, I_lambda, I2_lambda = data.T  # trick: columns to variables

# Convert in SI
wavelenght = 10**(-6) * wavelenght
F_lambda = 10**(13) * F_lambda
F2_lambda = 10**(13) * F2_lambda
I_lambda = 10**(13) * I_lambda
I2_lambda = 10**(13) * I2_lambda


def Chisquare(arr1,arr2): #arrays as input, the first is the experimental, the second is from theory
    values = ((arr1 - arr2)**2)/arr2
    chi = np.sum(values) #returns the sum of the array
    return chi

# Temperature array
T = np.arange(5000,6500, 0.1)
fit_data = np.zeros(T.size)

# Array for the chisquare (change here the experimental value you want to verify
for i in range(T.size):
	fit_data[i] = Chisquare(F_lambda,planck_lambda(wavelenght, T[i]))

# Fit of the polynimial, polyfit(x,y, degree of polynomial), returns the coefficient from the highest degree
fit_array = np.polyfit(T,fit_data,2)

# Define the polynome and find the minimum with minimize_scalar
poly = np.poly1d(fit_array)
T_fit = minimize_scalar(poly)

print(T_fit)

# Second method
arr_check = F_lambda # the array I am checking
# The 5600 is the value the function start with
T_fit2 = curve_fit(planck_lambda, wavelenght, F_lambda, 5600  )
print(T_fit2) # it prints the value and the covariance 
    
