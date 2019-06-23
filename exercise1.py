import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sci


# Graphical solution

def MaximumT(t): #Maximum
	return t - 3*(1 - np.exp(-t))

def Wien(t):#Wien
	return t - 5*(1 - np.exp(-t))	

def null(t):
	return 0*t


def plot_intersect(f1,f2,acc): # acc is accuracy
    t1 = np.arange(-0.5, 5.0, acc)
    plt.plot(t1, f1(t1), 'r--')
    plt.plot(t1, f2(t1),'b--' )
    #make the array of values with the corresponding function
    arr_f1= f1(t1)
    arr_f2 = f2(t1)
    #This computes the index of intersections, i.e. idx will be an array corresponding to the
    #indices of the intersection (argwhere gives the indices, diff tells when it changes
    #sign, sign takes the sign
    #Since I cannot subtract functions, I have to pass to sign function the array values
    idx = np.argwhere(np.diff(np.sign(arr_f1 - arr_f2))).flatten()
    plt.plot(t1[idx], arr_f1[idx], 'ro')
    print ("Graphical result: ", t1[idx])
    plt.show()


#Iterative solution
# Important to put the elif
def find_zero(f, t_min, t_max):
	tolerance = 0.0000000001
	if f(t_min) > 0 + tolerance and f(t_max) < 0 - tolerance:
		t_new = 0.5*(t_max + t_min)
		if f(t_new) > 0 + tolerance:
			find_zero(f, t_new, t_max)
		elif f(t_new) < 0 - tolerance:
			find_zero(f, t_min, t_new)
		else:
			print("the result is: ", t_new)
	elif f(t_min) < 0 - tolerance and f(t_max) > 0 + tolerance:
		t_new = 0.5*(t_max + t_min)
		if f(t_new) > 0 + tolerance:
			find_zero(f, t_min, t_new)
		elif f(t_new) < 0 - tolerance:
			find_zero(f, t_new, t_max)
		else:
			print("the result is: ", t_new)
	elif 0 + tolerance > f(t_min) > 0-tolerance :
		print("the result is: ", t_min)
	else:
		print("the result is: ", t_max)	
				

# Don't put a too high accuracy! Otherwise it will block
#Uncomment this to show the first part of the exercise
#plot_intersect(MaximumT, null, 0.01)
#plot_intersect(Wien, null, 0.01)

print("Maximum of Temperature")
find_zero(MaximumT, 2.5,3.)
print("Wien law")
find_zero(Wien, 4.5, 5)


# Second exercise

#sci.h is the planck constant in SI, sci.k Boltzmann constant in SI
def planck_nu(nu, T): # planck with frequency
    x = (sci.h* nu)/(sci.k*T)
    return ((2*sci.h*nu**3)/sci.c**2)*(1/(np.exp(x)-1))

def planck_lambda(lambd, T):
    x = (sci.h* sci.c )/(sci.k*T * lambd )
    return ((2*sci.h*sci.c**2)/lambd **5)*(1/(np.exp(x)-1))

#Loading data
data = np.loadtxt('solspect.dat')
wavelenght, F_lambda, F2_lambda, I_lambda, I2_lambda = data.T  # trick: columns to variables

#convert in SI
wavelenght = 10**(-6) * wavelenght
F_lambda = 10**(13) * F_lambda
F2_lambda = 10**(13) * F2_lambda
I_lambda = 10**(13) * I_lambda
I2_lambda = 10**(13) * I2_lambda


def Chisquare(arr1,arr2): #arrays as input, the first is the experimental, the second is from theory
    values = ((arr1 - arr2)**2)/arr2
    chi = np.sum(values) #returns the sum of the array
    return chi

#loop to find the minimum chi square value
def findMinimum(start,end,step, arr_obs, wave): #in input the wavelenght array
    T = np.arange(start,end,step)
    chi = Chisquare(arr_obs, planck_lambda(wave,T[0]) )
    for i in range(T.size):
        chi_provv = Chisquare(arr_obs, planck_lambda(wave,T[i]) )
        if chi_provv < chi:
            chi = chi_provv
            Temperature = T[i]
    return Temperature


# compute the various temperatures corresponding to the distributions
# the 2 variables should give higher values than the 1 ones, the 1 makes an average over
# the wavelenght bins, the 2 takes the higher (or something like that), also the flux should
# have smaller values
Temperature_I = findMinimum(5500,6500,0.1, I_lambda, wavelenght) #5981
Temperature_I2 = findMinimum(5500,6500,0.1, I2_lambda, wavelenght) #6296
Temperature_F = findMinimum(5500,6500,0.1, F_lambda, wavelenght) #5688
Temperature_F2 = findMinimum(5500,6500,0.1, F2_lambda, wavelenght) #5970

#The type of temperature to plot, change this for different results
Temperature_plot = Temperature_F
print("Temperature obtained by chi square test: ", Temperature_plot)

#Plotting
plt.plot(wavelenght, F_lambda, wavelenght,F2_lambda, wavelenght,I_lambda,wavelenght,I2_lambda,wavelenght,planck_lambda(wavelenght,Temperature_plot))
plt.legend(('F','F prime','I','I prime', 'theory'), loc='upper right')
plt.xlabel('lambda')
plt.title("Solar Spectrum")
plt.show()












	
	
