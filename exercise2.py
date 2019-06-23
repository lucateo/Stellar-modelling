import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sci
from mpl_toolkits.mplot3d import Axes3D
from astropy.modeling.models import Voigt1D
import scipy.integrate as integrate

# Defining the Planckian
def planck_lambda(lambd, T):
    x = (sci.h* sci.c )/(sci.k*T * lambd )
    return ((2*sci.h*sci.c**2)/lambd **5)*(1/(np.exp(x)-1))

def I_lambda(ratio, lambd, T, tau):
		return planck_lambda(lambd, T) *(1 + np.exp(-tau)*(ratio - 1))
		 
# Defining variables (all in SI)
tau = np.arange(0, 10, 0.01)
lambd = 5e-07 
T = 6000
ratio = np.arange(0.01, 10, 0.01)
I_0 = ratio * planck_lambda(lambd, T) # Typical intensity of sun at the surface 2e-16
 

# To plot in 3D
fig = plt.figure(1)
tau, ratio = np.meshgrid(tau, ratio) #to make the 3d plot work
z = I_lambda(ratio, lambd, T, tau)
ax = fig.gca(projection='3d')
ax.plot_surface(tau, ratio , z)
ax.set_xlabel('tau')
ax.set_ylabel('ratio')
ax.set_zlabel('I')
plt.show()
# For big tau it tend to the value of B_lambda, for ratio < 1 it grows, for ratio > 1 i decays, for r=1 it is constant


# Second part

# Implementation of Voigt from astropy
# v1 = Voigt1D(x_0=0, amplitude_L=10, fwhm_L=0.5, fwhm_G=1)

# Mine Voigt
def integrand(x,a,u):
	return np.exp(-x**2)/((x-u)**2 + a**2)

# Normalization factors
m_iron = 54* 1.66*10**(-27)
normaliz_factor= (1/sci.c)* np.sqrt(2*sci.k/m_iron)	

def voigt_mine(a,u,T): # T should be the T layer
	# omit the delta_lambdaD, it is a scale factor
	return (a/(sci.pi*np.sqrt(sci.pi)) )*integrate.quad(integrand, -np.inf, np.inf, args=(a, u))[0]

# To make the function defined from integrate.quad a vector
vec_voigt_mine = np.vectorize(voigt_mine)

# Plot 2D Voigt vs u
u_plot = np.arange(-10,10,0.01)
plt.plot(u_plot,vec_voigt_mine(0.5,u_plot,5000))
plt.title('Voigt vs u')
plt.show()

# 3D plot Voigt vs u and a
fig = plt.figure(2)
u = np.arange(-5, 5, 0.1)
a = np.arange(0.01, 0.5, 0.01)
a, u = np.meshgrid(a, u) #to make the 3d plot work
z = vec_voigt_mine(a,u,5000)
ax = fig.gca(projection='3d')
ax.plot_surface(u, a , z)
ax.set_xlabel('u')
ax.set_ylabel('a')
ax.set_zlabel('Voigt')
plt.show()


def emergent_intensity(T_surf, T_layer, a, lambd, tau_0, u):
	tau_lambda = tau_0*vec_voigt_mine(a, u,T_layer)
	return planck_lambda(lambd, T_surf) * np.exp(-tau_lambda) + planck_lambda(lambd, T_layer)*(1 -np.exp(-tau_lambda)) 

# Fixed parameters
T_layer = 4200
T_surf = 5700
wavelenght = 5.7e-07
tau_0fixed = 1
a_fixed = 0.1
u=0

# Variable parameters (for plotting)
tau_0 = np.arange(0.1, 10, 0.1)
a = np.arange(0.01, 0.5, 0.01)
T_layerVariable = np.arange(2000, 6000, 1)
wave_variable = np.arange(1e-7, 1e-5, 1e-8)

# Computing the specific value
print('emergent intensity: ')
print(emergent_intensity(T_surf, T_layer, a_fixed, wavelenght, tau_0fixed,u))

# Changing u
fig = plt.figure(7)
# various wavelenghts
a1=emergent_intensity(T_surf, T_layer, a_fixed, 1e-7, tau_0fixed,u_plot)
a2=emergent_intensity(T_surf, T_layer, a_fixed, 3e-7, tau_0fixed,u_plot)
a3=emergent_intensity(T_surf, T_layer, a_fixed, 5e-7, tau_0fixed,u_plot)
a4=emergent_intensity(T_surf, T_layer, a_fixed, 6e-7, tau_0fixed,u_plot)
a5=emergent_intensity(T_surf, T_layer, a_fixed, 8e-7, tau_0fixed,u_plot)
plt.plot(u_plot, a1, u_plot, a2,u_plot, a3,u_plot, a4,u_plot, a5)
plt.xlabel('u')
plt.ylabel('I')
plt.title("I lambda vs u")
plt.legend(('100','300','500','600', '800'), loc='upper right', title='wavelenghts (nm)')

# Changing tau_0
fig = plt.figure(3)
plt.plot(tau_0, emergent_intensity(T_surf, T_layer, a_fixed, wavelenght, tau_0,u))
plt.xlabel('tau_0')
plt.ylabel('I')
plt.title("I lambda vs tau_0")

# Changing central wavelenght
fig = plt.figure(4)
plt.plot(wave_variable, emergent_intensity(T_surf, T_layer, a_fixed, wave_variable, tau_0fixed,u))
plt.xlabel('Central lambda')
plt.ylabel('I')
plt.title("I lambda vs Central lambda")

# Changing T_layer
fig = plt.figure(5)
plt.plot(T_layerVariable, emergent_intensity(T_surf, T_layerVariable, a_fixed, wavelenght, tau_0fixed,u))
plt.xlabel('T layer')
plt.ylabel('I')
plt.title("I lambda vs T layer")

plt.show()

# Facultative part
# Idea: normalize the curve with the maximum value it gets (it suffice to divide by the intensity at u=10), then
# to compute the area "above the curve", divide by one (that now is the new maximum value), integrate and then 
# multiply by -1
# the equivalent widht is the area of a rectangle with height normalized to 1 (so de facto is the lenght of it)
# that is equal to the area "above" the curve (that is like a hole) 
fig = plt.figure(8)
# The normalized one
def normalized_int(u,tau_0):
	a1=emergent_intensity(T_surf, T_layer, a_fixed, 1e-7, tau_0,u)
	normalized = a1/emergent_intensity(T_surf, T_layer, a_fixed, 1e-7, tau_0,10)
	return normalized - 1

# The equivalent width	
def equiv_width(tau_0):
	return	 -1*integrate.quad(normalized_int, -10, 10, args=(tau_0))[0]

# To make the function defined from integrate.quad a vector
vec_equiv_width = np.vectorize(equiv_width)

# Plotting
plt.plot(tau_0, vec_equiv_width(tau_0))
plt.ylabel('equivalent width')
plt.xlabel('tau_0')
plt.title("equivalent width vs tau_0")

plt.show()




