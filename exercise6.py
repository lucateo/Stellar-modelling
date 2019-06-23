import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sci
import scipy.integrate as integrate

######## TO DO/ASK##########
# ASK: P_q is P_gas? lambda_0 is the central wavelength of the transition? Task 3 what are the data in .dat file? 
#      Voigt issue, it is right that u? Why seems to diverge?
# TODO: Correct the exercise 5 tau and intensity computations
# correct exercise6_alternative.py
# solve r_up - r_low issue
# find a faster implementation of Voigt
############################

# # Mine Voigt
# def integrand(x,a,u):
	# return np.exp(-x**2)/((x-u)**2 + a**2)
# def voigt_mine(a,u): # T should be the T layer
	# voigt = np.zeros(a.size)
	# for j in range(a.size):
		# voigt[j] = (a[j]/(sci.pi*np.sqrt(sci.pi)) )*integrate.quad(integrand, -np.inf, np.inf, args=(a[j], u[j]))[0]
	# return voigt
# # To make the function defined from integrate.quad a vector
# vec_voigt_mine = np.vectorize(voigt_mine)


def voigt_mine(A,U):
    VOIGT = np.zeros(A.size)
    for j in range(A.size):    
	    ZH=complex(A[j],U[j])
	    CERROR=(((((ZH*.56418958+4.6750602)*ZH+18.6465)*ZH+43.162801)*ZH+ \
	           57.90332)*ZH+37.244294)/((((((ZH+8.2863279)*ZH+33.550102)* \
	           ZH+80.645949)*ZH+118.6764)*ZH+99.929001)*ZH+37.244295)     
	    VOIGT[j]=CERROR.real
    return VOIGT



# Parameters
c = 3e10 # speed of light in cgs
h = 6.62e-27 # Planck constant in cgs
evToErg = 1.6021*10**(-12)
C = 2.07*10**(-16) # Factor for Saha equation
K_b = 1.38*10**(-16) # Boltzmann constant in cgs
e = 1.602e-19 # charge electron in Coulomb
m_e = 9.1e-28 # mass electron in g
m_Na = 22.98*1.672e-24 # mass of Na in g
E_ion = 0.754 # in eV, ionization H^-/H
sigma_th = 8*(1e-18)*3  # sigma bound free for neutral H for Balmer (n=3)
lambda_th = 8204 # threshold wavelenght (in Armstrong) for opacity bound-free of neutral H for Balmer (n=3) 
sigma_thomson = 6.6*1e-25


#Loading data
data = np.loadtxt('solspect.dat')
wavelenght, F_lambda, F2_lambda, I_lambda, I2_lambda = data.T  # trick: columns to variables
# skiprows skip the first rows (unfortunately different from entries, that's why it is 49)
data1 = np.loadtxt('falc.dat', skiprows=49)
depth, tau, mass, T, v_t, n_H, n_p, n_e, Pressure, P_gasOverP_tot, density = data1.T  # trick: columns to variables
data2 = np.loadtxt('sun.dat')
wave_plot_Na, I_plot_Na = data2.T

# Convert in cgs
wavelenght = 10**(-4) * wavelenght
F_lambda = 10**(14) * F_lambda
F2_lambda = 10**(14) * F2_lambda
I_lambda = 10**(14) * I_lambda
I2_lambda = 10**(14) * I2_lambda
depth = 10**5* depth
v_t = 10**5*v_t
wave_plot_Na = 1e-8*wave_plot_Na

# Na ionization energies (eV)
chi_1 = 5.134
chi_2 = 47.29
chi_3 = 71.64

f_NaID1 = 0.318
f_NaID2 = 0.631
U_NaII = 1
U_NaIII = 6

def U_NaI(Temperature):
	theta = 5040/Temperature
	c0 = 0.30955
	c1 = -0.17778
	c2 = 1.10594
	c3 = -2.42487
	c4 = 1.70721
	logU = c0 + c1*np.log(theta) + c2*(np.log(theta))**2 + c3*(np.log(theta))**3+ c4*(np.log(theta))**4
	return np.exp(logU)

A_Na = 1.74e-6

def delta_lambdaD(lambda_0, T, v_t):
	square_argument = 2*K_b*T/m_Na + v_t**2
	return (lambda_0/c) * np.sqrt(square_argument)


### I think lambda_0 is the central wavelenght of the transition (of the Na doublet)
lambda_0I = 589.6*10**(-7) # in cm
lambda_0II = 589*10**(-7)

def Voigt(wavelenght, T, v_t, lambda_0): # what is P_q? Maybe P_gas
	u = np.abs((wavelenght - lambda_0) /delta_lambdaD(lambda_0, T,v_t))
	P_gas = Pressure * P_gasOverP_tot # check if this is true
	# n of the excited level
	n_square = 13.6/(chi_1 - h*c/lambda_0I)
	r_up = (n_square/2) * (5*n_square + 1 -6)  # r_up = (n_square/2) * (5*n_square + 1 -6 )
	n_square_low = 13.6/(chi_1 - h*c/lambda_0II)
	r_low = 0.5* n_square_low *(5*n_square_low + 1) # n=1, l=0, it is the ground state
	#print(r_up - r_low)
	gamma_vdW = 6.33 + 0.4*np.log10(-r_up + r_low) + np.log10(P_gas) - 0.7*np.log10(T)
	gamma_vdW = np.exp(gamma_vdW)
	a = wavelenght**2 * gamma_vdW/(4*np.pi*c*delta_lambdaD(lambda_0, T,v_t))
	#print(voigt_mine(a,u))
	return voigt_mine(a,u)


######## Boltzmannn
# N_E comprises all ionization stages

# # It returns the ionization energy of a given ionization state (find_minimum(j) = ionization energy of j-1 ionization state )
def find_minimum(j):# Energies in eV
	if j==0:
		return 0
	elif j==1:
		return chi_1
	elif j==2:
		return chi_2
	elif j==3:
		return chi_3
	
# Compute partition function U_j, j=0,1,2,3,4
def U(T,j):
	if j==0:
		return U_NaI(T)
	elif j==1:
		return U_NaII
	elif j==2:
		return U_NaIII	
				
def phi_k(T,k):
	a = (C/T**(3/2))*(U(T,k)/U(T,k+1))
	E = (find_minimum(k+1) - find_minimum(k))*evToErg
	return a*np.exp(E/(K_b*T)) 

def ratioI(T):
	n_over_NI = 2/U(T,0) * np.exp(-c*h/lambda_0I)
	NI_over_N = (n_e * phi_k(T,0) *n_e* phi_k(T, 1) )/(1 + n_e * phi_k(T,0)*n_e*phi_k(T,1) + n_e*phi_k(T,1))
	return n_over_NI * NI_over_N  

def ratioII(T):
	n_over_NI = 4/U(T,0) * np.exp(-c*h/lambda_0II)
	NI_over_N = (n_e * phi_k(T,0) *n_e* phi_k(T, 1) )/(1 + n_e * phi_k(T,0)*n_e*phi_k(T,1) + n_e*phi_k(T,1))
	return n_over_NI * NI_over_N

# opacity coming from the ground state
def alphaI(wavelenght, T):
	# Define const factor in front
	const = np.sqrt(np.pi)* e**2 / (m_e * c**2)
	# Factor in the end
	end_factor = 1 - np.exp(h*c/(wavelenght*K_b*T))
	ratio_n_l = ratioI(T)
	return const * wavelenght**2 *n_H * ratio_n_l * f_NaID1* A_Na * (Voigt(wavelenght, T, v_t,lambda_0I)/delta_lambdaD(lambda_0I, T, v_t) ) * end_factor


# opacity coming from the first excited state
def alphaII(wavelenght, T):
	# Define const factor in front
	const = np.sqrt(np.pi)* e**2 / (m_e * c**2)
	# Factor in the end
	end_factor = 1 - np.exp(h*c/(wavelenght*K_b*T))
	ratio_n_l = ratioII(T)
	return const * wavelenght**2 *n_H *ratio_n_l*  f_NaID2* A_Na *(Voigt(wavelenght, T, v_t, lambda_0II)/delta_lambdaD(lambda_0II, T, v_t) ) * end_factor	




# Task 2 (just copying from exercise 5)
def planck_lambda(lambd, T):
    x = (h* c )/(K_b*T * lambd )
    return ((2*h*c**2)/lambd **5)*(1/(np.exp(x)-1))
# Opacity of H^- per neutral H atom
def opacityH_minus(wav, temp, eldens): # temp = temperature, eldend = electron density
	# wav = wavelenght
	# Computing the bound-free opacity
	wav = wav * 1e8 # converting in armstrong
	sigmabf = 0 # cross section for bound free is zero for wav > 16444, see plot from lecture
	if wav < 16444:
		sigmabf = 1.99654 -1.18267E-5*wav +2.64243E-6*wav**2 -4.40524E-10*wav**3 \
			+3.23992E-14*wav**4 -1.39568E-18*wav**5 +2.78701E-23*wav**6
		sigmabf=sigmabf*1E-18 # cm^2 per H-min ion, opacity for every H^- ion
	# conversion factor H^- to neutral H, n_(H^-)/n_H, CONTROL THIS !!!!!!!!!!!!!!!!!!!!!!!!!! (I didn't put the partition function)
	conversion = eldens * temp**(-3/2)*C * np.exp(E_ion * evToErg/(K_b*temp)) # It still gives a reasonable result, close to the one given in lecture
	sigmabf=sigmabf*conversion
		
	# Computing free free opacity
	theta=5040./temp # variable to use in polynomial to compute the opacity
	elpress=eldens*K_b*temp # electron pressure from ideal gas law
	lwav=np.log10(wav)
	f0 = -2.2763 -1.6850*lwav +0.76661*lwav**2 -0.0533464*lwav**3
	f1 = 15.2827 -9.2846*lwav +1.99381*lwav**2 -0.142631*lwav**3
	f2 = -197.789 +190.266*lwav -67.9775*lwav**2 +10.6913*lwav**3 -0.625151*lwav**4
	ltheta=np.log10(theta)
	kappaff = 1E-26*elpress*10**(f0+f1*ltheta+f2*ltheta**2) # value of H^- opacity free-free for every neutral H atom
	return kappaff + sigmabf # CHECK! 
# Since it contains an if, this is to make it work
opacityH_minus = np.vectorize(opacityH_minus)



def opacityH(wave, Temperature):
	k_nu = 0
	# Compute factor conversion n_3/n_1 (ratio level n= 3 to ground state of Hydrogen)
	ratio_n_3 = 9*np.exp(-13.6*evToErg*(1 -1/9) /(K_b*Temperature)) # the initial 9 is the ratio of degenerations
	wave = wave * 1e8 # converting in armstrong
	if wave < lambda_th:
		k_nu = ratio_n_3 * sigma_th * (wave/lambda_th)**3
	return k_nu	
opacityH = np.vectorize(opacityH)

## add the new opacities
def kappa_tot(height, temperature, n_e, n_H, n_p, wave):
	k_Hminus = opacityH_minus(wave, temperature, n_e)
	k_H = opacityH(wave, temperature)
	# it is here the multiplication with number density of H
	return (k_Hminus + k_H) * (n_H - n_p) + n_e * sigma_thomson + alphaI(wave, temperature) + alphaII(wave, temperature) 




################## again copying from exercise 5, it could be wrong!!!!
def tau(wave):
	tau = np.zeros(depth.size)
	tau[0] = ((depth[0] - depth[1])) * kappa_tot( depth, T, n_e, n_H, n_p, wave)[0]
	for i in range(1, depth.size -1):
		tau_temp = ((depth[i]- depth[i+1])) * kappa_tot( depth, T, n_e, n_H, n_p, wave)[i]
		tau[i] = tau_temp + tau[i-1]
		
	tau[depth.size-1] = ((depth[depth.size-2] - depth[depth.size-1]) * kappa_tot( depth, T, n_e, n_H, n_p, wave)[depth.size-1])
	tau[depth.size-1] = tau[depth.size-1] + tau[depth.size-2]
	return tau

def Intensity(wave, mu):
	delta_tau = np.zeros(depth.size) 
	# defining the tau intervals
	tau_int = tau(wave)
	delta_tau[0] = ((tau_int[1])) 
	for i in range(1, depth.size -1):
		delta_tau[i] = (tau_int[i+1] - tau_int[i]) 
	delta_tau[depth.size-1] = ((tau_int[depth.size-1] - tau_int[depth.size-2]))
	 
	integrand = planck_lambda(wave, T)*np.exp(-tau_int/mu) * (1/mu)
	Intensity = np.sum(integrand * delta_tau)
	return Intensity


def Flux(wave):
	flux = 0
	for i in np.arange(0.1,1.01,0.1):
		flux = flux + 0.1*Intensity(wave, i)
	#flux = flux + 0.5 * Intensity(wave,1)
	return flux


#wave_plot = np.arange(5.889e-5, 5.899e-5, 0.001e-5)
wave_plot = []
flux_plot = []
flux_data = []
for i in range(2890, 4000):
	wave_plot.append(wave_plot_Na[i])
	flux_data.append(I_plot_Na[i])
for i in range(len(wave_plot)):
	flux_plot.append( Flux(wave_plot[i]))
	if (i%100 == 0):
		print('computing... step %f' %i)
print(flux_plot)

#flux_plot = np.zeros(wave_plot_Na.size)
# for i in range(wave_plot_Na.size-8000):
	# flux_plot[i] = Flux(wave_plot_Na[i])
	# if (i%100 == 0):
		# print('computing... step %f' %i)
# print(flux_plot)


####################################################################################################

#print(Flux(5e-5))

fig = plt.figure(1)
plt.plot(wave_plot, flux_plot/np.max(flux_plot)) # plot the ratio w.r.t. maximum since variations are very tiny
plt.plot(wave_plot, flux_data)
#plt.plot(wave_plot_Na, flux_plot/np.max(flux_plot)) # plot the ratio w.r.t. maximum since variations are very tiny
#plt.plot(wave_plot_Na, I_plot_Na)
plt.legend(('Flux theoretical','Flux data'), loc='upper right')
plt.xlabel('$\lambda$ (cm)')
plt.title("Flux around Na line")
plt.show()

