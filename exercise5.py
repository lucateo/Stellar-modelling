import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sci

######## TO DO/ASK##########
# ASK:
# Understand how to compute the opacity for H minus (check the return and the various normalization (Use 
# Saha for the fraction of hydrogen ions? How can I compute partition function for H^-? It is correct the way I did?)
# Possible errors: computation of opacities (conversion from armstrong to cm, Saha,
# Computations of integrals
#
############################

# Parameters
evToErg = 1.6021*10**(-12)
K_b = 1.38*10**(-16) # Boltzmann constant in cgs
C = 2.07*10**(-16) # Factor for Saha equation
E_ion = 0.754 # in eV, ionization H^-/H
sigma_th = 8*(1e-18)*3  # sigma bound free for neutral H for Balmer (n=3)
lambda_th = 8204 # threshold wavelenght (in Armstrong) for opacity bound-free of neutral H for Balmer (n=3) 
sigma_thomson = 6.6*1e-25
c = 3e10 # speed of light in cgs
h = 6.62e-27 # Planck constant in cgs

def planck_lambda(lambd, T):
    x = (h* c )/(K_b*T * lambd )
    return ((2*h*c**2)/lambd **5)*(1/(np.exp(x)-1))

# Task 1
# Opacity of H^- per neutral H atom
def opacityH_minus(wav, temp, eldens): # temp = temperature, eldend = electron density
	# wav = wavelenght
	# Computing the bound-free opacity
	wav_arm = wav * 1e8 # converting in armstrong
	sigmabf = 0 # cross section for bound free is zero for wav > 16444, see plot from lecture
	if wav_arm < 16444:
		sigmabf = 1.99654 -1.18267E-5*wav_arm +2.64243E-6*wav_arm**2 -4.40524E-10*wav_arm**3 \
			+3.23992E-14*wav_arm**4 -1.39568E-18*wav_arm**5 +2.78701E-23*wav_arm**6
		sigmabf=sigmabf*1E-18 # cm^2 per H-min ion, opacity for every H^- ion
	
	elpress=eldens*K_b*temp # electron pressure from ideal gas law
	theta=5040./temp # variable to use in polynomial to compute the opacity
	graysaha=4.158E-10*elpress*theta**2.5*10.**(0.754*theta) # approximation of full Saha from professor
	conversion = eldens * temp**(-3/2)*C * np.exp(E_ion * evToErg/(K_b*temp)) # It still gives a reasonable result, close to the one given in lecture
	#sigmabf=sigmabf*conversion
	sigmabf=sigmabf*graysaha
		
	# Computing free free opacity
	lwav=np.log10(wav_arm)
	f0 = -2.2763 -1.6850*lwav +0.76661*lwav**2 -0.0533464*lwav**3
	f1 = 15.2827 -9.2846*lwav +1.99381*lwav**2 -0.142631*lwav**3
	f2 = -197.789 +190.266*lwav -67.9775*lwav**2 +10.6913*lwav**3 -0.625151*lwav**4
	ltheta=np.log10(theta)
	kappaff = 1E-26*elpress*10**(f0+f1*ltheta+f2*ltheta**2) # value of H^- opacity free-free for every neutral H atom
	return kappaff + sigmabf # CHECK! 
# Since it contains an if, this is to make it work
opacityH_minus = np.vectorize(opacityH_minus)

# Task 2

def opacityH(wave, Temperature):
	k_nu = 0
	# Compute factor conversion n_3/n_1 (ratio level n= 3 to ground state of Hydrogen)
	ratio_n_3 = 9*np.exp(-13.6*evToErg*(1 -1/9) /(K_b*Temperature)) # the initial 9 is the ratio of degenerations
	wave_arm = wave * 1e8 # converting in armstrong
	if wave_arm < lambda_th:
		k_nu = ratio_n_3 * sigma_th * (wave_arm/lambda_th)**3
	return k_nu	
opacityH = np.vectorize(opacityH)

#Loading data
data = np.loadtxt('solspect.dat')
wavelenght, F_lambda, F2_lambda, I_lambda, I2_lambda = data.T  # trick: columns to variables
# skiprows skip the first rows (unfortunately different from entries, 
# that's why it is 49; readlines allows to not read the last 7 lines (actually I don't need it, I leave it here for future reference) 
#data1 = np.loadtxt(open('falc.dat','rt').readlines()[:-7], skiprows=49)
data1 = np.loadtxt('falc.dat', skiprows=49)
depth, tau, mass, T, v_t, n_H, n_p, n_e, Pressure, P_gasOverP_tot, density = data1.T  # trick: columns to variables
# Convert in cgs
wavelenght = 10**(-4) * wavelenght
F_lambda = 10**(14) * F_lambda
F2_lambda = 10**(14) * F2_lambda
I_lambda = 10**(14) * I_lambda
I2_lambda = 10**(14) * I2_lambda
depth = 10**5* depth
v_t = 10**5*v_t

# maybe the height is not directly needed here, height determines temperature, n_e and such
def kappa_tot(height, temperature, n_e, n_H, n_p, wave):
	k_Hminus = opacityH_minus(wave, temperature, n_e)
	k_H = opacityH(wave, temperature)
	# maybe is here the multiplication with number density of H, CHECK!!!!!!!!!!
	return (k_Hminus + k_H) * (n_H - n_p) + n_e * sigma_thomson 

# optical depth (where do I put the zero?)
# Here I start from the outside, i.e. I consider tau=0 the value I'm starting with (that is not the corona)
# recall that depth is decreasing, and depth.size gives 40, so to match it with the maximum index I must subtract it by 1 
def tau(wave):
	tau = np.zeros(depth.size)
	# The left and right points of the grid
	for i in range(0, depth.size-1):
		kappa_left = kappa_tot( depth[i], T[i], n_e[i], n_H[i], n_p[i], wave)
		kappa_right = kappa_tot( depth[i+1], T[i+1], n_e[i+1], n_H[i+1], n_p[i+1], wave)
		tau_temp = ((depth[i]- depth[i+1])) * 0.5* (kappa_left + kappa_right)
		tau[i] = tau_temp + tau[i-1]
	tau[depth.size-1] = tau[depth.size -2]
	return tau
print(tau(1e-5))
# To compare with the values given on the table, is a little off actually
# tau_comparison = np.zeros(depth.size)
# for i in range(depth.size):
	# tau_comparison[i] = tau(1e-5)[i] - tau[i]
# #tau_comparison = np.vectorize(tau_comparison)
# print(tau_comparison - tau)

# Assume LTE so S_lambda = B_lambda; wave is in cgs here
def Intensity(wave, mu):
	delta_tau = np.zeros(depth.size) 
	# defining the tau intervals
	tau_int = tau(wave)
	delta_tau[0] = ((tau_int[1])) 
	for i in range(1, depth.size -1):
		delta_tau[i] = (tau_int[i+1] - tau_int[i]) 
	delta_tau[depth.size-1] = ((tau_int[depth.size-1] - tau_int[depth.size-2]))
	Intensity = 0
	for i in range(1, depth.size -1):
		integrand_left = planck_lambda(wave, T[i])*np.exp(-tau_int[i]/mu) * (1/mu)
		integrand_right =planck_lambda(wave, T[i+1])*np.exp(-tau_int[i+1]/mu) * (1/mu)
		Intensity = Intensity + delta_tau[i] * 0.5*(integrand_right + integrand_left)
	return Intensity

# This is the version with lambda variable (i.e. I can pass an array of wavelenghts without worries)
def Intensity_lambda(wave, mu):
	Intensity = np.zeros(wave.size)
	delta_tau = np.zeros(depth.size) 
	for j in range(wave.size):
	# defining the tau intervals
		tau_int = tau(wave[j])
		delta_tau[0] = ((tau_int[1])) 
		for i in range(1, depth.size -1):
			delta_tau[i] = (tau_int[i+1] - tau_int[i])
		delta_tau[depth.size-1] = ((tau_int[depth.size-1] - tau_int[depth.size-2])) 
		for i in range(1, depth.size -1):
			integrand_left = planck_lambda(wave[j], T[i])*np.exp(-tau_int[i]/mu) * (1/mu)
			integrand_right =planck_lambda(wave[j], T[i+1])*np.exp(-tau_int[i+1]/mu) * (1/mu)
			Intensity[j] = Intensity[j] + delta_tau[i] * 0.5*(integrand_right + integrand_left)
	
	return Intensity

# This is the version with mu variable
def Intensity_mu(wave, mu):
	Intensity = np.zeros(mu.size)
	delta_tau = np.zeros(depth.size) 
	for j in range(mu.size):
	# defining the tau intervals
		tau_int = tau(wave)
		delta_tau[0] = ((tau_int[1])) 
		for i in range(1, depth.size -1):
			delta_tau[i] = (tau_int[i+1] - tau_int[i])
		delta_tau[depth.size-1] = ((tau_int[depth.size-1] - tau_int[depth.size-2])) 
		integrand = planck_lambda(wave, T)*np.exp(-tau_int/mu[j]) * (1/mu[j])
		for i in range(1, depth.size -1):
			integrand_left = planck_lambda(wave, T[i])*np.exp(-tau_int[i]/mu[j]) * (1/mu[j])
			integrand_right =planck_lambda(wave, T[i+1])*np.exp(-tau_int[i+1]/mu[j]) * (1/mu[j])
			Intensity[j] = Intensity[j] + delta_tau[i] * 0.5*(integrand_right + integrand_left)
	return Intensity

	
#Plotting
fig = plt.figure(1)
plt.plot(wavelenght, Intensity_lambda(wavelenght, 0.1), wavelenght,Intensity_lambda(wavelenght, 0.2), wavelenght,Intensity_lambda(wavelenght, 0.3),\
	wavelenght, Intensity_lambda(wavelenght, 0.4), wavelenght, Intensity_lambda(wavelenght, 0.5), wavelenght, Intensity_lambda(wavelenght, 0.6), \
	wavelenght, Intensity_lambda(wavelenght, 0.7),wavelenght, Intensity_lambda(wavelenght, 0.8),wavelenght, Intensity_lambda(wavelenght, 0.9),wavelenght, Intensity_lambda(wavelenght, 1)) 
plt.legend(('$\mu = 0.1 $','$\mu = 0.2 $','$\mu = 0.3 $','$\mu = 0.4 $', '$\mu = 0.5 $','$\mu = 0.6 $', 
	'$\mu = 0.7 $', '$\mu = 0.8 $', '$\mu = 0.9 $', '$\mu = 1 $'), loc='upper right')
plt.xlabel('lambda (cm)')
plt.title("Intensity")

# Task 2.5
mu_plot = np.arange(0.1, 1+0.1, 0.1)
fig = plt.figure(2)
wave1 = 1e-4 
wave2 = 3e-5
wave3 = 5e-5
plt.plot(mu_plot, (Intensity_mu(wave1, mu_plot)/Intensity(wave1, 1)),mu_plot, (Intensity_mu(wave2, mu_plot)/Intensity(wave2, 1)), \
	mu_plot, (Intensity_mu(wave3, mu_plot)/Intensity(wave3, 1)))
plt.legend(('$\lambda$ = %f cm' %wave1,'$\lambda$ = %f cm'%wave2,'$\lambda$ = %f cm'%wave3), loc='lower right')
plt.xlabel('mu')
plt.title("Intensity: limb darkening")

# Task 2.6
def Flux(wave):
	flux = 0
	for i in np.arange(0.1,1.01,0.1):
		flux = flux + 0.1*Intensity(wave, i)
	#flux = flux + 0.5 * Intensity(wave,1)
	return flux
flux_plot = np.zeros(wavelenght.size)
for i in range(wavelenght.size):
	flux_plot[i] = Flux(wavelenght[i])

fig = plt.figure(3)
plt.plot(wavelenght, flux_plot, wavelenght, F2_lambda)
plt.legend(('Flux theoretical','Flux data'), loc='upper right')
plt.xlabel('$\lambda$ (cm)')
plt.title("Flux comparisons")
plt.show()

