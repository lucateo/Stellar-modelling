import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sci
from mpl_toolkits.mplot3d import Axes3D


global C # constant in Boltzmann eq
global evToErg # to convert from ev to erg
global n_e # FREE electron density
global max_level # max level to be considered in the partition function
global e_pressure # pressure of electrons
global K_b # Boltzmann constant in cgs

# Defining parameters (in cgs)
C = 2.07*10**(-16)
evToErg = 1.6021*10**(-12)
e_pressure = 1e2
max_level = 30
K_b = 1.38*10**(-16)

# Degeneration of the i-principal level
def g_i0(i):
	return 2*i**2

# electron density with gas perfect law; for charge conservation n_e = N - N_0 (N =total density of hydrogen)
def n_e(T):
	return e_pressure/(K_b*T)

# Definition of partition function of neutral hydrogen, max level is the maximum I'm considering
def U_0(T):
	U0 = 0
	U0 = float(U0)
	for k in range(1, max_level+1, 1):
		E_k = 13.6*evToErg*(1 -1/k**2)
		U0_temp=g_i0(k)*np.exp(-E_k/(K_b*T))
		U0 = U0 + U0_temp
	return U0	

# Computes n_i0 (we put N_0 = 1) 
def n_i0(T,i):
	E_i0 = 13.6*evToErg*(1 -1/i**2)
	a = g_i0(i)* np.exp(-E_i0/(K_b*T))
	#return a/((1+n_e(T))*U_0(T)) # this is the ratio
	return a/(U_0(T)) # this is n_io without dividing by N

# Try to plot U_0
def U_0plot(level,T):
	U0 = 0
	k=1
	U0 = float(U0)
	while k <= level:
		E_k = 13.6*evToErg*(1 -1/k**2)
		U0_temp=g_i0(k)*np.exp(-E_k/(K_b*T))
		U0 = U0 + U0_temp
		k=k+1
	return U0

# Plotting U_0 with a specific max level
level=10
fig= plt.figure(1)
T_plot = np.arange(1000,20000,10)
plt.plot(T_plot, U_0plot(level, T_plot) )
plt.xlabel('T')
plt.title('U_0 for max level = %d' %level)
plt.ylabel("U_0")
#plt.show()

# Plotting first excited states
fig= plt.figure(2)
T_plot = np.arange(1000,20000,10)
plt.plot(T_plot, n_i0(T_plot,2), T_plot, n_i0(T_plot, 3) ) 
plt.legend(('$n_{20}$','$n_{30}$' ), loc='upper right')
plt.xlabel('T')
plt.ylabel('n_i0')
plt.title("population")
#plt.show()


# Second part

# Ionization energies
E_0 = 7
E_1 = 16
E_2 = 31
E_3 = 51

# Parameters
U_4= 2

# It returns the ionization energy of a given ionization state (find_minimum(j) = ionization energy of j-1 ionization state )
def find_minimum(j):# Energies in eV
	if j==0:
		return 0
	elif j==1:
		return E_0
	elif j==2:
		return E_1
	elif j==3:
		return E_2
	elif j==4:
		return E_3
	else:
		return 100 # just a value greater than E_3
	
# Compute partition function U_j, j=0,1,2,3,4
def U(T,j):
	U_j = 0 
	for k in range(find_minimum(j+1)):# I don't consider levels beyond the ionization energy
		E = k*evToErg
		U_jtemp = np.exp(-E/(K_b*T))
		U_j = U_j + U_jtemp
	return U_j	
				
def phi_k(T,k):
	a = (C/T**(3/2))*(U(T,k)/U(T,k+1))
	E = (find_minimum(k+1) - find_minimum(k))*evToErg
	return a*np.exp(E/(K_b*T)) 

# Compute N_j recursively, remember N_0 =1
def N_j(T,j):
	Nj = 1
	for k in range(j):
		Nj = Nj/(n_e(T)*phi_k(T,j))
	return Nj	 

# The total N
def N(T):
	return 1+N_j(T,1)+N_j(T,2)+N_j(T,3)+N_j(T,4)
	
# compute n_ij 
def n_ij(T,i,j):
	a = N_j(T,j)/(U(T,j))
	E_ij = (i-1)*evToErg # i=0 is the ground state that has zero energy
	return a*np.exp(-E_ij/(K_b*T))

# Plot Saha equation
fig= plt.figure(6)
T_plot = np.arange(1000,20000,10)
plt.plot(T_plot, N_j(T_plot,0)/N(T_plot),T_plot, N_j(T_plot,1)/N(T_plot),T_plot, N_j(T_plot,2)/N(T_plot),T_plot, N_j(T_plot,3)/N(T_plot)   )
plt.legend(('N_0','N_1', 'N_2','N_3')
	, loc='upper right')	
plt.xlabel('T')
plt.ylabel('n_ij')
plt.title("population")

#Plotting various levels
fig= plt.figure(3)
T_plot = np.arange(1000,20000,10)
level_plot = 1 # the energy level of each
plt.plot(T_plot, n_ij(T_plot,level_plot,0), T_plot, n_ij(T_plot,level_plot,1),T_plot, n_ij(T_plot,level_plot,2), 
	T_plot, n_ij(T_plot,level_plot,3),T_plot, n_ij(T_plot,level_plot,4) )
plt.legend(('n_%d0' %level_plot,'n_%d1'%level_plot,'n_%d2'%level_plot,'n_%d3'%level_plot, 'n_%d4' %level_plot)
	, loc='upper right')	
plt.xlabel('T')
plt.ylabel('n_ij')
plt.title("population")

fig= plt.figure(4)
T_plot = np.arange(1000,20000,10)
level_plot = 2 # the energy level of each
plt.plot(T_plot, n_ij(T_plot,level_plot,0), T_plot, n_ij(T_plot,level_plot,1),T_plot, n_ij(T_plot,level_plot,2), 
	T_plot, n_ij(T_plot,level_plot,3),T_plot, n_ij(T_plot,level_plot,4) )
plt.legend(('n_%d0' %level_plot,'n_%d1'%level_plot,'n_%d2'%level_plot,'n_%d3'%level_plot, 'n_%d4' %level_plot)
	, loc='upper right')	
plt.xlabel('T')
plt.ylabel('n_ij')
plt.title("population")
plt.show()

# plot Ca K line and H line
fig= plt.figure(5)
T_plot = np.arange(5000,6000,1)
plt.plot(T_plot, n_ij(T_plot,0,1), T_plot, n_i0(T_plot,2)* 2*(10**(6)))
plt.legend(('Ca K line', 'H Balmer alpha line'), loc='upper right')
plt.yscale('log')	
plt.xlabel('T')
plt.ylabel('n_ij')
plt.title("Ca K line and H Balmer alpha line (Normalized)")
plt.show()

