import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sci
from scipy import stats

# ALL IN cgs
K_b = 1.38*10**(-16) # Boltzmann constant in cgs

# Task 1

#Loading data
data = np.loadtxt('falc.dat')
depth, tau, mass, T, v_t, n_H, n_p, n_e, Pressure, P_gasOverP_tot, density = data.T  # trick: columns to variables
# Convert in cgs
depth = 10**5* depth
v_t = 10**5*v_t # The turbulent velocity 
# Plotting, here last temperature corresponds to the cromosphere, that's why they are higher 
fig=plt.figure(1)
plt.plot(depth,T)
plt.ylabel('T (K)')
plt.xlabel('h (cm)')
plt.title("T VS Depth")

# Task 2
fig=plt.figure(2)
plt.plot(mass,Pressure)
plt.ylabel('Pressure')
plt.xlabel('mass')
slope = stats.linregress(mass,Pressure) # it returns an array with various values, 0 is the slope
plt.title("Pressure VS mass: g = %d $cm^2/s$" %slope[0])
print(slope[0])

# Task 3
m_He_m_H = 3.97
n_He_n_H = 0.1
m_H = 1.67*10**(-24)
beta_metals = 1 - (m_H*n_H/density) * (1 + m_He_m_H*n_He_n_H)
# Plotting, the behavior is highly irregular but still varies in a small range (resolution error)
fig = plt.figure(3)
plt.plot(depth,beta_metals)
plt.ylabel(r'$\beta$') # put the r before to display in latex
plt.xlabel(r'$\tau$')
plt.title(r"$\beta$ VS $\tau$")

# Task 4, the agreement is good if you plot the ratio
P_gas = (n_H+ n_e +n_He_n_H*n_H  )* K_b*T
fig = plt.figure(4)
plt.plot(depth,(P_gas - Pressure * P_gasOverP_tot)/(P_gas + Pressure * P_gasOverP_tot))
plt.ylabel('Pressure difference')
plt.xlabel('depth')
plt.title("$P_{exp}$ - $P_{ideal}$  VS depth")

# Task 5
# I think that n_p = number of FREE protons, so if the helium is neutral, n_e - n_p should be zero (discrepancy should give the 
# ionization of the helium and other heavier elements)
# The ionization of helium seems to be different from zero for tau < 0.4
fig = plt.figure(5)
plt.plot(depth, n_e - n_p)
plt.ylabel(r'$n_e - n_p$')
plt.xlabel('depth')
plt.title(r"$n_{e}$ - $n_{p}$  VS depth")
plt.show()



