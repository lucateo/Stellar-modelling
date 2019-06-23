import numpy as np
import matplotlib.pyplot as plt

# physics cgs
k=1.380658e-16         # Boltzmann constant (erg K# double precision)
h=6.626076e-27         # Planck constant (erg s)
c=2.997929e10          # speed of light (cm/s)

pieemc = 0.026537
mel    = 9.1094e-28        # elecron mass
mNa    = 21.99*1.6066e-24  # mass Na


def PLANCK(T,wavcm):
    # computes Planck function in erg / (cm2 sec [delta lambda=1 micron] ster)
    # input: temp = temperature (K)
    # wav = wavelength in cm

    B_lambda = 2*h*c**2/(wavcm**5*(np.exp(h*c/(wavcm*k*T))-1))
   
    return B_lambda

def exthmin(wav,temp,eldens):
    # in:  wav = wavelength [Angstrom] (float or fltarr)
    #      temp = temperature [K]
    #      eldens = electron density [electrons cm-3]
    # out: H-minus bf+ff extinction [cm**2 per neutral hydrogen atom]
    #      assuming LTE ionization H/H-min


    theta=5040./temp
    elpress=eldens*k*temp

    # evaluate H-min bound-free per H-min ion = Gray (8.11)
    # his alpha = my sigma in NGSB/AFYC (per particle without stimulated)
    sigmabf = 1.99654 -1.18267E-5*wav +2.64243E-6*wav**2 -4.40524E-10*wav**3 \
             +3.23992E-14*wav**4 -1.39568E-18*wav**5 +2.78701E-23*wav**6
    sigmabf=sigmabf*1E-18  # cm**2 per H-min ion

    if wav > 16444:
        sigmabf=0.  


    # convert into bound-free per neutral H atom assuming Saha = Gray p135
    # units: cm2 per neutral H atom in whatever level (whole stage)
    # I use an approximation formular instead of the full saha equation
    graysaha=4.158E-10*elpress*theta**2.5*10.**(0.754*theta)    # Gray (8.12)

    kappabf=sigmabf*graysaha                            # per neutral H atom
    kappabf=kappabf*(1.-np.exp(-h*c/(wav*1E-8*k*temp))) # correct stimulated; not necessay


    # evaluate H-min free-free including stimulated emission = Gray p136
    lwav= np.log10(wav)
    f0  =  -2.2763 -1.6850*lwav +0.76661*lwav**2 -0.0533464*lwav**3
    f1  =  15.2827 -9.2846*lwav +1.99381*lwav**2 -0.142631*lwav**3
    f2  = -197.789 +190.266*lwav -67.9775*lwav**2 +10.6913*lwav**3 -0.625151*lwav**4
    ltheta  = np.log10(theta)
    kappaff = 1E-26*elpress*10**(f0+f1*ltheta+f2*ltheta**2)   # Gray (8.13)

    return kappabf+kappaff

def partition(T):
    k= 0.00008617385
    U = 0
    for i in range(1,40):
        gi = 2*i**2
        ei = 13.6*(1.-1./i**2)
        U = U + gi*np.exp(-ei/(k*T))
        
    return U

def exth(nh,temp,wave):
    # boubd-free opacity for Balmer continuum of hydrogen
    # in: nh  : density of NEUTRAL hydrogen
    #     temp: temperature
    #     wave: wavelength in angstroem
    # out:      opacity in cm**(-1) assuming LTE for the population of the
    #           n=2 state
    a0 = 1.58e-17   # Kantenquerschnitt
    UH = partition(temp)      # Partition Function HI
    En2 = h*2.466e15# Anregungsenergie n=2
    gn2 = 8.        # Statistisches Gewicht n=2 
    nu0 = 8.22e14   # Kantenfrequenz

    nu  = 1.e8*c/wave
    exthn2 = a0*(nu0/nu)**3*nh*gn2/UH*np.exp(-En2/(k*temp))
    if wave > c*1.e8/nu0:
        exthn2=0.  
    

    return exthn2

# read solar spectrum
sols = np.genfromtxt('solspect.dat',dtype='float',names=['wav','F','Fp','I','Ip'])
wave = sols['wav']
F    = sols['F']*1e10
Fp   = sols['Fp']*1e10
I    = sols['I']*1e10
Ip   = sols['Ip']*1e10
nw   = len(wave)

# read FALC model
falc = np.genfromtxt('falc.dat',dtype='float',names=['h','tau','m','T','vt','nH','np','ne','Ptot','Pgtot','rho'])
high = falc['h'][40:]
temp = falc['T'][40:]
nhyd = falc['nH'][40:]
nprot= falc['np'][40:]
nel  = falc['ne'][40:]
pg   = falc['Ptot'][40:]*falc['Pgtot'][40:]
vturb= falc['vt'][40:]
nh   = len(high)

def Voigt(A,U):
    VOIGT = []
    ZH=complex(A,U)
    CERROR=(((((ZH*.56418958+4.6750602)*ZH+18.6465)*ZH+43.162801)*ZH+ \
           57.90332)*ZH+37.244294)/((((((ZH+8.2863279)*ZH+33.550102)* \
           ZH+80.645949)*ZH+118.6764)*ZH+99.929001)*ZH+37.244295)     
    VOIGT=np.append(VOIGT,CERROR.real)
    return VOIGT

def extnad (temp,nel,nhyd,pg,vturb,wave,gamenhanc):
    # wave in Aangstroem

    kt       = k*temp
    sahafakt = (2.*np.pi*mel*kt/(h*h))**1.5

    wave = wave*1.e-8

    f1 = 0.318
    f2 = 0.631
    lam1 = 5895.94*1.e-8
    lam2 = 5889.97*1.e-8
    E1   = h*c/lam1
    E2   = h*c/lam2
    g_l= 2.
    Ae = 1.8e-6
    Ryd = 2.18e-11
    Z=1.
    chi1 = 41449.44*h*c
    chi2 = 381395.*h*c
    l_l  = 0.
    l_u1 = 1.
    l_u2 = 1.

    # partition function Na I
    ltheta = np.log10(5040./temp)
    c0 =  0.30955
    c1 = -1.7778
    c2 =  1.10594
    c3 = -2.42847
    c4 =  1.70721
    exponent = c0 + ltheta*(c1 + ltheta*(c2 + ltheta*(c3  + ltheta*c4))) 
    UI   = 10.**exponent
    UII  = 1.
    UIII = 6.

    # saha
    saha12 = 2.*UII /(nel*UI) *sahafakt*np.exp(-chi1/kt)
    saha23 = 2.*UIII/(nel*UII)*sahafakt*np.exp(-chi2/kt)

    # boltzmann
    NI  =  1./(1.+saha12+saha12*saha23)
    n_l = NI * g_l/UI

    # dopplerwidth
    delt_lam1 = lam1/c*np.sqrt(2.*kt/mNa + vturb*vturb)
    delt_lam2 = lam2/c*np.sqrt(2.*kt/mNa + vturb*vturb)

    # n_eff
    neff1sq = Ryd*Z*Z/(chi1-E1)
    neff2sq = Ryd*Z*Z/(chi1-E2)

    #atomic radii
    rlsq  = .5*neff1sq/(Z*Z)*(5.*neff1sq + 1. -3*l_l*(l_l+1.))
    ru1sq = .5*neff2sq/(Z*Z)*(5.*neff2sq + 1. -3*l_u1*(l_u1+1.))
    ru2sq = .5*neff2sq/(Z*Z)*(5.*neff2sq + 1. -3*l_u2*(l_u2+1.))

    # voigt parameter
    u1 = abs((wave - lam1 ) / delt_lam1)
    u2 = abs((wave - lam2 ) / delt_lam2)

    exponent = 6.33 + 0.4*np.log10(-ru1sq+rlsq) +np.log10(pg) -0.7*np.log10(temp)
    gam1 = 10.**exponent

    exponent = 6.33 + 0.4*np.log10(-ru2sq+rlsq) +np.log10(pg) -0.7*np.log10(temp)
    gam2 = 10.**exponent

    a1 = wave*wave*gam1/(4.*np.pi*c*delt_lam1)*gamenhanc
    a2 = wave*wave*gam2/(4.*np.pi*c*delt_lam2)*gamenhanc
    
    #extinction
    extnad1 = pieemc*wave*wave/c * n_l*nhyd*Ae * f1 * Voigt(a1,u1) *(1.-np.exp(-h*c/(wave*kt)))/delt_lam1/np.sqrt(np.pi)
    extnad2 = pieemc*wave*wave/c * n_l*nhyd*Ae * f2 * Voigt(a2,u2) *(1.-np.exp(-h*c/(wave*kt)))/delt_lam2/np.sqrt(np.pi)

    return extnad1+extnad2

def extinction(wave,temp,nel,nhyd,nprot,pg,vturb):
    extinction = exthmin(wave,temp,nel)*(nhyd-nprot) \
                 + 0.664E-24*nel + exth(nhyd-nprot,temp,wave) \
                 + extnad(temp,nel,nhyd,pg,vturb,wave,2)
    return extinction

def RadTrans():
    intenslammu =[]
    for im in range(nm):
        intenslam =[]
        for iw in range(nw):
            if iw%100 == 0: print("working at wavelength {0}".format(wave[iw]*1e4))
            tau1   = 0.
            intens = 0.
            for ih in range(nh-1):
                ext1 = extinction(wave[iw]*1e4,temp[ih],nel[ih],nhyd[ih],nprot[ih],pg[ih],vturb[ih])
                ext2 = extinction(wave[iw]*1e4,temp[ih+1],nel[ih+1],nhyd[ih+1],nprot[ih+1],pg[ih+1],vturb[ih+1])
                tau2 = tau1 + 0.5*(ext1+ext2) * (high[ih]-high[ih+1]) * 1E5
                wavcm    = wave[iw]*1E-4        # change wav into cm
                blambda1 = PLANCK(temp[ih]  ,wavcm)*1E-4  # change B_lambda into per micron
                blambda2 = PLANCK(temp[ih+1],wavcm)*1E-4  # change B_lambda into per micron
                integrand1 = blambda1 * np.exp(-tau1/mu[im])
                integrand2 = blambda2 * np.exp(-tau2/mu[im])
                if iw == 0 and im == 9: 
                    print(wave[iw],high[ih],blambda1*1e4,blambda2*1e4)
                    #print(tau1,tau2)
                intens     = intens + 0.5*(integrand2+integrand1)*(tau2-tau1)/mu[im]                
                tau1       = tau2

            intenslam = np.append(intenslam,intens)
    
        intenslammu = np.append(intenslammu,intenslam)
    
    intenslammu = intenslammu.reshape(nm,nw)
    return intenslammu


sun = np.genfromtxt('sun.dat',dtype='float',names=['wav','I'])

wave = sun['wav']*1e-4  # also in mu meter
I    = sun['I']

ind = np.where(wave*1e3 > 588.5)
wave = wave[ind]
I    = I[ind]
ind = np.where(wave*1e3 < 590)
wave = wave[ind]
I    = I[ind]


print("solving radiative transfer from {0} nm to {1} nm".format(np.min(wave*1.e3),np.max(wave*1.e3)))

nw   = len(wave)
nm   = 1
mu   = [1]

intenslammu = RadTrans()
intenslam   = intenslammu[-1,:]

fig = plt.figure()
fig.set_size_inches(8, 5)
plt.plot(wave,intenslam/np.max(intenslam),label='FALC model')
plt.plot(wave,I,label='Observation')
plt.xlabel('wavelength [$\mu$m]')
plt.ylabel('specific Intensity I$_\lambda$')
plt.legend()
plt.show()

