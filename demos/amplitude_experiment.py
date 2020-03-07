import sys
import os
import os.path
import tempfile
import scipy 
import numpy as np
from matplotlib import pyplot as pl

from crackclosuresim2 import inverse_closure
from crackclosuresim2 import crackopening_from_tensile_closure

from crackclosuresim2 import ModeI_throughcrack_CODformula
from crackclosuresim2 import Tada_ModeI_CircularCrack_along_midline
from crackclosuresim2 import ModeII_throughcrack_CSDformula
from crackclosuresim2.fabrikant import Fabrikant_ModeII_CircularCrack_along_midline


from function_as_script import scriptify

from angled_friction_model.angled_friction_model import angled_friction_model
#from angled_friction_model.angled_friction_model import angled_friction_model as angled_friction_model_function
#angled_friction_model = scriptify(angled_friction_model_function)


doplots=False  # extra plots from angled friction model
verbose=False  # More verbose prints from angled fiction model



friction_coefficient=0.3

vibration_frequency=20e3  # (Hz)

static_load=60e6  # tensile static load of 60MPa
# assume also that there is no synergy between heating from different modes. 

# Standard deviation representing crack surface tortuosity
angular_stddev = 28*np.pi/180.0

numdraws=20 # Number of draws from crack tortuosity  per step


# material parameters
# Youngs modulus, Poisson's ratio
E = 207.83e9  # Measured number from UTCB specimen set (Pa)
nu = 0.294 # Measured number from UTCB specimen set
sigma_yield=1182e6 # Material certification from UTCB specimen set
tau_yield=sigma_yield/2.0
G=E/(2*(1+nu))

# Soft closure model parameter:
msqrtR = 1000.0e6 * np.sqrt(15e-6) # asperity density (asperities/m^2) * sqrt(contact radius) (sqrt(m))

# Select crack models for normal and shear loading

#crack_model_normal = ModeI throughcrack_CODformula(E)
crack_model_normal = Tada_ModeI_CircularCrack_along_midline(E,nu)
#crack_model_shear = ModeII_throughcrack_CSDformula(E,nu)
crack_model_shear = Fabrikant_ModeII_CircularCrack_along_midline(E,nu)


xmax = 2e-3  # Maximum position from center to calculate to;
             # should exceed half-crack lengths 

# Desired approximate step size for calculations
approximate_xstep=25e-6 # 25um

# Define actual step size and the range of positions over
# which we will calculate
num_boundary_steps=int((xmax)//approximate_xstep)
numsteps = num_boundary_steps-1
xstep = (xmax)/(numsteps) # Actual step size so that xmax is a perfect multiple of this number


x_bnd = xstep*np.arange(num_boundary_steps) # Position of element boundaries
xrange = (x_bnd[1:] + x_bnd[:-1])/2.0 # Position of element centers


# Here we evaluate crack closure state from a series of observed
# effective crack lengths.

#   (alternatively we could define closure_stress_leftside, aleft,
#   closure_stress_rightside, and aright directly)


# half-crack lengths for the right-hand side (meters)
reff_rightside=np.array([ .5e-3, .7e-3, .9e-3, 1.05e-3, 1.2e-3, 1.33e-3, 1.45e-3, 1.56e-3, 1.66e-3],dtype='d')

# Corresponding opening stresses, units of Pa
#seff_rightside=np.array([ -150e6, 50e6, 100e6, 150e6, 200e6, 250e6, 300e6, 350e6, 400e6],dtype='d')
seff_rightside=np.array([ -50e6, 20e6, 40e6, 60e6, 70e6, 80e6, 90e6, 115e6, 130e6],dtype='d')

# half-crack lengths for the left-hand side (meters)
reff_leftside=np.array([ .5e-3, .7e-3, .9e-3, 1.05e-3, 1.2e-3, 1.33e-3, 1.45e-3, 1.56e-3, 1.66e-3],dtype='d')

# Corresponding opening stresses, units of Pa
seff_leftside=np.array([ -150e6, 50e6, 100e6, 150e6, 200e6, 250e6, 300e6, 350e6, 400e6],dtype='d')



# Fully open crack lengths for left and right side
aleft=np.max(reff_leftside) 
aright=np.max(reff_rightside)

assert(aleft < xmax) # Increase xmax if either of these assertions fail
assert(aright < xmax)


# Determine closure stress field from observed crack length data
closure_stress_leftside=inverse_closure(reff_leftside,seff_leftside,xrange,x_bnd,xstep,aleft,sigma_yield,crack_model_normal,verbose=verbose)
closure_stress_rightside=inverse_closure(reff_rightside,seff_rightside,xrange,x_bnd,xstep,aright,sigma_yield,crack_model_normal,verbose=verbose)


# Evaluate initial crack opening gaps from extrapolated tensile closure field
crack_initial_opening_leftside = crackopening_from_tensile_closure(xrange,x_bnd,closure_stress_leftside,xstep,aleft,sigma_yield,crack_model_normal)

crack_initial_opening_rightside = crackopening_from_tensile_closure(xrange,x_bnd,closure_stress_rightside,xstep,aright,sigma_yield,crack_model_normal)


# Range of vibration amplitudes
vib_ampls = np.arange(0.0,70.0e6,5.0e6)

# Array to store total calculated heating
total_heating_right=np.zeros(vib_ampls.shape,dtype='d')



for ampl_idx in range(vib_ampls.shape[0]):  # vibrational normal stress amplitude. 
    vib_normal_stress_ampl=vib_ampls[ampl_idx]
    vib_shear_stress_ampl = 0.0 # assume 0 shear
    #vib_shear_stress_ampl = vib_normal_stress_ampl  # Assume shear amplitude same as normal stress
    
    (power_per_m2_right,
     power_per_m2_mean_stddev_right,
     vibration_ampl_right,
     shear_vibration_ampl_right) = angled_friction_model(x_bnd,xrange,xstep,
                                                         numdraws,
                                                         E,nu,
                                                         sigma_yield,tau_yield,
                                                         friction_coefficient,
                                                         closure_stress_rightside,
                                                         crack_initial_opening_rightside,
                                                         angular_stddev, # beta_drawfunc,
                                                         aright,
                                                         static_load,
                                                         vib_normal_stress_ampl,
                                                         vib_shear_stress_ampl,
                                                         vibration_frequency,
                                                         crack_model_normal,
                                                         crack_model_shear,
                                                         1.0,
                                                         msqrtR,
                                                         "quarterpenny",
                                                         None,
                                                         verbose,
                                                         doplots)


    #if ampl_idx==1:
    #    raise ValueError("break!")
    total_heating_right[ampl_idx] = np.sum(power_per_m2_right*xrange*xstep*np.pi/2)  # integrate heating of half-semicircles (integrate power *rdrdtheta)
    pass

# linear + quadratic curve fit
# [ vib_ampls vib_ampls^2 ][ c1 ]  = [ total_heating_right ]
#                          [ c2 ]
Amat = np.array((vib_ampls,vib_ampls**2.0)).T
(linear_coeff, quadratic_coeff) = np.inner(np.dot(np.linalg.inv(np.dot(Amat.T,Amat)),Amat.T),total_heating_right)

# power law fit
init_idx = vib_ampls.shape[0]//2
(powerlaw_coeffs,cov_x) = scipy.optimize.leastsq(lambda p: total_heating_right-p[0]*vib_ampls**p[1],(total_heating_right[init_idx]/vib_ampls[init_idx]**1.5,1.5))

pl.figure()
pl.clf()
pl.plot(vib_ampls/1e6,total_heating_right*1e3,'-',
        #vib_ampls/1e6,(vib_ampls*linear_coeff + (vib_ampls**2.0)*quadratic_coeff)*1e3,'-',
        vib_ampls/1e6,(powerlaw_coeffs[0]*vib_ampls**powerlaw_coeffs[1])*1e3,'-')
pl.grid()
pl.xlabel('Vibration amplitude (MPa)')
pl.ylabel('Heating power (mW)')
pl.legend(('Calculated power',
           #'Linear+quadratic fit',
           'Power law power=%f' % (powerlaw_coeffs[1])),loc="best")
pl.title('static load = %f MPa' % (static_load/1e6))
# Save png image of figure in system temporary directory
pl.savefig(os.path.join(tempfile.gettempdir(),'amplitude_experiment_%fMPa.png' % (static_load/1e6)),dpi=300)


pl.show()  # Display figures

