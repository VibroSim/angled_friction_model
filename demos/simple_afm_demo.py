import sys
import os
import os.path
import tempfile
import scipy 
import numpy as np
from matplotlib import pyplot as pl

from crackclosuresim2 import inverse_closure
from crackclosuresim2 import crackopening_from_tensile_closure
from crackclosuresim2 import solve_normalstress

from crackclosuresim2 import ModeI_throughcrack_CODformula
from crackclosuresim2 import Tada_ModeI_CircularCrack_along_midline
from crackclosuresim2 import ModeII_throughcrack_CSDformula
from crackclosuresim2.fabrikant import Fabrikant_ModeII_CircularCrack_along_midline

from function_as_script import scriptify

from angled_friction_model.angled_friction_model import angled_friction_model
#from angled_friction_model.angled_friction_model import angled_friction_model as angled_friction_model_function
#angled_friction_model = scriptify(angled_friction_model_function)


from angled_friction_model.angled_friction_model import integrate_power



doplots=False  # extra plots from angled friction model
verbose=False  # More verbose prints from angled fiction model




friction_coefficient=0.3

vibration_frequency=20e3  # (Hz)

static_load=60e6  # tensile static load of 60MPa
vib_normal_stress_ampl =40e6  # vibrational normal stress amplitude (Pa). 
vib_shear_stress_ampl = 0.0 # Vibrational shear stress amplitude (Pa)


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
seff_rightside=np.array([ 0.0, 50e6, 100e6, 150e6, 200e6, 250e6, 300e6, 350e6, 400e6],dtype='d')

# half-crack lengths for the left-hand side (meters)
reff_leftside=np.array([ .5e-3, .7e-3, .9e-3, 1.05e-3, 1.2e-3, 1.33e-3, 1.45e-3, 1.56e-3, 1.66e-3],dtype='d')

# Corresponding opening stresses, units of Pa
seff_leftside=np.array([ 0.0, 50e6, 100e6, 150e6, 200e6, 250e6, 300e6, 350e6, 400e6],dtype='d')

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




# Plot the evaluated closure state
pl.figure()
pl.plot(xrange[xrange < aleft]*1e3,closure_stress_leftside[xrange < aleft]/1e6,'-',
        reff_leftside*1e3,seff_leftside/1e6,'x')
for observcnt in range(len(reff_leftside)):        
    (effective_length, sigma, tensile_displ, dsigmaext_dxt) = solve_normalstress(xrange,x_bnd,closure_stress_leftside,xstep,seff_leftside[observcnt],aleft,sigma_yield,crack_model_normal)
    pl.plot(effective_length*1e3,seff_leftside[observcnt]/1e6,'.')
    pass
pl.grid(True)
pl.legend(('Closure stress field','Observed crack tip posn','Recon. crack tip posn'),loc="best")
pl.xlabel('Radius from crack center (mm)')
pl.ylabel('Stress (MPa)')
pl.title('Crack closure state')



# Now calculate crack heating under the given conditions
(power_per_m2_left,
 power_per_m2_stddev_left,
 vibration_ampl_left) = angled_friction_model(x_bnd,xrange,xstep,
                                              numdraws,
                                              E,nu,
                                              sigma_yield,tau_yield,
                                              friction_coefficient,
                                              closure_stress_leftside,
                                              crack_initial_opening_leftside,
                                              angular_stddev,
                                              aleft,
                                              static_load,
                                              vib_normal_stress_ampl,
                                              vib_shear_stress_ampl,
                                              vibration_frequency,
                                              crack_model_normal,
                                              crack_model_shear,
                                              1.0,
                                              msqrtR,
                                              verbose,
                                              doplots)

(power_per_m2_right,
 power_per_m2_stddev_right,
 vibration_ampl_right) = angled_friction_model(x_bnd,xrange,xstep,
                                               numdraws,
                                               E,nu,
                                               sigma_yield,tau_yield,
                                               friction_coefficient,
                                               closure_stress_rightside,
                                               crack_initial_opening_rightside,
                                               angular_stddev,
                                               aright,
                                               static_load,
                                               vib_normal_stress_ampl,
                                               vib_shear_stress_ampl,
                                               vibration_frequency,
                                               crack_model_normal,
                                               crack_model_shear,
                                               1.0,
                                               msqrtR,
                                               verbose,
                                               doplots)

(totalpower_left, totalpower_stddev_left) = integrate_power(xrange,power_per_m2_left,power_per_m2_stddev_left)
(totalpower_right, totalpower_stddev_right) = integrate_power(xrange,power_per_m2_right,power_per_m2_stddev_right)

totalpower=totalpower_left + totalpower_right

pl.figure()
pl.clf()
pl.plot(-xrange*1e3,power_per_m2_left/1.e3,'-',
        xrange*1e3,power_per_m2_right/1.e3,'-',)
pl.grid()
pl.xlabel('Radius from center (mm)')
pl.ylabel('Heating power (kW/m^2)')
pl.title('Crack power deposition')
# Save png image of figure in system temporary directory
pl.savefig(os.path.join(tempfile.gettempdir(),'frictional_heating.png'),dpi=300)


pl.show()  # Display figures

