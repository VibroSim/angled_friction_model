import scipy 
import numpy as np
from matplotlib import pyplot as pl

from crackclosuresim2 import inverse_closure
from crackclosuresim2 import Tada_ModeI_CircularCrack_along_midline
from crackclosuresim2 import ModeII_throughcrack_CSDformula
from crackclosuresim2.fabrikant import Fabrikant_ModeII_CircularCrack_along_midline



from function_as_script import scriptify

from angled_friction_model import angled_friction_model as angled_friction_model_function

angled_friction_model = scriptify(angled_friction_model_function)



doplots=True

i=(0+1j) # imaginary number

# pdf (need not be normalized) of surface orientation beta
# domain: -pi (backwards trajectory, infinitesimally downward)
# to pi (backwards trajectory, infinitesimally upward)

# note: our beta is 90deg - evans and hutchinson beta
beta_components = ( (2.0,0.0,28*np.pi/180.0), )  # Each entry: component magnitude, gaussian mean, Gaussian sigma

friction_coefficient=0.3

vibration_frequency=20e3  # (Hz)

static_load=60e6  # tensile static load of 60MPa
vib_normal_stress_ampl =40e6  # vibrational normal stress amplitude. 
vib_shear_stress_ampl = 15e6  # Assume shear amplitude peaks simultaneously with
# normal stress. NOT CURRENTLY USED!!!
# assume also that there is no synergy between heating from different modes. 

# x is position along crack (currently no x dependence to beta 
beta_unnorm_pdf = lambda beta,x:  np.array([ (magnitude/np.sqrt(2*np.pi*sigma**2)) * np.exp(-(beta - mean)**2/(2.0*sigma**2.0)) for (magnitude,mean,sigma) in beta_components ],dtype='d').sum(0)

# crackclosuresim parameters
# Low K
#E = 109e9
nu = 0.33
E = 207.83e9 # Plane stress
sigma_yield=600e6 # CHECK THIS NUMBER
tau_yield=sigma_yield/2.0
#E = 207.83e9/(1.0-nu**2.0) # Plane strain
G=E/(2*(1+nu))
width=25.4e-3

crack_model_normal = Tada_ModeI_CircularCrack_along_midline(E,nu)
#crack_model_shear = ModeII_throughcrack_CSDformula(E,nu)
crack_model_shear = Fabrikant_ModeII_CircularCrack_along_midline(E,nu)


# units of meters? half-crack lengths for a surface crack  
reff_rightside=np.array([ .5e-3, .7e-3, .9e-3, 1.05e-3, 1.2e-3, 1.33e-3, 1.45e-3, 1.56e-3, 1.66e-3],dtype='d')

# opening stresses, units of Pa
seff_rightside=np.array([ 0.0, 50e6, 100e6, 150e6, 200e6, 250e6, 300e6, 350e6, 400e6],dtype='d')

# units of meters? half-crack lengths for a surface crack  
reff_leftside=np.array([ .5e-3, .7e-3, .9e-3, 1.05e-3, 1.2e-3, 1.33e-3, 1.45e-3, 1.56e-3, 1.66e-3],dtype='d')

# opening stresses, units of Pa
seff_leftside=np.array([ 0.0, 50e6, 100e6, 150e6, 200e6, 250e6, 300e6, 350e6, 400e6],dtype='d')







aleft=np.max(reff_leftside) # NOTE CHANGED SIGN OF aleft
aright=np.max(reff_rightside)

xmax = 2e-3
assert(xmax > aleft)
assert(xmax > aright)

approximate_xstep=25e-6 # 25um
num_boundary_steps=int((xmax)//approximate_xstep)
numsteps = num_boundary_steps-1
xstep = (xmax)/(numsteps)
numdraws=20 # draws per step

x_bnd = xstep*np.arange(num_boundary_steps) # 
xrange = (x_bnd[1:] + x_bnd[:-1])/2.0


closure_stress_leftside=inverse_closure(reff_leftside,seff_leftside,xrange,x_bnd,xstep,aleft,sigma_yield,crack_model_normal)
closure_stress_rightside=inverse_closure(reff_rightside,seff_rightside,xrange,x_bnd,xstep,aright,sigma_yield,crack_model_normal)



(power_per_m2_left,
 vibration_ampl_left) = angled_friction_model(x_bnd,xrange,xstep,
                                              numdraws,
                                              E,nu,
                                              sigma_yield,tau_yield,
                                              friction_coefficient,
                                              closure_stress_leftside,
                                              beta_unnorm_pdf,
                                              aleft,
                                              static_load,
                                              vib_normal_stress_ampl,
                                              vib_shear_stress_ampl,
                                              vibration_frequency,
                                              crack_model_normal,
                                              crack_model_shear,
                                              doplots)

(power_per_m2_right,
 vibration_ampl_right) = angled_friction_model(x_bnd,xrange,xstep,
                                               numdraws,
                                               E,nu,
                                               sigma_yield,tau_yield,
                                               friction_coefficient,
                                               closure_stress_rightside,
                                               beta_unnorm_pdf,
                                               aright,
                                               static_load,
                                               vib_normal_stress_ampl,
                                               vib_shear_stress_ampl,
                                               vibration_frequency,
                                               crack_model_normal,
                                               crack_model_shear,
                                               doplots)

pl.show()

