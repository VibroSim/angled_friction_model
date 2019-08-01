import scipy 
import numpy as np
from matplotlib import pyplot as pl

from crackclosuresim2 import inverse_closure
from crackclosuresim2 import Tada_ModeI_CircularCrack_along_midline
from crackclosuresim2 import ModeII_throughcrack_CSDformula
from crackclosuresim2.fabrikant import Fabrikant_ModeII_CircularCrack_along_midline

## !!!*** WARNING: THERE IS BIZARRE BEHAVIOR WHEN THE NET
## LOAD DURING A VIBRATION CYCLE GOES ZERO OR NEGATIVE:
## WE SEE A LARGE INCREMENT IN HEATING AT THE LARGEST
## VIB AMPLITUDES.
## CURIOUSLY, SETTING THE STATIC LOAD TO BE SMALLER
## IN SIMPLE_AFM_DEMO.PY DOESN'T SEEM TO GIVE THE SAME
## EFFECT (?) WHATEVER IT IS MUST CERTAINLY BE NON-PHYSICAL!!!

from function_as_script import scriptify

from angled_friction_model.angled_friction_model import angled_friction_model
#from angled_friction_model.angled_friction_model import angled_friction_model as angled_friction_model_function
#angled_friction_model = scriptify(angled_friction_model_function)



doplots=True
verbose=False

i=(0+1j) # imaginary number

# pdf (need not be normalized) of surface orientation beta
# domain: -pi (backwards trajectory, infinitesimally downward)
# to pi (backwards trajectory, infinitesimally upward)

# note: our beta is 90deg - evans and hutchinson beta
beta_components = ( (1.0,0.0,28*np.pi/180.0), )  # Each entry: component magnitude, gaussian mean, Gaussian sigma

friction_coefficient=0.3

vibration_frequency=20e3  # (Hz)

static_load=60e6  # tensile static load of 60MPa
# assume also that there is no synergy between heating from different modes. 

# x is position along crack (currently no x dependence to beta 
#beta_unnorm_pdf = lambda beta,x:  np.array([ (magnitude/np.sqrt(2*np.pi*sigma**2)) * np.exp(-(beta - mean)**2/(2.0*sigma**2.0)) for (magnitude,mean,sigma) in beta_components ],dtype='d').sum(0)

assert(len(beta_components)==1 and beta_components[0][0]==1.0)

#beta_drawfunc = lambda x: np.random.randn()*beta_components[0][2]+beta_components[0][1]
angular_stddev = beta_components[0][2]


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
seff_rightside=np.array([ -150e6, 50e6, 100e6, 150e6, 200e6, 250e6, 300e6, 350e6, 400e6],dtype='d')

# units of meters? half-crack lengths for a surface crack  
reff_leftside=np.array([ .5e-3, .7e-3, .9e-3, 1.05e-3, 1.2e-3, 1.33e-3, 1.45e-3, 1.56e-3, 1.66e-3],dtype='d')

# opening stresses, units of Pa
seff_leftside=np.array([ -150e6, 50e6, 100e6, 150e6, 200e6, 250e6, 300e6, 350e6, 400e6],dtype='d')







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

msqrtR = 1000.0e6 * np.sqrt(15e-6) # asperity density (asperities/m^2) * sqrt(contact radius) (sqrt(m))

x_bnd = xstep*np.arange(num_boundary_steps) # 
xrange = (x_bnd[1:] + x_bnd[:-1])/2.0


closure_stress_leftside=inverse_closure(reff_leftside,seff_leftside,xrange,x_bnd,xstep,aleft,sigma_yield,crack_model_normal,verbose=verbose)
closure_stress_rightside=inverse_closure(reff_rightside,seff_rightside,xrange,x_bnd,xstep,aright,sigma_yield,crack_model_normal,verbose=verbose)



vib_ampls = np.arange(0.0,70.0e6,5.0e6)

total_heating_right=np.zeros(vib_ampls.shape,dtype='d')



for ampl_idx in range(vib_ampls.shape[0]):  # vibrational normal stress amplitude. 
    vib_normal_stress_ampl=vib_ampls[ampl_idx]
    vib_shear_stress_ampl = 0.0 # assume 0 shear
    #vib_shear_stress_ampl = vib_normal_stress_ampl  # Assume shear amplitude peaks simultaneously with
# normal stress. 
    
    (power_per_m2_right,
     vibration_ampl_right) = angled_friction_model(x_bnd,xrange,xstep,
                                                   numdraws,
                                                   E,nu,
                                                   sigma_yield,tau_yield,
                                                   friction_coefficient,
                                                   closure_stress_rightside,
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
                                                   verbose,
                                                   doplots)
    
    total_heating_right[ampl_idx] = np.sum(power_per_m2_right*xrange*xstep*np.pi/2)  # integrate heating of half-semicircles (integrate power *rdrdtheta)
    pass

pl.figure()
pl.clf()
pl.plot(vib_ampls/1e6,total_heating_right*1e3,'-')
pl.grid()
pl.xlabel('Vibration amplitude (MPa)')
pl.ylabel('Heating power (mW)')
pl.title('static load = %f MPa' % (static_load/1e6))
pl.savefig('/tmp/amplitude_experiment_%fMPa.png' % (static_load/1e6),dpi=300)
#pl.show()


pl.show()

