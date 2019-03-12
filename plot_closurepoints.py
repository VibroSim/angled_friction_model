import scipy 
import numpy as np
from matplotlib import pyplot as pl

from crackclosuresim2 import inverse_closure,solve_normalstress
from crackclosuresim2 import Tada_ModeI_CircularCrack_along_midline
from crackclosuresim2 import ModeII_throughcrack_CSDformula
from crackclosuresim2.fabrikant import Fabrikant_ModeII_CircularCrack_along_midline

from function_as_script import scriptify

#from angled_friction_model import angled_friction_model

from angled_friction_model import angled_friction_model as angled_friction_model_function
angled_friction_model = scriptify(angled_friction_model_function)


verbose=False

# note: our beta is 90deg - evans and hutchinson beta
beta_components = ( (1.0,0.0,28*np.pi/180.0), )  # Each entry: component magnitude, gaussian mean, Gaussian sigma

friction_coefficient=0.3

static_load=40e6  # tensile static load of 60MPa
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

## units of meters? half-crack lengths for a surface crack  
#reff_leftside=np.array([ .5e-3, .7e-3, .9e-3, 1.05e-3, 1.2e-3, 1.33e-3, 1.45e-3, 1.56e-3, 1.66e-3],dtype='d')

## opening stresses, units of Pa
#seff_leftside=np.array([ -150e6, 50e6, 100e6, 150e6, 200e6, 250e6, 300e6, 350e6, 400e6],dtype='d')







#aleft=np.max(reff_leftside) # NOTE CHANGED SIGN OF aleft
aright=np.max(reff_rightside)

xmax = 2e-3
#assert(xmax > aleft)
assert(xmax > aright)

approximate_xstep=25e-6 # 25um
num_boundary_steps=int((xmax)//approximate_xstep)
numsteps = num_boundary_steps-1
xstep = (xmax)/(numsteps)
numdraws=20 # draws per step

x_bnd = xstep*np.arange(num_boundary_steps) # 
xrange = (x_bnd[1:] + x_bnd[:-1])/2.0


#closure_stress_leftside=inverse_closure(reff_leftside,seff_leftside,xrange,x_bnd,xstep,aleft,sigma_yield,crack_model_normal,verbose=verbose)
closure_stress_rightside=inverse_closure(reff_rightside,seff_rightside,xrange,x_bnd,xstep,aright,sigma_yield,crack_model_normal,verbose=verbose)



vib_ampls = np.arange(0.0,70.0e6,5.0e6)

closure_point_sub=np.zeros(vib_ampls.shape,dtype='d')
closure_point_add=np.zeros(vib_ampls.shape,dtype='d')

sigma_sub=np.zeros((vib_ampls.shape[0],xrange.shape[0]),dtype='d')
sigma_add=np.zeros((vib_ampls.shape[0],xrange.shape[0]),dtype='d')

tensile_displ_sub=np.zeros((vib_ampls.shape[0],xrange.shape[0]),dtype='d')
tensile_displ_add=np.zeros((vib_ampls.shape[0],xrange.shape[0]),dtype='d')

for ampl_idx in range(vib_ampls.shape[0]):  # vibrational normal stress amplitude.
    vib_normal_stress_ampl=vib_ampls[ampl_idx]

    (closure_point_sub[ampl_idx], sigma_sub[ampl_idx,:], tensile_displ_sub[ampl_idx,:]) = solve_normalstress(xrange,x_bnd,closure_stress_rightside,xstep,static_load-vib_normal_stress_ampl,aright,sigma_yield,crack_model_normal,calculate_displacements=True)

    (closure_point_add[ampl_idx], sigma_add[ampl_idx,:], tensile_displ_add[ampl_idx,:]) = solve_normalstress(xrange,x_bnd,closure_stress_rightside,xstep,static_load+vib_normal_stress_ampl,aright,sigma_yield,crack_model_normal,calculate_displacements=True)

    
    pass

pl.figure()
pl.clf()
pl.plot(vib_ampls/1e6,closure_point_add*1e3,'-',
        vib_ampls/1e6,closure_point_sub*1e3,'-')
pl.grid()
pl.legend(('closure_point_add','closure_point_sub'))
pl.xlabel('Vibration amplitude (MPa)')
pl.ylabel('Closure point (mm)')
pl.title('static load = %f MPa' % (static_load/1e6))
pl.savefig('/tmp/closurepoints_%fMPa.png' % (static_load/1e6),dpi=300)


pl.figure()
pl.clf()
pl.plot(vib_ampls/1e6,sigma_add/1e6,'-')
pl.grid()
pl.xlabel('Vibration amplitude (MPa)')
pl.ylabel('sigma_add (MPa)')
pl.title('sigma_add static load = %f MPa' % (static_load/1e6))
pl.savefig('/tmp/closurepoints_sigma_add_%fMPa.png' % (static_load/1e6),dpi=300)

pl.figure()
pl.clf()
pl.plot(vib_ampls/1e6,np.mean(sigma_add,1)/1e6,'-')
pl.grid()
pl.xlabel('Vibration amplitude (MPa)')
pl.ylabel('sigma_add mean (MPa)')
pl.title('sigma_add static load = %f MPa' % (static_load/1e6))
pl.savefig('/tmp/closurepoints_sigma_add_mean_%fMPa.png' % (static_load/1e6),dpi=300)


pl.figure()
pl.clf()
pl.plot(vib_ampls/1e6,tensile_displ_add*1e9,'-')
pl.grid()
pl.xlabel('Vibration amplitude (MPa)')
pl.ylabel('tensile_displ_add (nm)')
pl.title('tensile_displ_add static load = %f MPa' % (static_load/1e6))
pl.savefig('/tmp/closurepoints_tensile_displ_add_%fMPa.png' % (static_load/1e6),dpi=300)

pl.figure()
pl.clf()
pl.plot(vib_ampls/1e6,np.mean(tensile_displ_add,1)*1e9,'-')
pl.grid()
pl.xlabel('Vibration amplitude (MPa)')
pl.ylabel('tensile_displ_add (nm)')
pl.title('tensile_displ_add mean static load = %f MPa' % (static_load/1e6))
pl.savefig('/tmp/closurepoints_tensile_displ_add_%fMPa.png' % (static_load/1e6),dpi=300)


pl.figure()
pl.clf()
pl.plot(vib_ampls/1e6,sigma_sub/1e6,'-')
pl.grid()
pl.xlabel('Vibration amplitude (MPa)')
pl.ylabel('sigma_sub (MPa)')
pl.title('sigma_sub static load = %f MPa' % (static_load/1e6))
pl.savefig('/tmp/closurepoints_sigma_sub_%fMPa.png' % (static_load/1e6),dpi=300)

pl.figure()
pl.clf()
pl.plot(vib_ampls/1e6,np.mean(sigma_sub,1)/1e6,'-')
pl.grid()
pl.xlabel('Vibration amplitude (MPa)')
pl.ylabel('sigma_sub mean (MPa)')
pl.title('sigma_sub static load = %f MPa' % (static_load/1e6))
pl.savefig('/tmp/closurepoints_sigma_sub_mean_%fMPa.png' % (static_load/1e6),dpi=300)


pl.figure()
pl.clf()
pl.plot(vib_ampls/1e6,tensile_displ_sub*1e9,'-')
pl.grid()
pl.xlabel('Vibration amplitude (MPa)')
pl.ylabel('tensile_displ_sub (nm)')
pl.title('tensile_displ_sub static load = %f MPa' % (static_load/1e6))
pl.savefig('/tmp/closurepoints_tensile_displ_sub_%fMPa.png' % (static_load/1e6),dpi=300)

pl.figure()
pl.clf()
pl.plot(vib_ampls/1e6,np.mean(tensile_displ_sub,1)*1e9,'-')
pl.grid()
pl.xlabel('Vibration amplitude (MPa)')
pl.ylabel('tensile_displ_sub (nm)')
pl.title('tensile_displ_sub mean static load = %f MPa' % (static_load/1e6))
pl.savefig('/tmp/closurepoints_tensile_displ_sub_%fMPa.png' % (static_load/1e6),dpi=300)


pl.show()
