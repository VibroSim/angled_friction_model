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
from angled_friction_model.asperity_stiffness import asperity_stiffness

#from function_as_script import scriptify


from crackclosuresim2 import soft_closure



verbose=False


friction_coefficient=0.3

static_load=40e6  # tensile static load of 60MPa

# Standard deviation representing crack surface tortuosity
angular_stddev = 28*np.pi/180.0


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
seff_rightside=np.array([ -150e6, 50e6, 100e6, 150e6, 200e6, 250e6, 300e6, 350e6, 400e6],dtype='d')



# Fully open crack length
aright=np.max(reff_rightside)

assert(aright < xmax) # Increase xmax if this assertion fails



# Determine closure stress field from observed crack length data
closure_stress_rightside=inverse_closure(reff_rightside,seff_rightside,xrange,x_bnd,xstep,aright,sigma_yield,crack_model_normal,verbose=verbose)

# Evaluate initial crack opening gaps from extrapolated tensile closure field
crack_initial_opening_rightside = crackopening_from_tensile_closure(xrange,x_bnd,closure_stress_rightside,xstep,aright,sigma_yield,crack_model_normal)

# Initialze soft closure model parameters

Hm = asperity_stiffness(msqrtR,E,nu,angular_stddev)

scp = soft_closure.sc_params.fromcrackgeom(crack_model_normal,x_bnd[-1],num_boundary_steps,aright,1,Hm)

# Define initial closure state of soft closure model based on evaluated closure stress and initial opening
# For initialize_contact, treat closure_stress_rightside as strictly compressive (positive)
scp.initialize_contact(closure_stress_rightside*(closure_stress_rightside > 0.0),crack_initial_opening_rightside)


vib_ampls = np.arange(0.0,70.0e6,5.0e6)

closure_point_sub=np.zeros(vib_ampls.shape,dtype='d')
closure_point_add=np.zeros(vib_ampls.shape,dtype='d')

sigma_sub=np.zeros((vib_ampls.shape[0],xrange.shape[0]),dtype='d')
sigma_add=np.zeros((vib_ampls.shape[0],xrange.shape[0]),dtype='d')

tensile_displ_sub=np.zeros((vib_ampls.shape[0],xrange.shape[0]),dtype='d')
tensile_displ_add=np.zeros((vib_ampls.shape[0],xrange.shape[0]),dtype='d')

for ampl_idx in range(vib_ampls.shape[0]):  # vibrational normal stress amplitude.
    vib_normal_stress_ampl=vib_ampls[ampl_idx]

    #  This commented-out code finds the closure point using the hard contact model
    #(closure_point_sub[ampl_idx], sigma_sub[ampl_idx,:], tensile_displ_sub[ampl_idx,:], dsigmaext_dxt_sub) = solve_normalstress(xrange,x_bnd,closure_stress_rightside,xstep,static_load-vib_normal_stress_ampl,aright,sigma_yield,crack_model_normal,calculate_displacements=True)

    #(closure_point_add[ampl_idx], sigma_add[ampl_idx,:], tensile_displ_add[ampl_idx,:],dsigmaext_dxt_add) = solve_normalstress(xrange,x_bnd,closure_stress_rightside,xstep,static_load+vib_normal_stress_ampl,aright,sigma_yield,crack_model_normal,calculate_displacements=True)


    # The active code here uses the soft contact model.
    # nb. displacements returned by the soft contact model
    # are displacements between the two crack surfaces,
    # whereas the displacements returned by the hard
    # contact model are displacements from the symmetry plane.
    # So expect a factor of two in reported displacement between
    # the two scenarios

    (du_da_sub,contact_stress_sub,tensile_displ_sub[ampl_idx,:])=soft_closure.calc_contact(scp,static_load-vib_normal_stress_ampl)
    (du_da_add,contact_stress_add,tensile_displ_add[ampl_idx,:])=soft_closure.calc_contact(scp,static_load+vib_normal_stress_ampl)

    sigma_sub[ampl_idx,:]=-contact_stress_sub
    sigma_add[ampl_idx,:]=-contact_stress_add


    sub_notopening = np.where(tensile_displ_sub[ampl_idx,:] < 0)[0]
    if sub_notopening.shape[0] > 0:
        closure_point_sub[ampl_idx] = xrange[sub_notopening[0]]
        pass
    else:
        closure_point_sub[ampl_idx]=aright
        pass

    add_notopening = np.where(tensile_displ_add[ampl_idx,:] < 0)[0]
    if add_notopening.shape[0] > 0:
        closure_point_add[ampl_idx] = xrange[add_notopening[0]]
        pass
    else:
        closure_point_add[ampl_idx]=aright
        pass


    
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
pl.savefig(os.path.join(tempfile.gettempdir(),'closurepoints_%fMPa.png' % (static_load/1e6)),dpi=300)


pl.figure()
pl.clf()
pl.plot(vib_ampls/1e6,sigma_add[:,xrange <= aright]/1e6,'-')
pl.grid()
pl.xlabel('Vibration amplitude (MPa)')
pl.ylabel('sigma_add (MPa)')
pl.title('sigma_add static load = %f MPa' % (static_load/1e6))
pl.savefig(os.path.join(tempfile.gettempdir(),'closurepoints_sigma_add_%fMPa.png' % (static_load/1e6)),dpi=300)

pl.figure()
pl.clf()
pl.plot(vib_ampls/1e6,np.mean(sigma_add[:,xrange <= aright],1)/1e6,'-')
pl.grid()
pl.xlabel('Vibration amplitude (MPa)')
pl.ylabel('sigma_add mean (MPa)')
pl.title('sigma_add static load = %f MPa' % (static_load/1e6))
pl.savefig(os.path.join(tempfile.gettempdir(),'closurepoints_sigma_add_mean_%fMPa.png' % (static_load/1e6)),dpi=300)


pl.figure()
pl.clf()
pl.plot(vib_ampls/1e6,tensile_displ_add[:,xrange <= aright]*1e9,'-')
pl.grid()
pl.xlabel('Vibration amplitude (MPa)')
pl.ylabel('tensile_displ_add (nm)')
pl.title('tensile_displ_add static load = %f MPa' % (static_load/1e6))
pl.savefig(os.path.join(tempfile.gettempdir(),'closurepoints_tensile_displ_add_%fMPa.png' % (static_load/1e6)),dpi=300)

pl.figure()
pl.clf()
pl.plot(vib_ampls/1e6,np.mean(tensile_displ_add[:,xrange <= aright],1)*1e9,'-')
pl.grid()
pl.xlabel('Vibration amplitude (MPa)')
pl.ylabel('tensile_displ_add (nm)')
pl.title('tensile_displ_add mean static load = %f MPa' % (static_load/1e6))
pl.savefig(os.path.join(tempfile.gettempdir(),'closurepoints_tensile_displ_add_%fMPa.png' % (static_load/1e6)),dpi=300)


pl.figure()
pl.clf()
pl.plot(vib_ampls/1e6,sigma_sub[:,xrange <= aright]/1e6,'-')
pl.grid()
pl.xlabel('Vibration amplitude (MPa)')
pl.ylabel('sigma_sub (MPa)')
pl.title('sigma_sub static load = %f MPa' % (static_load/1e6))
pl.savefig(os.path.join(tempfile.gettempdir(),'closurepoints_sigma_sub_%fMPa.png' % (static_load/1e6)),dpi=300)

pl.figure()
pl.clf()
pl.plot(vib_ampls/1e6,np.mean(sigma_sub[:,xrange <= aright],1)/1e6,'-')
pl.grid()
pl.xlabel('Vibration amplitude (MPa)')
pl.ylabel('sigma_sub mean (MPa)')
pl.title('sigma_sub static load = %f MPa' % (static_load/1e6))
pl.savefig(os.path.join(tempfile.gettempdir(),'closurepoints_sigma_sub_mean_%fMPa.png' % (static_load/1e6)),dpi=300)


pl.figure()
pl.clf()
pl.plot(vib_ampls/1e6,tensile_displ_sub[:,xrange <= aright]*1e9,'-')
pl.grid()
pl.xlabel('Vibration amplitude (MPa)')
pl.ylabel('tensile_displ_sub (nm)')
pl.title('tensile_displ_sub static load = %f MPa' % (static_load/1e6))
pl.savefig(os.path.join(tempfile.gettempdir(),'closurepoints_tensile_displ_sub_%fMPa.png' % (static_load/1e6)),dpi=300)

pl.figure()
pl.clf()
pl.plot(vib_ampls/1e6,np.mean(tensile_displ_sub[:,xrange <= aright],1)*1e9,'-')
pl.grid()
pl.xlabel('Vibration amplitude (MPa)')
pl.ylabel('tensile_displ_sub (nm)')
pl.title('tensile_displ_sub mean static load = %f MPa' % (static_load/1e6))
pl.savefig(os.path.join(tempfile.gettempdir(),'closurepoints_tensile_displ_sub_%fMPa.png' % (static_load/1e6)),dpi=300)


pl.show()
