import copy
import sys
import numpy as np
from numpy import sqrt,log,pi,cos,arctan
import scipy.optimize
import scipy as sp

import pandas as pd

from matplotlib import pylab as pl
#pl.rc('text', usetex=True) # Support greek letters in plot legend

from crackclosuresim2 import inverse_closure,crackopening_from_tensile_closure
from crackclosuresim2 import ModeI_throughcrack_CODformula
from crackclosuresim2 import Tada_ModeI_CircularCrack_along_midline
from crackclosuresim2 import inverse_closure,crackopening_from_tensile_closure
from crackclosuresim2 import ModeI_throughcrack_CODformula
from crackclosuresim2 import Tada_ModeI_CircularCrack_along_midline
from crackclosuresim2 import solve_normalstress
from crackclosuresim2 import load_closurestress
from crackclosuresim2 import ModeII_throughcrack_CSDformula
from crackclosuresim2.fabrikant import Fabrikant_ModeII_CircularCrack_along_midline

from crackclosuresim2.soft_closure import sc_params
from crackclosuresim2.soft_closure import calc_contact
from crackclosuresim2.soft_closure import soft_closure_plots
from crackclosuresim2.soft_closure import sigmacontact_from_displacement
from crackclosuresim2.soft_closure import sigmacontact_from_stress
#from angled_friction_model.asperity_stiffness import asperity_stiffness

from angled_friction_model.angled_friction_model import angled_friction_model
from angled_friction_model.angled_friction_model import integrate_power

from function_as_script import scriptify


if __name__=="__main__":
        #####INPUT VALUES
    E = 117.66e9    #Plane stress Modulus of Elasticity
    Eeff=E
    sigma_yield = 1061e6
    tau_yield = sigma_yield/2.0 # limits stress concentration around singularity
    nu = 0.32   #Poisson's Ratio


    friction_coefficient=0.3
    msqrtR = 8e6
    
    vibration_frequency=20e3  # (Hz)
    
    static_load=505071620.05602247  # tensile static load of 
    vib_normal_stress_ampl =2905917.0601613005  # vibrational normal stress amplitude (Pa). 
    vib_shear_stress_ampl = 2992.831058873761 # Vibrational shear stress amplitude (Pa)
    
    
    # Standard deviation representing crack surface tortuosity
    angular_stddev = 0.388978692989779
    aleft = 0.0020961382738497914 # half-crack length

    numdraws=20 # Number of draws from crack tortuosity  per step

    
    #UTCA_002X_closure = pd.read_csv()
    


    #crack_model_normal = ModeI throughcrack_CODformula(E)
    crack_model_normal = Tada_ModeI_CircularCrack_along_midline(E,nu)
    #crack_model_shear = ModeII_throughcrack_CSDformula(E,nu)
    crack_model_shear = Fabrikant_ModeII_CircularCrack_along_midline(E,nu)


    xmax = 3.5e-3  # Maximum position from center to calculate to;
    # should exceed half-crack lengths 

    doplots = True
    verbose = True
    
    (x_left,
     x_bnd_left,
     dx_left,
     aleft_verify,
     closure_stress_left,
     crack_opening_left_notpresent) = load_closurestress("0000-C14-UTCA-002X_hl_optical_collect_optical_data_dic_closureprofile_closurestress_side1.csv")


    # Evaluate initial crack opening gaps from extrapolated tensile closure field
    crack_initial_opening_left = crackopening_from_tensile_closure(x_left,x_bnd_left,closure_stress_left,dx_left,aleft,sigma_yield,crack_model_normal)

    # Now calculate crack heating under the given conditions
    (power_per_m2_left,
     power_per_m2_stddev_left,
     vibration_ampl_left,
     shear_vibration_ampl_left) = angled_friction_model(x_bnd_left,x_left,dx_left,
                                                        numdraws,
                                                        E,nu,
                                                        sigma_yield,tau_yield,
                                                        friction_coefficient,
                                                        closure_stress_left,
                                                        crack_initial_opening_left,
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
                                                        "quarterpenny",
                                                        None,
                                                        verbose,
                                                        doplots)


    (totalpower_left, totalpower_stddev_left) = integrate_power(x_left,"quarterpenny",None,power_per_m2_left,power_per_m2_stddev_left)



    pl.figure()
    pl.clf()
    pl.plot(-x_left*1e3,power_per_m2_left/1.e3,'-')
    pl.grid()
    pl.xlabel('Radius from center (mm)')
    pl.ylabel('Heating power (kW/m^2)')
    pl.title('Crack power deposition')
    # Save png image of figure in system temporary directory
    pl.savefig(os.path.join(tempfile.gettempdir(),'frictional_heating_of_UTCA-002X.png'),dpi=300)
    

    pl.show()  # Display figures
