import scipy 
import numpy as np
from scipy.integrate import quad
from scipy.optimize import newton
from numpy.random import rand

try: 
  from collections.abc import Iterable
  pass
except ImportError:
  from collections import Iterable
  pass

from crackclosuresim2 import crackopening_from_tensile_closure
from crackclosuresim2 import solve_normalstress
from crackclosuresim2 import solve_shearstress
from crackclosuresim2 import soft_closure

from .asperity_stiffness import asperity_stiffness


def angled_friction_model(x_bnd,xrange,xstep,
                          numdraws,
                          E,nu,
                          sigma_yield,tau_yield,
                          friction_coefficient,  # ADJUSTABLE, can pass vector
                          closure_stress,
                          crack_initial_opening,
                          angular_stddev, 
                          a_crack,
                          static_load, # Positive tensile
                          vib_normal_stress_ampl,
                          vib_shear_stress_ampl,
                          vibration_frequency,
                          crack_model_normal,
                          crack_model_shear,  
                          crack_model_shear_factor, # ADJUSTABLE shear sensitivity factor (nominally 1.0)
                          msqrtR, # ADJUSTABLE asperity density * sqrt(asperity radius)
                          verbose,doplots):


  beta_drawfunc = lambda x: np.random.randn()*angular_stddev + 0.0
  
  numsteps=xrange.shape[0]

  # soft closure parameters
  #Lm = 10e6/(100e-9**(3.0/2.0))  # rough order of magnitude guess... should be ADJUSTABLE?
  Lm = asperity_stiffness(msqrtR,E,nu,angular_stddev)

  
  scp = soft_closure.sc_params.fromcrackgeom(crack_model_normal,x_bnd[-1],numsteps+1,a_crack,1,Lm)

  
  closure_stress_softmodel = closure_stress.copy()

  # In the soft closure model, sigma_closure (positive compression) can't be negativej
  # (use crack_initial_opening values instead in that domain)

  closure_stress_softmodel[closure_stress_softmodel < 0.0]=0.0


  #scp.setcrackstate(closure_stress_softmodel,crack_initial_opening)

  # initialize_contact() solves to find the compressive portion
  # of crack_initial_opening based on the closure stress profile
  scp.initialize_contact(closure_stress_softmodel,crack_initial_opening)

  # Evaluate contact stress on both sides of static load
  (du_da_sub,contact_stress_sub,tensile_displ_sub)=soft_closure.calc_contact(scp,static_load-vib_normal_stress_ampl)
  (du_da_add,contact_stress_add,tensile_displ_add)=soft_closure.calc_contact(scp,static_load+vib_normal_stress_ampl)


  sigma_sub=-contact_stress_sub  # sigma_sub and sigma_add are positive tensile
  sigma_add=-contact_stress_add

  sub_notopening = np.where(tensile_displ_sub < 0)[0]
  if sub_notopening.shape[0] > 0:
    closure_point_sub = xrange[sub_notopening[0]]
    pass
  else:
    closure_point_sub=a_crack
    pass

  add_notopening = np.where(tensile_displ_add < 0)[0]
  if add_notopening.shape[0] > 0:
    closure_point_add = xrange[add_notopening[0]]
    pass
  else:
    closure_point_add=a_crack
    pass

  
  #(closure_point_sub, sigma_sub, tensile_displ_sub) = solve_normalstress(xrange,x_bnd,closure_stress,xstep,static_load-vib_normal_stress_ampl,a_crack,sigma_yield,crack_model_normal,calculate_displacements=True)
    
  
  #(closure_point_add, sigma_add, tensile_displ_add) = solve_normalstress(xrange,x_bnd,closure_stress,xstep,static_load+vib_normal_stress_ampl,a_crack,sigma_yield,crack_model_normal,calculate_displacements=True)

  
  if not isinstance(friction_coefficient,Iterable):
    friction_coefficient = (friction_coefficient,)
    unwrapflag=True
    pass
  else:
    unwrapflag=False
    pass

  power_per_m2 = np.zeros((len(friction_coefficient),xrange.shape[0]),dtype='d')
  power_per_m2_stddev = np.zeros((len(friction_coefficient),xrange.shape[0]),dtype='d')
  vibration_ampl = np.zeros((len(friction_coefficient),xrange.shape[0]),dtype='d')

  
  for fc_idx in range(len(friction_coefficient)):
    for xcnt in range(numsteps):
      x=xrange[xcnt]
      
      if verbose: 
        print("x=%f um" % (x*1e6))
        pass
      
      xleft=x_bnd[xcnt]
      xright=x_bnd[xcnt+1]
      
      ## determine normalization factor for pdf at this x position
      #beta_unnorm_int=quad(lambda beta: beta_unnorm_pdf(beta,x),-np.pi,np.pi)[0]
      #
      ## normalized pdf
      #beta_pdf = lambda beta: beta_unnorm_pdf(beta,x)/beta_unnorm_int    
      #
      ## cdf 
      #beta_cdf = lambda beta: quad(beta_pdf,-np.pi,beta)[0]
      #
      ## inverse of cdf   # CDF(beta) = prob -> 
      #beta_cdf_inverse = lambda prob: newton(lambda beta: beta_cdf(beta)-prob,0.0)
      
      beta_draws = np.array([ beta_drawfunc(x) for cnt in range(numdraws) ]) # = np.vectorize(beta_cdf_inverse)(rand(numdraws))
      
      
      #closure_state_x = splev(x,stress_field_spl,ext=1) 
      #closure_state_x = closure_stress[xcnt]
      #print("closure_stress[%d]=%f; contact_stress_static[%d]=%f" % (xcnt,closure_stress[xcnt],xcnt,contact_stress_static[xcnt]))
      #closure_state_x = contact_stress_static[xcnt]  # positive compression
      
      
      
      # Evaluate at x
      if x <= closure_point_sub:
        closure_state_sub_x=0.0
        pass
      else:
        closure_state_sub_x = sigma_sub[xcnt] # positive tensile
        pass
      
      if x <= closure_point_add:
        closure_state_add_x=0.0
        pass
      else:
        closure_state_add_x = sigma_add[xcnt] # positive tensile
        pass
      
      if verbose:
        print("sigma_sub[%d]=%f; sigma_add[%d]=%f" % (xcnt,sigma_sub[xcnt],xcnt,sigma_add[xcnt]))
        pass
      
      # uyy is directly calculated value because soft closure model
      # already doubles displacement from the fracture mechanics formula
      uyy_add = tensile_displ_add[xcnt]*1.0
      
      uyy_sub = tensile_displ_sub[xcnt]*1.0
      
      # sigma_sub and sigma_add are positive tensile representations of the
      # normal tractions on the crack surfaces.
      # The closure stress in each of these states therefore would be
      # -sigma_sub and -sigma_add respectively. 
      
      # ss variables are for shear_stickslip calculations
      # !!!*** These solve_shearstress calls I think can be factored out of the xcnt loop but not the shear loop !!!***
      (effective_length_sub, tau_sub, shear_displ_sub) = solve_shearstress(xrange,x_bnd,-sigma_sub,xstep,vib_shear_stress_ampl,a_crack,friction_coefficient[fc_idx],tau_yield,crack_model_shear)
      
      (effective_length_add, tau_add, shear_displ_add) = solve_shearstress(xrange,x_bnd,-sigma_add,xstep,vib_shear_stress_ampl,a_crack,friction_coefficient[fc_idx],tau_yield,crack_model_shear)
    
      
      # Warning: We are not requiring shear continuity between left and right
      # sides of the crack (!) 
    
      
      # Got two closure stress values at this point:
      # closure_state_add_x and closure_state_sub_x
      
      # and corresponding displacements uyy_add and uyy_sub
      
      # For the moment, treat global shear stress and
      # global shear displacement as zero. 
      
      
      # for each beta draw, evaluate local stress field
      # on the angled asperity
      
      # Model: normal stress sinusodially varies with closure_ampl around closure_state_ref
      # shear stress sinusoidally varies also in phase
      # (vib_shear_stress_ampl)
      
      # now consider our draws...
      # We can consider each corresponds to xstep/numdraws units of horizontal
      # distance (but we don't anymore) 


      P_sub = -sigma_sub[xcnt] * xstep * np.pi*x/2.0 # Overall force, normal to crack plane, positive compression on a quarter annulus, in the 'sub' state
      Q_sub = (-(tau_add[xcnt]+tau_sub[xcnt])/2.0)*crack_model_shear_factor*xstep*np.pi*x/2.0 # Overall force, parallel to crack plane, positive compression on a quarter annulus, in the 'sub' state

      P_add = -sigma_add[xcnt] * xstep * np.pi*x/2.0 # Overall force, normal to crack plane, positive compression on a quarter annulus, in the 'add' state
      Q_add = ((tau_add[xcnt]+tau_sub[xcnt])/2.0)*crack_model_shear_factor*xstep*np.pi*x/2.0 # Overall force, parallel to crack plane, positive compression on a quarter annulus, in the 'add' state


      # N_add, T_add, N_sub, T_sub are arrays that are __per_draw__
      # They represent normal (positive compressive) and shear
      # force on each asperity
      N_add = (P_add/numdraws)*np.cos(beta_draws) + (Q_add/numdraws)*np.sin(beta_draws)
      T_add = -(P_add/numdraws)*np.sin(beta_draws) + (Q_add/numdraws)*np.cos(beta_draws)

      N_sub = (P_sub/numdraws)*np.cos(beta_draws) + (Q_sub/numdraws)*np.sin(beta_draws)
      T_sub = -(P_sub/numdraws)*np.sin(beta_draws) + (Q_sub/numdraws)*np.cos(beta_draws)


      
      # P and Q are FORCES... treat them as acting on quarter-annulus
    
      #slip_sub=(np.abs(T_sub) >=  friction_coefficient[fc_idx]*(N_sub)) & (N_sub > 0.0)
      #slip_add=(np.abs(T_add) >=  friction_coefficient[fc_idx]*(N_add)) & (N_add > 0.0)
      net_normal = (N_sub+N_add)/2.0 > 0.0 # Is there a net compressive normal stress?
      slip_noresidualshear = (np.abs(T_sub-T_add) >= friction_coefficient[fc_idx]*(N_sub+N_add)/2.0) & net_normal 

      slip_residualshear = (np.abs((T_sub+T_add)/2.0) >= friction_coefficient[fc_idx]*(N_sub+N_add)/2.0) & net_normal 

      #residualshear_override_fraction = 0.1
      
      #residualshear_override = np.random.rand(numdraws) < residualshear_override_fraction

      slip = slip_noresidualshear # | (slip_residualshear & residualshear_override)
           
      #slip = (slip_sub & slip_add) & net_normal  # Consider any asperity that can slip anywhere in the cycle as full slippage, so long as there is overall normal compression

      # utt is a vibration amplitude... shear_displ_add[xcnt] + shear_displ_sub[xcnt] gives the peak-to-peak value for __each_side__ of the crack. Ampltiude is half that... but also double because relative motion of flanks is twice the motion of each flank... so net no change,
      utt = (shear_displ_add[xcnt] + shear_displ_sub[xcnt])*crack_model_shear_factor
      PP_vibration_y=uyy_add-uyy_sub
      vibration_ampl[fc_idx,xcnt]=PP_vibration_y/2.0
      #PP_vibration_t=utt*2.0
      tangential_vibration_ampl=np.abs(vibration_ampl[fc_idx,xcnt] * np.sin(beta_draws) + utt*np.cos(beta_draws))*slip
      tangential_vibration_velocity_ampl = 2*np.pi*vibration_frequency*tangential_vibration_ampl
      
      # Power = (1/2)Fampl*vampl
      # where Fampl = mu*Normal force
      # Q: Are force and velocity always in-phase (probably not)
      # ... What happens when they are not?
      # Expect dynamic stresses and displacements to be in phase, which
      # would put dynamic stresses out of phase with velocity....
      # What effect would that have on power? Should the N_dynamic term
      # drop out?
      
      # Problem: Experimental observation suggests heating
      # is between linear and quadratic in vibration amplitude.
      # Quadratic would require N to have vibration dependence.
      # P=uNv = u(Nstatic+Ndynamic)v=u(Cn1 + Cn2v)v
      
      # (Note: N_static term was missing from original calculation)
      # sdh 1/10/20 N_static is per draw... comparable to N_dynamic and or T_dynamic
      if x >= closure_point_sub:
        Power = 0.5 * (friction_coefficient[fc_idx]*np.abs(N_sub+N_add)/2.0)*tangential_vibration_velocity_ampl
        #Power = 0.5 * (np.abs(T_sub-T_add))*tangential_vibration_velocity_ampl
        pass
      else:
        Power=0.0
        pass
    
      TotPower = np.sum(Power) # Actually mean, because we already divided by numdraws
      # Calculate square root of variance, with Bessel's correction
      TotPowerStdDev = np.sqrt(np.sum((Power*numdraws - TotPower)**2.0)/(numdraws-1))
      
      power_per_m2[fc_idx,xcnt] = TotPower/(xstep*np.pi*x/2.0)
      power_per_m2_stddev[fc_idx,xcnt] = TotPowerStdDev/(xstep*np.pi*x/2.0)
    
      pass

  if unwrapflag:
    # if we added another axis to friction_coefficient,
    # remove it from power_per_m2, etc. now.
    power_per_m2=power_per_m2[0,:]
    power_per_m2_stddev=power_per_m2_stddev[0,:]

    vibration_ampl=vibration_ampl[0,:]
    pass

  
  if (doplots): # NOTE: plots only compatible with non-vectorized friction coefficient
    from matplotlib import pyplot as pl
    #betarange=np.linspace(-np.pi,np.pi,800)
    #pl.figure()
    #pl.clf()
    #pl.plot(betarange*180.0/np.pi,beta_pdf(betarange),'-')
    #pl.xlabel('Facet orientation (degrees from flat)')
    #pl.ylabel('Probability density (/rad)')
    ##pl.savefig('/tmp/facet_pdf.png',dpi=300)
    
    
    pl.figure()
    pl.clf()
    pl.plot(xrange*1e3,power_per_m2/1.e3,'-')
    pl.xlabel('Position (mm)')
    pl.ylabel('Heating power (kW/m^2)')
    #pl.savefig('/tmp/frictional_heating.png',dpi=300)
    #pl.show()
    pass

  return (power_per_m2,power_per_m2_stddev,vibration_ampl)



def integrate_power(xrange,power_per_m2,power_per_m2_stddev=None):
  # integrate power over half of a penny shaped crack... applies
  # over last axis
  dx=abs(xrange[1]-xrange[0])
  slice_area = dx * np.pi*abs(xrange)/2.0 
  totalpower = np.sum(power_per_m2*slice_area,axis=len(power_per_m2.shape)-1)

  if power_per_m2_stddev is not None:
    totalpower_stddev = np.sqrt(np.sum(power_per_m2_stddev**2.0 * slice_area**2.0,axis=len(power_per_m2.shape)-1))
    return (totalpower,totalpower_stddev)
    pass
  else:
    return totalpower
  pass
  
