import scipy 
import numpy as np
from matplotlib import pyplot as pl
from scipy.integrate import quad
from scipy.optimize import newton
from numpy.random import rand


from crackclosuresim2 import solve_normalstress
from crackclosuresim2 import solve_shearstress



def angled_friction_model(x_bnd,xrange,xstep,
                          numdraws,
                          E,nu,
                          sigma_yield,tau_yield,
                          friction_coefficient,
                          closure_stress,
                          beta_drawfunc,
                          a_crack,
                          static_load,
                          vib_normal_stress_ampl,
                          vib_shear_stress_ampl,
                          vibration_frequency,
                          crack_model_normal,
                          crack_model_shear,
                          verbose,doplots):

  power_per_m2 = np.zeros((xrange.shape[0]),dtype='d')
  vibration_ampl = np.zeros((xrange.shape[0]),dtype='d')
  numsteps=xrange.shape[0]

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
    closure_state_x = closure_stress[xcnt]
    
    # Evaluate closure state on both sides of static load
    
    (closure_point_sub, sigma_sub, tensile_displ_sub) = solve_normalstress(xrange,x_bnd,closure_stress,xstep,static_load-vib_normal_stress_ampl,a_crack,sigma_yield,crack_model_normal,calculate_displacements=True)
    
    
    (closure_point_add, sigma_add, tensile_displ_add) = solve_normalstress(xrange,x_bnd,closure_stress,xstep,static_load+vib_normal_stress_ampl,a_crack,sigma_yield,crack_model_normal,calculate_displacements=True)
    
    
    # Evaluate at x
    if x <= closure_point_sub:
      closure_state_sub_x=0.0
      pass
    else:
      closure_state_sub_x = sigma_sub[xcnt]
      pass
    
    if x <= closure_point_add:
      closure_state_add_x=0.0
      pass
    else:
      closure_state_add_x = sigma_add[xcnt]
      pass

    if verbose:
      print("sigma_sub[%d]=%f; sigma_add[%d]=%f" % (xcnt,sigma_sub[xcnt],xcnt,sigma_add[xcnt]))
      pass
    
    # uyy is double calculated value because each side moves by this much
    uyy_add = tensile_displ_add[xcnt]*2.0

    uyy_sub = tensile_displ_sub[xcnt]*2.0

    # sigma_sub and sigma_add are positive tensile representations of the
    # normal tractions on the crack surfaces.
    # The closure stress in each of these states therefore would be
    # -sigma_sub and -sigma_add respectively. 
    
    # ss variables are for shear_stickslip calculations
    (effective_length_sub, tau_sub, shear_displ_sub) = solve_shearstress(xrange,x_bnd,-sigma_sub,xstep,vib_shear_stress_ampl,a_crack,friction_coefficient,tau_yield,crack_model_shear)
    
    (effective_length_add, tau_add, shear_displ_add) = solve_shearstress(xrange,x_bnd,-sigma_add,xstep,vib_shear_stress_ampl,a_crack,friction_coefficient,tau_yield,crack_model_shear)
    
    
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
    
    # represent stress field along crack as a complex number,
    # real part is shear, imaginary part is normal stress (following
    # Evans and Hutchinson Q+iP) except in our case these represent
    # vibration amplitudes as vertical and horizontal forces on a crack
    # segment 
    #
    # Throughout each vibration cycle,
    # stresses cycle between  closure_state_add and closure_state_sub
    
    # So based on temporary limitation above
    # (no external shear loading applied), Q=0
    # P and Q are FORCES... treat them as acting on quarter-annulus
    
    # Nominal P and Q are the external loading on this xstep unit
    # that we are trying to match
    # ... We match P (normal)  and don't worry too much about Q (shear)
    Q_static_nominal=0.0
    P_static_nominal=closure_state_x * xstep * np.pi*x/2.0 # A force, total for all of the little contributions from each of the draws
    
    # For each draw, transform nominal Q and P to 
    # Normal and shear forces on the sliding point.
    N_static_nominal=P_static_nominal*np.cos(beta_draws)+Q_static_nominal*np.sin(beta_draws)
    # T_static_nominal = -P_static_nominal*np.sin(beta_draws) + Q_static_nominal*np.cos(beta_draws)
    # Set T_static_nominal to zero because we don't expect any long-term
    # static shear on the asperity contacts
    T_static_nominal=0.0
    
    # Transform back to P and Q... sum contributions from all draws
    P_contributions=np.sum(N_static_nominal*np.cos(beta_draws) - T_static_nominal*np.sin(beta_draws))
    #Q_contributions=np.sum(N_static_nominal*np.sin(beta_draws) + T_static_nominal*np.cos(beta_draws))
    
    # Determine scaling factor for all draws to sum to desired value
    if P_contributions==0.0: # Avoid warning message
      normal_force_factor=np.inf
      pass
    else:
      normal_force_factor=P_static_nominal/P_contributions
      pass
    
    if not np.isfinite(normal_force_factor):
        normal_force_factor=1.0  # in case P_contributions are 0 just set scaling factor to 1.0
        pass
    
    
    # Evaluate P,Q, N, & T assuming this scaling factor
    #P_static = P_static_nominal*normal_force_factor
    #Q_static = Q_static_nominal*normal_force_factor
    
    #N_static = P_static*np.cos(beta_draws)+Q_static*np.sin(beta_draws)
    #T_static = -P_static*np.sin(beta_draws)+Q_static*np.cos(beta_draws)
    N_static = N_static_nominal*normal_force_factor
    T_static = T_static_nominal*normal_force_factor  # 0
    
    P_static = N_static*np.cos(beta_draws) - T_static*np.sin(beta_draws)
    Q_static = N_static*np.sin(beta_draws) + T_static*np.cos(beta_draws)


    # NOTE:
    # N_static, T_static, P_static, and Q_static
    # are total for the entire band.
    # P_dynamic, Q_dynamic N_dynamic, and T_dynamic
    # will be __per_draw__
    
    # P_dynamic per draw
    P_dynamic = (closure_state_add_x - closure_state_sub_x)* (xstep * np.pi*x/2.0)/(2.0*numdraws) # dynamic stress amplitude

    # For the moment, just make the shear correct near the surface
    # because that's where we have data
    #Q_dynamic = 0.0 # Should be shear vibration amplitude
    # Per number of draws because we assume that the stress
    # is distributed over the asperities.
    #Q_dynamic = vib_shear_stress_ampl*xstep*np.pi*r/(2.0*numdraws)
    Q_dynamic = ((tau_add[xcnt]+tau_sub[xcnt])/2.0)*xstep*np.pi*x/(2.0*numdraws)
    
    N_dynamic = P_dynamic*np.cos(beta_draws)+Q_dynamic*np.sin(beta_draws)
    T_dynamic = -P_dynamic*np.sin(beta_draws)+Q_dynamic*np.cos(beta_draws)
    
    
    # exp( iBeta) = cos(beta)+i*sin(Beta)
    # (Q+iP)*exp(iBeta) = Qcos(beta)-Psin(Beta) + i(Qsin(beta) + Pcos(beta))
    # N =Im( (Q+iP)exp(iBeta) )
    # T =Re( (Q+iP)exp(iBeta) )
    # omega=(np.pi/2-beta_draws)-atan(friction_coefficient)
    
    # Locking condition: T < mu*N...  here T=T_dynamic, N=N_static-N-dynamic
    ## i.e. Re( (Q+iP)*exp(iBeta)) < mu*Im( (Q+iP)*exp(iBeta))
    ## i.e. Re( (Q+iP)*exp(iBeta))/Im( (Q+iP)*exp(iBeta)) < mu
    ## i.e. cot(angle((Q+iP)*exp(iBeta))) < mu
    ## i.e. acot(cot(angle((Q+iP)*exp(iBeta)))) > acot(mu)
    ## OR acot(cot(angle((Q+iP)*exp(iBeta)))) > acot(mu)-pi
    ## i.e. angle((Q+iP)*exp(iBeta)) > acot(mu) > 0
    ##  OR 0 > angle((Q+iP)*exp(iBeta)) > acot(mu)-pi
    
    #ang=np.angle((Q + i*P)*np.exp(i*beta_draws))
    #slip = (((ang > 0) & (ang  > np.arctan(1/friction_coefficient))) |
    #        ((ang < 0) & (ang > np.arctan(1/friction_coefficient)-np.pi)))
    # sdh 11/6/18 .... change N_static-abs(N_dynamic) to N_static + abs(N_dynamic)
    # because slip occurs in easier case where compressive (negative) N_static
    # is opposed by N_dynamic
    # sdh 11/6/18 N_static is total load... must be divided by numdraws
    # to be comparable to N_dynamic and or T_dynamic
    slip=np.abs(T_dynamic) >=  -friction_coefficient*(N_static/numdraws+np.abs(N_dynamic))

    utt = (shear_displ_add[xcnt] + shear_displ_sub[xcnt])/2.0
    PP_vibration_y=uyy_add-uyy_sub
    vibration_ampl[xcnt]=PP_vibration_y/2.0
    #PP_vibration_t=utt*2.0
    tangential_vibration_ampl=np.abs(vibration_ampl[xcnt] * np.sin(beta_draws) + utt*np.cos(beta_draws))*slip
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
    # sdh 11/6/18 N_static is total load... must be divided by numdraws
    # to be comparable to N_dynamic and or T_dynamic
    if x >= closure_point_sub:
        Power = 0.5 * (friction_coefficient*(np.abs(N_static)/numdraws+np.abs(N_dynamic)))*tangential_vibration_velocity_ampl
        pass
    else:
        Power=0.0
        pass
    
    TotPower = np.sum(Power)
    
    power_per_m2[xcnt] = TotPower/(xstep*np.pi*x/2.0)
      
    pass


  if (doplots):
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

  return (power_per_m2,vibration_ampl)



