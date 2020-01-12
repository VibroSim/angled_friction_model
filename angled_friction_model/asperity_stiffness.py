from __future__ import print_function

import numpy as np
import scipy
import scipy.integrate


# From soft_closure_mode.pdf,
# EFtotal/n = d^(3/2)*H
#  n = total number of asperities
#  m = asperities/m^2
# EFtotal*m/n = d^(3/2)*H*m = force/area

# H = integral_theta (1/n) sum_i(S'(theta_i)^(-3/2)) f_theta(theta) dtheta
# Given that this is an expectation already the averaging over n samples
# should be unnecessary
# H = integral_theta (S'(theta)^(-3/2)) f_theta(theta) dtheta
#
# Where S'(theta) = [ (9/(16 cos(theta) E*^2))^(1/3)cos^2(theta) + (9/(16 cos(theta)G^2))^(1/3)sin^2(theta)]/R^(1/3)
#
# So
#H = integral_theta [ (9/(16 cos(theta) E*^2))^(1/3)cos^2(theta) + (9/(16 cos(theta)G^2))^(1/3)sin^2(theta)]^(-3/2) R^(1/2) f_theta(theta) dtheta
#
# Factor out R^(1/2)
#H = R^(1/2) integral_theta [ (9/(16 cos(theta) E*^2))^(1/3)cos^2(theta) + (9/(16 cos(theta)G^2))^(1/3)sin^2(theta)]^(-3/2) f_theta(theta) dtheta
#
# Therefore
# Hm = m*sqrt(R) * integral_theta [ (9/(16 cos(theta) E*^2))^(1/3)cos^2(theta) + (9/(16 cos(theta)G^2))^(1/3)sin^2(theta)]^(-3/2) f_theta(theta) dtheta

f_theta = lambda theta,angular_center,angular_stddev: (1.0/(np.sqrt(2.0*np.pi*angular_stddev**2.0)))*np.exp(-(theta-angular_center)**2.0/(2.0*angular_stddev**2.0))

integrand_old = lambda theta,Estar,G,angular_center,angular_stddev: ( (9.0/(16.0*np.cos(theta)*Estar**2.0))**(1.0/3.0)*np.cos(theta)**2.0 + (9.0/(16.0*np.cos(theta)*G**2.0))**(1.0/3.0)*np.sin(theta)**2.0)**(-3.0/2.0) *f_theta(theta,angular_center,angular_stddev)

integrand = lambda theta,angular_center,angular_stddev: f_theta(theta,angular_center,angular_stddev)*(np.sqrt(8)/(3.0*np.cos(theta)*((np.cos(theta) + np.tan(theta)*np.sin(theta))**(3.0/2.0))))



Estar = lambda E,nu: 1.0/(2.0*(1.0-nu**2.0)/E)


def asperity_stiffness_old(msqrtR,E,nu,angular_stddev):
    angular_center=0.0
    G=E/(2.0*(1.0+nu))

    Hm=msqrtR * 2.0*scipy.integrate.quad(lambda theta: integrand_old(theta,Estar(E,nu),G,angular_center,angular_stddev),0,np.pi/2.1)[0] # We don't integrate quite up to pi/2 because of convergence issues with zero denominator... Leading factor of 2 because we are integrating from 0..pi/2.1 not -pi/2.1...pi/2.1

    return Hm

# REPLACEMENT: Based on new logic described in soft_closure paper
def asperity_stiffness(msqrtR,E,nu,angular_stddev):
    angular_center=0.0
    # What was Hm is now known as Lm
    Lm=msqrtR * Estar(E,nu) * scipy.integrate.quad(lambda theta: integrand(theta,angular_center,angular_stddev),-np.pi/2.0,np.pi/2.0)[0] 
    return Lm



if __name__=="__main__":
    # Reasonable numbers....
    m=1000.0e6  # asperities/m^2
    R=15e-6
    E=207.83e9
    nu=0.33
    angular_stddev=30.0*np.pi/180.0    

    Lm = asperity_stiffness(m*np.sqrt(R),E,nu,angular_stddev)
    
    print("Lm =",Lm,"Pa/m^(3/2)") 
    pass
