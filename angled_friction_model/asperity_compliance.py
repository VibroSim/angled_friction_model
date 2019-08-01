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

f_theta = lambda theta,mu,sigma: (1.0/(np.sqrt(2.0*np.pi*sigma**2.0)))*np.exp(-(theta-mu)**2.0/(2.0*sigma**2.0))

integrand = lambda theta,R,Estar,G,mu,sigma: ( (9.0/(16.0*np.cos(theta)*Estar**2.0))**(1.0/3.0)*np.cos(theta)**2.0 + (9.0/(16.0*np.cos(theta)*G**2.0))**(1.0/3.0)*np.sin(theta)**2.0)**(-3.0/2.0) * R**(1.0/2.0)*f_theta(theta,mu,sigma)

Estar = lambda E,nu: 1.0/(2.0*(1.0-nu**2.0)/E)

# Reasonable numbers....
nu=0.33
E=207.83e9
G=E/(2*(1+nu))
R=15e-6
mu=0.0
sigma=30.0*np.pi/180.0

m=1000.0e6  # asperites/m^2


H=2.0*scipy.integrate.quad(lambda theta: integrand(theta,R,Estar(E,nu),G,mu,sigma),0,np.pi/2.1)[0] # We don't integrate quite up to pi/2 because of convergence issues with zero denominator...

Hm = H*m
print("Hm =",Hm,"Pa/m^(3/2)") 
