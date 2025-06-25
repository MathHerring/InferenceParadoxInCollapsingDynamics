import numpy as np
import scipy as sp
import os
import pickle


################# NEED TO INITIALIZE THESE FUNCTION SO PICKLE CAN LOAD THE SAMPLER!!!! REAL phasediagram code at the end.

######### FORCES
def force_func(gam, D, alpha, x):
    """ general expression for TSA force. """
    eps = 0.05
    if(alpha>(-1+eps) ):
        y = - (2*x**(1 + alpha) *gam)/(D*(1 + alpha))           
        denom = x*sp.hyp1f1(1,1 + 1/(1 + alpha) , y)
        force = D / denom - gam*x**alpha
    elif(alpha< (-1 -eps) ):
        y = - (2*x**(1 + alpha) *gam)/(D*(1 + alpha))
        denom = x*sp.hyperu(1,1 + 1/(1 + alpha) , y)
        force = - D*(alpha+1) /denom - gam*x**alpha
    else:
        force = - alpha*D/x + gam*x**alpha
    return force

def RW_force(D, x):
    """ TSA force for RW process (alpha = egal, gamma =0) """
    force = D/x
    return force

def force_func_beta(gam, D, alpha, beta, x):
    """ general expression for TSA force including beta. """
    """same as without beta but according to trafo f_beta = x**beta * f_alpha(L, alpha-beta), see SI mult"""
    alphaminusbeta = alpha-beta
    eps = 0.05
    if(alphaminusbeta > (-1+eps) ):
        y = - (2*x**(1 + alphaminusbeta) *gam)/(D*(1 + alphaminusbeta))           
        denom = x*sp.hyp1f1(1,1 + 1/(1 + alphaminusbeta) , y)
        force = D / denom - gam*x**alphaminusbeta
    elif(alphaminusbeta < (-1 -eps) ):
        y = - (2*x**(1 + alphaminusbeta) *gam)/(D*(1 + alphaminusbeta))
        denom = x*sp.hyperu(1,1 + 1/(1 + alphaminusbeta) , y)
        force = - D*(alphaminusbeta+1) /denom - gam*x**alphaminusbeta
    else:
        force = - alphaminusbeta*D/x + gam*x**alphaminusbeta
    return force * x**beta

def RW_force_beta(D, beta, x):
    """ TSA force for RW process (alpha = egal, gamma =0) """
    force = D * x**(beta-1)
    return force

def bessel_force(gam, D, x):
    """ TSA force for bessel process (alpha = -1) """
    force = (D+gam)/x
    return force

def OU_force(gam, D, x):
    """ TSA force for Ornstein-Uhlenbeck process (alpha = 1) """
    force = -gam*x + np.sqrt(gam*D)/sp.dawsn(x * np.sqrt(gam/D))
    return force

def drift_force(gam, D, x):
    """ TSA force for advection process (alpha = 0) """
    expon = np.exp(-(2*gam*x)/D )
    nl_fac = (2*gam ) /(1.0- expon) 
    force = -gam + nl_fac
    return force

def neg_two_force(gam, D, x):
    """ TSA force for alpha = -2 """
    y = 2*gam/(D*x)
    force = D/x+ 2*gam*sp.expn(1,y)/(x**2 * sp.expn(2,y) ) - gam/x**2
    
    # for numerical stability treat values y>100 separately
    if np.any(y>100):
        mask = np.where(y>100)
        force[mask] = D/x[mask] + gam /x[mask]**2
    return force

def force_discrete(gamma, D, alpha, x):
    """ specifiying which force function is used for given alpha """
    if alpha == 1.0:
        force = OU_force(gamma, D, x)
        return force
    elif alpha == 0.0:
        force = drift_force(gamma, D, x)
        return force
    elif alpha == -1.0:
        force = bessel_force(gamma, D, x)
        return force
    elif alpha == -2.0:
        force = neg_two_force(gamma, D, x)
        return force
    else:
        print('alpha value is not within the set of values used in the cytokinesis example.')



########## PRIORS and LOGL


def prior_transform_continuous_alpha(u):
    """ Transforms unit cube samples u to a flat prior between the specified values in each variable """
    ustar = u*np.array([0.2,0.1,1.8]) + np.array([-0.05, 0.01, -0.9])
    
    return ustar

def loglikelihood_array_continuous_alpha(p, x, dx, dt):
    """ path likelihood function """
    
    gamma = p[0]
    D = p[1]
    alpha = p[2]
    
    var_p = 2*D*dt
    norm = np.sqrt(np.pi * var_p)    
    singleLogLike_ar = - ((dx - force_func(gamma, D, alpha, x)*dt)**2) / var_p - np.log(norm)      
    singleLogLike = np.nansum(singleLogLike_ar)
    
    if np.isnan(singleLogLike) or np.isinf(singleLogLike):
        singleLogLike = -np.inf
    
    return singleLogLike

def loglikelihood_array_discretealpha_RW(p, x, dx, dt, alpha):
    """ path likelihood function """
    D = p[0]
    
    var_p = 2*D*dt
    norm = np.sqrt(np.pi * var_p)    
    singleLogLike_ar = - ((dx - RW_force(D, x)*dt)**2) / var_p - np.log(norm)      
    singleLogLike = np.nansum(singleLogLike_ar)
    
    if np.isnan(singleLogLike) or np.isinf(singleLogLike):
        singleLogLike = -np.inf
    
    return singleLogLike

def loglikelihood_array_discretealpha_genforce(p, x, dx, dt, alpha):
    """ path likelihood function """
    
    gamma = p[0]
    D = p[1]
    
    var_p = 2*D*dt
    norm = np.sqrt(np.pi * var_p)    
    singleLogLike_ar = - ((dx - force_func(gamma, D, alpha, x)*dt)**2) / var_p - np.log(norm)      
    singleLogLike = np.nansum(singleLogLike_ar)
    
    if np.isnan(singleLogLike) or np.isinf(singleLogLike):
        singleLogLike = -np.inf
    return singleLogLike

def prior_transform_discretealpha_RW(u):
    """ Transforms unit cube samples u to a flat prior between the specified values in each variable """
    ustar = u*np.array([0.4]) + np.array([0.1])
    return ustar

def prior_transform_discretealpha_genforce(u):
    """ Transforms unit cube samples u to a flat prior between the specified values in each variable """
    ustar = u*np.array([2,0.4]) + np.array([-0.5, 0.1])
    return ustar


############## logLs for Beta case


def loglikelihood_array_discretealpha_genforce_beta(p, x, dx, dt, alpha, beta):
    """ path likelihood function """
    
    gamma = p[0]
    D = p[1]
    
    var_p = x**beta * 2*D*dt
    norm = np.sqrt(np.pi * var_p)    
    singleLogLike_ar = - ((dx - force_func_beta(gamma, D, alpha, beta, x)*dt)**2) / var_p - np.log(norm)      
    singleLogLike = np.nansum(singleLogLike_ar)
    
    if np.isnan(singleLogLike) or np.isinf(singleLogLike):
        singleLogLike = -np.inf
    return singleLogLike


def loglikelihood_array_discretealpha_RW_beta(p, x, dx, dt, alpha, beta):
    """ path likelihood function """
    D = p[0]
    alp=alpha
    
    var_p = x**beta * 2*D*dt
    norm = np.sqrt(np.pi * var_p)    
    singleLogLike_ar = -((dx - RW_force_beta(D, beta, x)*dt)**2) / var_p - np.log(norm)     
    #singleLogLike_ar = - ((dx - RW_force(D, x))/(2*np.sqrt(D)))**2 *dt - np.log(norm)    
    singleLogLike = np.nansum(singleLogLike_ar)
    
    if np.isnan(singleLogLike) or np.isinf(singleLogLike):
        singleLogLike = -np.inf
    return singleLogLike


def prior_transform_discretealpha_RW_beta(u):
    """ Transforms unit cube samples u to a flat prior between the specified values in each variable """
    ustar = u*np.array([0.4]) + np.array([0.1])
    return ustar

def prior_transform_discretealpha_genforce_beta(u):
    """ Transforms unit cube samples u to a flat prior between the specified values in each variable """
    ustar = u*np.array([5,0.4]) + np.array([-0.5, 0.1])
    return ustar



def analyze_single_ns(path_data, whichinference, alpha, beta, gamma, D, nrealiz, t_inf, t_inf_length):    
    '''take care: you have to load the function for likelihood and prior_transform, because they are part of the sampler object, but itself not saved by pickle.'''

    file = os.path.join(path_data, 'sampler_opaquetrans_bessel_type{}_gamma{}_D{}_alpha{}_beta{}_N{}_tstart{}_tlength{}.pickle'.format(whichinference, int(gamma*10),int(D*10), int(alpha*10), int(beta*10), nrealiz, int(t_inf*1000), int(t_inf_length*1000)))

    if os.path.isfile(file):
        #print(file)
        try:
            with open(file, 'rb') as handle:
                sampler_load = pickle.load(handle)            

            res = sampler_load.results
            evi = res.logz[-1]
            evierr = res.logzerr[-1]
        except:
            print('couldnt open file')
            evi = 1111
            evierr = 1111
    else:
        print('file does not exist: ', file)
        evi = 1111
        evierr = 1111
    
    return evi, evierr










################################## HERE STARTS THE EXTRACTION OF THE EVIDENCE
'''
This script loads data from nested sampling sampler objects and saves evidence difference.
'''

gamma = 1
D = 0.2
beta = 1
alpha= 0
nrealiz = 1000

t_inf_length = 0.15


#inferences = ['random_walk', 'general_force']
#path_data = "D:\\Forschung\\PapersAndWriting\\SDEmitNico\\multnoise\\inference\\clusterresults"
path_data = '/scratch/users/mhaerin1/evidence_phasetransition/opaquetransition/results'
path_save = '/scratch/users/mhaerin1/evidence_phasetransition/opaquetransition/results'

'''
t_starts = np.array([0.01 , 0.015, 0.02 , 0.025, 0.03 , 0.035, 0.04 , 0.045, 0.05 ,
       0.055, 0.06 , 0.065, 0.07 , 0.075, 0.08 , 0.085, 0.09 , 0.095,
       0.1  , 0.105, 0.11 , 0.115, 0.12 , 0.125, 0.13 , 0.135, 0.14 ,
       0.145, 0.15 , 0.155, 0.16 , 0.165, 0.17 , 0.175, 0.18 , 0.185,
       0.19 , 0.195, 0.2  , 0.205, 0.21 , 0.215, 0.22 , 0.225, 0.23 ,
       0.235, 0.24 , 0.245, 0.25 , 0.255, 0.26 , 0.265, 0.27 , 0.275,
       0.28 , 0.285, 0.29 , 0.295, 0.3  , 0.305, 0.31 , 0.315, 0.32 ,
       0.325, 0.33 , 0.335, 0.34 , 0.345, 0.35 , 0.355, 0.36 , 0.365,
       0.37 , 0.375, 0.38 , 0.385, 0.39 , 0.395, 0.4  , 0.405, 0.41 ,
       0.415, 0.42 , 0.425, 0.43 , 0.435, 0.44 , 0.445, 0.45 , 0.455,
       0.46 , 0.465, 0.47 , 0.475, 0.48 , 0.485, 0.49 , 0.495, 0.5  ,
       0.505, 0.51 , 0.515, 0.52 , 0.525, 0.53 , 0.535, 0.54 , 0.545,
       0.55 , 0.555, 0.56 , 0.565, 0.57 , 0.575, 0.58 , 0.585, 0.59 ,
       0.595, 0.6  , 0.605, 0.61 , 0.615, 0.62 , 0.625, 0.63 , 0.635,
       0.64 , 0.645, 0.65 , 0.655, 0.66 , 0.665, 0.67 , 0.675, 0.68 ,
       0.685, 0.69 , 0.695, 0.7  , 0.705, 0.71 , 0.715, 0.72 , 0.725,
       0.73 , 0.735, 0.74 , 0.745, 0.75 , 0.755, 0.76 , 0.765, 0.77 ,
       0.775, 0.78 , 0.785, 0.79 , 0.795, 0.8  , 0.805, 0.81 , 0.815,
       0.82 , 0.825, 0.83 , 0.835, 0.84 , 0.845, 0.85 , 0.855, 0.86 ,
       0.865, 0.87 , 0.875, 0.88 , 0.885, 0.89 , 0.895, 0.9  , 0.905,
       0.91 , 0.915, 0.92 , 0.925, 0.93 , 0.935, 0.94 , 0.945, 0.95 ,
       0.955, 0.96 , 0.965, 0.97 , 0.975, 0.98 , 0.985, 0.99 , 0.995,
       1.   ])
'''

t_starts = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1 , 0.11,
       0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2 , 0.21, 0.22,
       0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3 , 0.31, 0.32, 0.33,
       0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4 , 0.41, 0.42, 0.43, 0.44,
       0.45, 0.46, 0.47, 0.48, 0.49, 0.5 , 0.51, 0.52, 0.53, 0.54, 0.55,
       0.56, 0.57, 0.58, 0.59, 0.6 , 0.61, 0.62, 0.63, 0.64, 0.65, 0.66,
       0.67, 0.68, 0.69, 0.7 , 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77,
       0.78, 0.79, 0.8 , 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88,
       0.89, 0.9 , 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99,
       1.  , 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1 ,
       1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19, 1.2 , 1.21,
       1.22, 1.23, 1.24, 1.25, 1.26, 1.27, 1.28, 1.29, 1.3 , 1.31, 1.32,
       1.33, 1.34, 1.35, 1.36, 1.37, 1.38, 1.39, 1.4 , 1.41, 1.42, 1.43,
       1.44, 1.45, 1.46, 1.47, 1.48, 1.49, 1.5 , 1.51, 1.52, 1.53, 1.54,
       1.55, 1.56, 1.57, 1.58, 1.59, 1.6 , 1.61, 1.62, 1.63, 1.64, 1.65,
       1.66, 1.67, 1.68, 1.69, 1.7 , 1.71, 1.72, 1.73, 1.74, 1.75, 1.76,
       1.77, 1.78, 1.79, 1.8 , 1.81, 1.82, 1.83, 1.84, 1.85, 1.86, 1.87,
       1.88, 1.89, 1.9 , 1.91, 1.92, 1.93, 1.94, 1.95, 1.96, 1.97, 1.98,
       1.99, 2.  ])


gamma = 1.0
t_inf = 0.1
#t_inf_length = 0.15
phdg_opaque = np.zeros(len(t_starts))
phdg_opaque[:] = np.nan

for i in np.arange(len(t_starts)):
    print(i)

    t_inf_length = t_starts[i]
    #t_inf = t_starts[i]

    evi_rw, evierr_rw = analyze_single_ns(path_data, 'random_walk', alpha, beta, gamma, D, nrealiz, t_inf, t_inf_length)
    evi_gen, evierr_gen = analyze_single_ns(path_data, 'general_force', alpha, beta, gamma, D, nrealiz, t_inf, t_inf_length)

    if evi_rw == 1111 or evi_gen == 1111:
        continue
    
    #print('juhu')

    phdg_opaque[i] = evi_gen-evi_rw


np.savetxt(os.path.join(path_save, "evidence_opaque_largerinterval_WF.csv"), phdg_opaque, delimiter=",")