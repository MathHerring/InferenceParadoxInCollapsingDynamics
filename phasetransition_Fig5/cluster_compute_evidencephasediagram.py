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



def analyze_single_ns(path_data, whichinference, alpha, beta, gamma, D, nrealiz):    
    '''take care: you have to load the function for likelihood and prior_transform, because they are part of the sampler object, but itself not saved by pickle.'''

    file = os.path.join(path_data, 'sampler_phasetrans_type{}_gamma{}_D{}_alpha{}_beta{}_N{}.pickle'.format(whichinference, int(gamma*10),int(D*10), int(alpha*10), int(beta*10), nrealiz))

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






################################## HERE STARTS THE ACTUAL CODE

'''
This script loads all the previously calculated nested sampling sampler objects, extracts the evidence, compares RW evidence to general model evidence and saves the result in a new array.
'''

#gamma = 1
D = 0.2
#beta = 0
nrealiz = 2000

path_data = "" # path to previously computed nested sampling sampler objects
path_save = "" # specify where to save the evidence array


alpha_ar = [-2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5 ,-0.4, -0.3 ,-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
gamma_ar = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]#[1]#[0.03, 0.31, 2.16]
D_ar = [0.2]
beta_ar = [-2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5 ,-0.4, -0.3 ,-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]#[0.5]


##### calculate evidence array for gamma - alpha
'''
beta = 0
phdg_gamalp = np.zeros((len(gamma_ar), len(alpha_ar)))
phdg_gamalp[:] = np.nan

for i in np.arange(len(gamma_ar)):
    for j in np.arange(len(alpha_ar)):
        print(i, j)

        alpha = alpha_ar[j]
        gamma = gamma_ar[i]

        evi_rw, evierr_rw = analyze_single_ns(path_data, 'random_walk', alpha, beta, gamma, D, nrealiz)
        evi_gen, evierr_gen = analyze_single_ns(path_data, 'general_force', alpha, beta, gamma, D, nrealiz)
        
        if evi_rw == 1111 or evi_gen == 1111:
            continue
        
        #print('juhu')

        phdg_gamalp[i, j] = evi_gen-evi_rw

np.savetxt(os.path.join(path_save, "phdg_gamalp.csv"), phdg_gamalp, delimiter=",")
'''

##### calculate evidence array for alpha -beta
gamma = 1.0
phdg_betalp = np.zeros((len(beta_ar), len(alpha_ar)))
phdg_betalp[:] = np.nan

for i in np.arange(len(beta_ar)):
    for j in np.arange(len(alpha_ar)):
        print(i, j)

        alpha = alpha_ar[j]
        beta = beta_ar[i]

        evi_rw, evierr_rw = analyze_single_ns(path_data, 'random_walk', alpha, beta, gamma, D, nrealiz)
        evi_gen, evierr_gen = analyze_single_ns(path_data, 'general_force', alpha, beta, gamma, D, nrealiz)

        if evi_rw == 1111 or evi_gen == 1111:
            continue
        
        #print('juhu')

        phdg_betalp[i, j] = evi_gen-evi_rw


np.savetxt(os.path.join(path_save, "phdg_betalp.csv"), phdg_betalp, delimiter=",")