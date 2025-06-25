import time, sys, os
import numpy as np
from numpy import linalg
from matplotlib import pyplot as plt
rstate = np.random.default_rng(5647)
import dynesty
import pandas as pd
import random as rd
import pickle

from dynesty import plotting as dyplot
from dynesty import DynamicNestedSampler
import scipy.special as sp



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
    """ TSA force for RW process (alpha = any, gamma =0) """
    force = D/x
    return force

def force_func_beta(gam, D, alpha, beta, x):
    """ general expression for TSA force including ultiplicative noise exponent beta. """
    """ same as without beta but according to trafo f_beta = x**beta * f_alpha(L, alpha-beta), see SI"""
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
    """ TSA force for RW process (alpha = any, gamma =0) """
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



########## PRIOR-DISTRIBUTIONS and LOG-LIKELIHOOD

def prior_transform_continuous_alpha(u):
    """ Transforms unit cube samples u to a flat prior between the specified values in each variable """
    ustar = u*np.array([0.2,0.1,1.8]) + np.array([-0.05, 0.01, -0.9])
    return ustar

def loglikelihood_array_continuous_alpha(p, x, dx, dt):
    """ path likelihood function for inferring gamma, D and alpha"""
    
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
    """ path likelihood function only inferring D (gamma=0)"""
    D = p[0]
    
    var_p = 2*D*dt
    norm = np.sqrt(np.pi * var_p)    
    singleLogLike_ar = - ((dx - RW_force(D, x)*dt)**2) / var_p - np.log(norm)      
    singleLogLike = np.nansum(singleLogLike_ar)
    
    if np.isnan(singleLogLike) or np.isinf(singleLogLike):
        singleLogLike = -np.inf
    
    return singleLogLike

def loglikelihood_array_discretealpha_genforce(p, x, dx, dt, alpha):
    """ path likelihood function for specified alpha"""
    
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


############## logLs for multiplicative (Beta) case


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
    ustar = u*np.array([2,0.4]) + np.array([-0.5, 0.1])
    return ustar




### MAIN INFERENCE FUNCTION
def run_inference(whichinference, data, rstate, alpha, beta, gamma, D, nrealiz, t_inf, t_inf_length, savepath):
    if whichinference == 'random_walk':
        ndim = 1
        Nlivepoints = 200
        sampler = dynesty.NestedSampler(loglikelihood_array_discretealpha_RW_beta, prior_transform_discretealpha_RW_beta, 
                                ndim, nlive=Nlivepoints,
                                rstate=rstate, 
                                logl_args=data)

        sampler.run_nested()
        if path_save != 'nosave':
            with open(os.path.join(savepath, 'sampler_opaquetrans_bessel_type{}_gamma{}_D{}_alpha{}_beta{}_N{}_tstart{}_tlength{}.pickle'.format(whichinference, int(gamma*10),int(D*10), int(alpha*10), int(beta*10), nrealiz, int(t_inf*1000), int(t_inf_length*1000))), 'wb') as handle:
                    pickle.dump(sampler, handle, protocol=pickle.HIGHEST_PROTOCOL)
    elif whichinference == 'general_force': 
        ndim = 2
        Nlivepoints = 200
        sampler = dynesty.NestedSampler(loglikelihood_array_discretealpha_genforce_beta, prior_transform_discretealpha_genforce_beta, 
                                ndim, nlive=Nlivepoints,
                                rstate=rstate, 
                                logl_args=data)
        sampler.run_nested()
        if path_save != 'nosave':
            with open(os.path.join(savepath, 'sampler_opaquetrans_bessel_type{}_gamma{}_D{}_alpha{}_beta{}_N{}_tstart{}_tlength{}.pickle'.format(whichinference, int(gamma*10),int(D*10), int(alpha*10), int(beta*10), nrealiz, int(t_inf*1000), int(t_inf_length*1000))), 'wb') as handle:
                    pickle.dump(sampler, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print('ERROR: neither random_walk nor general_force selected')
            


###### EXECUTING THE INFERENCE

### run the script by executing the following command in a console:
### python cluster_inference_opaquetransition.py X XX
### where X = t_inf and XX = t_inf_length are the start and length of the inference window (start was)

### get external options
t_inf, t_inf_length = sys.argv[1:]

t_inf = float(t_inf)
t_inf_length = float(t_inf_length)

# parameters (only needed for saving string)
alpha = 0
beta = 1
gamma = 1
D = 0.2
nrealiz = 1000
timestep = 0.0005

# specfiy paths for saving and for loading the ensemble data
path_data = ''
path_save = ''

### load the respective ensemble data to perform inference on
#datastring = "ensemble_feller_alpha1_beta0_dt00005_IC10_N1000_gam1_D02.csv"   ## CIR model
#datastring = "ensemble_heston_alpha1_beta1_dt00005_IC10_N1000_gam1_D02.csv"   ## Vasicek model
#datastring = "ensemble_bessel_alpha-1_beta0_dt00005_IC10_N1000_gam1_D02.csv"  ## Bessel model
datastring = "ensemble_WF_alpha0_beta1_dt00005_IC10_N1000_gam1_D02.csv"        ## Wright-Fisher model

# prepare ensemble data for inference
tsarr = np.loadtxt(os.path.join(path_data, datastring), delimiter=',')
tsarr = tsarr[:,~np.isnan(tsarr[1,:])]
x1 = np.roll(tsarr,-1,0)
dx = x1[:-1]-tsarr[:-1]
x = tsarr[:-1]

idx_start = int(t_inf/timestep)
idx_end = idx_start + int(t_inf_length/timestep)

data = [x[idx_start:idx_end,:], dx[idx_start:idx_end,:], timestep, alpha, beta]


### runs the inference for the general model and the fixed-gamma model (gamma=0)
### saves the full sampler object in path_save
run_inference('general_force', data, rstate, alpha, beta, gamma, D, nrealiz, t_inf, t_inf_length, path_save)
run_inference('random_walk', data, rstate, alpha, beta, gamma, D, nrealiz, t_inf, t_inf_length, path_save)
