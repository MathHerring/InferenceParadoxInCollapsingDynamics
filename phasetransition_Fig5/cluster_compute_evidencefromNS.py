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
    """ TSA force for RW process (alpha = egal, gamma =0) """
    force = D/x
    return force

def force_func_beta(gam, D, alpha, beta, x):
    """ general expression for TSA force including beta. """
    """same as without beta but according to trafo f_beta = x**beta * f_alpha(L, alpha-beta), see SI"""
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
    ustar = u*np.array([2,0.4]) + np.array([-0.5, 0.1])
    return ustar




####################### MAIN INFERENCE FUNCTION

def run_inference(whichinference, data, rstate, alpha, beta, gamma, D, nrealiz, savepath):
    if whichinference == 'random_walk':
        ndim = 1
        Nlivepoints = 200
        sampler = dynesty.NestedSampler(loglikelihood_array_discretealpha_RW_beta, prior_transform_discretealpha_RW_beta, 
                                ndim, nlive=Nlivepoints,
                                rstate=rstate, 
                                logl_args=data)

        sampler.run_nested()
        if path_save != 'nosave':
            with open(os.path.join(savepath, 'sampler_phasetrans_type{}_gamma{}_D{}_alpha{}_beta{}_N{}.pickle'.format(whichinference, int(gamma*10),int(D*10), int(alpha*10), int(beta*10), nrealiz)), 'wb') as handle:
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
            with open(os.path.join(savepath, 'sampler_phasetrans_type{}_gamma{}_D{}_alpha{}_beta{}_N{}.pickle'.format(whichinference, int(gamma*10),int(D*10), int(alpha*10), int(beta*10), nrealiz)), 'wb') as handle:
                    pickle.dump(sampler, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print('ERROR: neither random_walk nor general_force selected')
            


'''
This script reads externally provided parameters to run inference on a specified ensemble of stochastic trajectories.
The trajectory data would have to be generated separately and saved. The script to generate the trajectories should be bundled together with this script in the same folder.
The resulting sampler object from the nested sampling run is saved for analysis.
'''

### get external options:
# "path_data" - path to trajectories
# "path_save" - path to save sampler object
# "type_inference" - specify whether random walk or general inference is wanted
# parameters of TSA ensemble: (alpha, beta, gamma, D, timestep, number of trajectories)
path_data, path_save, type_inference, alpha, beta, gamma, D, dt, nrealiz = sys.argv[1:]

alpha = float(alpha)
gamma = float(gamma)
D = float(D)
beta = float(beta)
nrealiz = int(nrealiz)
dt = float(dt)

### load data
datastring = "trajectories_gamma{}_D{}_alpha{}_beta{}_N{}.txt".format(int(gamma*10),int(D*10), int(alpha*10), int(beta*10), nrealiz)
loadts = np.loadtxt(os.path.join(path_data, datastring))
tsarr = loadts.T
x1 = np.roll(tsarr,-1,0)
dx = x1[:-1]-tsarr[:-1]
x = tsarr[:-1]

### run inference and save sampler object
timestep = dt
data = [x[50:500,:1500], dx[50:500,:1500], timestep, alpha, beta]

run_inference(type_inference, data, rstate, alpha, beta, gamma, D, nrealiz, path_save)
