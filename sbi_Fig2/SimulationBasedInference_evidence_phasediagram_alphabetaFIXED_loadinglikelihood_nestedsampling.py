#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals,division
import numpy as np
import os

# visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle

from sbi import analysis as analysis

# sbi
from sbi import utils as utils
from sbi.inference import SNLE, simulate_for_sbi
from sbi.inference import likelihood_estimator_based_potential
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)

import dynesty
from dynesty import plotting as dyplot

rstate = np.random.default_rng(5647)

###############################################################################
class generatePowerEnsemble(object):

    def __init__(self,fit_until_this_t=300,nrealiz=5,gamma=1.0,D=0.2,alpha=-1.0,beta=0.0, IC=5):
        
        ##################################### these parameters are passed as argument
        self.fit_until_this_t = fit_until_this_t
        self.nrealiz=nrealiz
        self.gamma = gamma
        self.D = D
        self.alpha = alpha
        self.beta = beta

        ##################################### need to set these parameters in this class
        self.var_start_sim = 0
        self.x_upper_bound = 1000 
        self.max_rtime = fit_until_this_t
        self.traFreq = 1
        self.wfreq = 50
        self.total_time=3000
        self.x0_for_reverse = 0
        self.dt = 0.005
        self.dt_eff = self.dt*self.traFreq
        self.seed = 12345
        self.x_start_sim = IC


    def run(self):
        ###########################################      
        os.system('./src_5_pureFwdPowerlaw_notrj/prog %f %d %d %f %f %f %d %d %d %d %f %f %f %f %f' % (
        self.x_start_sim,self.nrealiz,self.seed,self.dt,self.D,
        self.x0_for_reverse,self.total_time,self.wfreq,
        self.traFreq,self.max_rtime,self.x_upper_bound,
        self.var_start_sim,self.gamma,self.alpha,self.beta)
        )
        ###########################################
        
         
##############################################################################


def get_observation(params):
    """
    Returns summary statistics from full TSA dynamics.
    Summarizes the output.
    """
    gamma = params[0]
    D = params[1]
    alpha = params[2]
    beta = params[3]

    # calls the c code
    initialcond = 3
    gPE = generatePowerEnsemble(fit_until_this_t,nrealiz,gamma,D,alpha,beta, IC=initialcond )
    gPE.run()

    
    # load output of c++ simulation and save in python
    # can also load the raw trajectories and more, for info look into c code. the 'num_...' files are the c outputs. their naming should already be telling as well.
    md = np.loadtxt('num_mean.txt')
    mv = np.loadtxt('num_var.txt')

    if not list(md) or not list(mv):
        md = np.zeros(int(fit_until_this_t/dt))
        mv = np.zeros(int(fit_until_this_t/dt))
        md[:] = np.nan
        mv[:] = np.nan

    return np.concatenate((md[3:int(len(md)*0.2)], mv[3:int(len(md)*0.2)]), axis=None)


def simulation_wrapper(params):
    """
    Returns summary statistics from simulation of full TSA dynamics.
    Summarizes the output and converts it to `torch.Tensor`.
    """
    gamma = params[0]
    D = params[1]
    alpha = params[2]
    beta = params[3]
    
    # calls the c code
    initialcond = 3
    gPE = generatePowerEnsemble(fit_until_this_t,nrealiz,gamma,D,alpha,beta, IC=initialcond)
    gPE.run()

    
    # load output of c simulation and save in python
    # can also load the raw trajectories and more, for info look into c code. the 'num_...' files are the c outputs. their naming gives away the content such as moments.
    md = np.loadtxt('num_mean.txt')
    mv = np.loadtxt('num_var.txt')

    if not list(md) or not list(mv):
        md = np.zeros(int(fit_until_this_t/dt))
        mv = np.zeros(int(fit_until_this_t/dt))
        md[:] = np.nan
        mv[:] = np.nan

    summstats = torch.as_tensor(np.concatenate((md[3:int(len(md)*0.2)], mv[3:int(len(md)*0.2)]), axis=None))
    return summstats


def simulation_wrapper_RW(params):
    """
    Returns summary statistics from simulation of random walk (RW) dynamics.
    Summarizes the output and converts it to `torch.Tensor`.
    """
    D = params[0]
    beta = params[1]
    
    # calls the c code 
    initialcond=0.5
    gPE = generatePowerEnsemble(fit_until_this_t,nrealiz,0.0,D,0.0,beta, IC=initialcond)
    gPE.run()

    
    # load output of c simulation and save in python
    # can also load the raw trajectories and more, for info look into c code. the 'num_...' files are the c outputs. their naming gives away the content such as moments.
    md = np.loadtxt('num_mean.txt')
    mv = np.loadtxt('num_var.txt')

    if not list(md) or not list(mv):
        md = np.zeros(int(fit_until_this_t/dt))
        mv = np.zeros(int(fit_until_this_t/dt))
        md[:] = np.nan
        mv[:] = np.nan

    summstats = torch.as_tensor(np.concatenate((md[3:int(len(md)*0.2)], mv[3:int(len(md)*0.2)]), axis=None))
    return summstats


### simulation parameters
fit_until_this_t=1
dt=0.005
nrealiz = 1000


###### FULL DYNAMICS
prior_min = [0.1, 0.01, -2, -2]
prior_max = [2, 0.3, 2, 2]
prior = utils.torchutils.BoxUniform(
    low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max)
)


######### RANDOM WALK
prior_min_RW = [0.01, -3]
prior_max_RW = [0.3, 2]
prior_RW = utils.torchutils.BoxUniform(
    low=torch.as_tensor(prior_min_RW), high=torch.as_tensor(prior_max_RW)
)


### loading likelihood estimators
with open("likelihood_general_Nens1000_Ntrain40004000_dt0005_untiltau02.pkl", "rb") as handle:
    likelihood_estimator = pickle.load(handle)
with open("likelihood_RW_Nens1000_Ntrain40004000_dt0005_untiltau02.pkl", "rb") as handle:
    likelihood_estimator_RW = pickle.load(handle)


## get prior volume
dist = np.array(prior_max)-np.array(prior_min)
prior_volume = dist.prod()
dist = np.array(prior_max_RW)-np.array(prior_min_RW)
prior_volume_RW = dist.prod()


### now set up nested sampling likelihood and prior
def prior_transform(u):
    """ Transforms unit cube samples u to a flat prior between the specified values in each variable """
    #ustar = u*np.array([2.0, 1.0, 5.0, 5.0]) + np.array([0.0, 0.0, -3.0, -3.0])
    ustar = u*np.array([2.0, 1.0]) + np.array([0.0, 0.0])
    return ustar


def logL(p, alpha, beta):
    paras = [p[0], p[1], alpha, beta]
    return potential_fn(torch.from_numpy(np.array(paras)).to(torch.float32)).detach().numpy()[0] - np.log(1/prior_volume)


def prior_transform_RW(u):
    """ Transforms unit cube samples u to a flat prior between the specified values in each variable """
    #ustar = u*np.array([1.0, 5.0]) + np.array([0.0, -3.0])
    ustar = u*np.array([1.0]) + np.array([0.0])
    return ustar


def logL_RW(p, beta):
    paras = [p[0], beta]
    return potential_fn_RW(torch.from_numpy(np.array(paras)).to(torch.float32)).detach().numpy()[0] - np.log(1/prior_volume_RW)



#### MAIN LOOP 
# calculate the evidence for each alpha-beta combination

alpha_ar = [-2.0, -1.8, -1.6, -1.4, -1.2, -1.0, -0.8, -0.6 ,-0.4,-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
beta_ar = [-2.0, -1.8, -1.6, -1.4, -1.2, -1.0, -0.8, -0.6 ,-0.4,-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

#alpha_ar = [1.0]
#beta_ar = [0.0]

Nalpha = len(alpha_ar)
Nbeta = len(beta_ar)

evidences = np.zeros((Nalpha, Nbeta))
evidences_error = np.zeros((Nalpha, Nbeta))
evidences_RW = np.zeros((Nalpha,Nbeta))
evidences_error_RW = np.zeros((Nalpha,Nbeta))

for i in np.arange(Nalpha): # loop alpha
    for j in np.arange(Nbeta): # loop beta
        
        alpha = alpha_ar[i]
        beta = beta_ar[j]

        # get observation
        current_observation = get_observation([1., 0.2, alpha, beta])

        # get likelihood potential to be fed into nested sampling
        potential_fn, parameter_transform = likelihood_estimator_based_potential(
            likelihood_estimator, prior, current_observation
        )
        potential_fn_RW, parameter_transform_RW = likelihood_estimator_based_potential(
            likelihood_estimator_RW, prior_RW, current_observation
        )


        ## nested sampling
        nLivepoints = 100

        ## full dynamics
        try:
            sampler = dynesty.NestedSampler(logL, prior_transform, 2, nlive=nLivepoints,
                                        rstate=rstate,
                                            logl_args=[alpha, beta])

            sampler.run_nested()

            res = sampler.results
            evi = res.logz[-1]
            evierr = res.logzerr[-1]

            evidences[i,j] = evi
            evidences_error[i,j] = evierr
        except:
            evidences[i,j] = np.nan
            evidences_error[i,j] = np.nan


        ## random walk
        try:
            sampler = dynesty.NestedSampler(logL_RW, prior_transform_RW, 1, nlive=nLivepoints,
                                        rstate=rstate,
                                        logl_args=[beta])

            sampler.run_nested()

            res = sampler.results
            evi = res.logz[-1]
            evierr = res.logzerr[-1]

            evidences_RW[i,j] = evi
            evidences_error_RW[i,j] = evierr
        except:
            evidences_RW[i,j] = np.nan
            evidences_error_RW[i,j] = np.nan


print(evidences)
print(evidences_RW)

np.savetxt("phdg_logevidence_new_SBI_FIXEDbetalp_together_training40004000_tau02_dt0005_start3_Ntraj1000.csv", evidences, delimiter=",")
np.savetxt("phdg_logevidence_new_error_SBI_FIXEDbetalp_together_training40004000_tau02_dt0005_Start3_Ntraj1000.csv", evidences_error, delimiter=",")
np.savetxt("phdg_logevidence_new_SBI_FIXEDbetalp_RW_together_training40004000_tau02_dt0005_start3_Ntraj1000.csv", evidences_RW, delimiter=",")
np.savetxt("phdg_logevidence_new_error_SBI_FIXEDbetalp_RW_together_training40004000_tau02_dt0005_start3_Ntraj1000.csv", evidences_error_RW, delimiter=",")


