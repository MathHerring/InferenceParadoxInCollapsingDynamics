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
from sbi.inference import SNPE, SNLE, simulate_for_sbi
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
        os.system('./generateTSAtrajectories/prog %f %d %d %f %f %f %d %d %d %d %f %f %f %f %f' % (
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
nrealiz = 5000


# FULL DYNAMICS
prior_min = [0.1, 0.01, -2, -2]
prior_max = [2, 0.3, 2, 2]
prior = utils.torchutils.BoxUniform(
    low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max)
)

# Check prior, simulator, consistency
prior, num_parameters, prior_returns_numpy = process_prior(prior)
simulation_wrapper = process_simulator(simulation_wrapper, prior, prior_returns_numpy)
check_sbi_inputs(simulation_wrapper, prior)
    

###########################
###########################
##### plotting script starts here


# load trained posterior estimator
with open("posterior__TSAalpbet_Nens5000_Ntrain10000_dt0005_untiltau02.pkl", "rb") as handle:
    posterior = pickle.load(handle)


## generating "observation" for inference test
# choose model use
# parameters are always: [gamma, D, alpha, beta]
#true_params = [1., 0.2, -1., 0.]  # Bessel
#true_params = [1., 0.2, 1., 0.]  # Vasicek
#true_params = [1., 0.2, 0., 0.]  # advection-diffusion
#true_params = [1., 0.2, 1., 1.]  # Cox-Ingersoll-Ross 
true_params = [1., 0.2, 0., 1.]  # Wright-Fisher
observation_summary_statistics = get_observation(true_params)


samples = posterior.sample((20000,), x=observation_summary_statistics)


## plotting results
# use seaborn for plotting
import seaborn as sns
sns.set()
sns.set_style("white")
sns.set_context("notebook", font_scale=2, rc={"lines.linewidth": 3.0})


fig, axes = analysis.pairplot(
    samples,
    limits=[[0.11, 1.5], [0, 0.4], [-1.5,2], [-0.5,1.5]],
    #ticks=[[0.5, 80], [1e-4, 15.0]],
    figsize=(5, 5),
    points=np.array(true_params),
    #points_offdiag={"markersize": 6},
    points_colors="g",
    labels=[r"$\gamma$", r"$D$", r"$\alpha$", r"$\beta$"]
)
fig.tight_layout()
fig.savefig('pairplot_WF.png')
fig.savefig('pairplot_WF.pdf')

fig, axes = analysis.pairplot(
    samples[:,:2],
    limits=[[0.11, 1.5], [0, 0.4]],
    #ticks=[[0.5, 80], [1e-4, 15.0]],
    figsize=(5, 6.5),
    points=np.array(true_params),
    #points_offdiag={"markersize": 6},
    points_colors="g",
    labels=[r"$\gamma$", r"$D$"]
)
fig.tight_layout()
fig.savefig('WF_gamD_Nens5000_Ntrain10000_dt0005_untiltau02.png')
fig.savefig('WF_gamD_Nens5000_Ntrain10000_dt0005_untiltau02.pdf')







