#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals,division#, print_function
import numpy as np
import scipy as sp

import os

import matplotlib.pyplot as plt
from matplotlib import ticker
###############################################################################


class PureFwdPowerlaw(object):

    def __init__(self,fit_until_this_t=300,nrealiz=5,gamma=1.0,D=0.2,alpha=-1.0,beta=0.0):


        self.fit_until_this_t = fit_until_this_t
        self.nrealiz=nrealiz
        self.gamma = gamma
        self.D = D
        self.alpha = alpha
        self.beta = beta



        #####################################
        self.var_start_sim = 0
        self.x_upper_bound = 1000 #3
        self.max_rtime = fit_until_this_t
        self.traFreq = 1#100
        self.wfreq = 50#100#100
        self.total_time=2000#80#max(40,fit_until_this_t)
        self.x0_for_reverse = 0
        #self.D = 0.2
        self.dt = 0.0005#0.00005#0.005#0.05##0.5#0.05#0.0005
        self.dt_eff = self.dt*self.traFreq
        self.seed = 12345
        self.x_start_sim = 2#4#3#20#1#2#3#7#5#2#10#2#1#0.5#1#2  


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
fit_until_this_t=30
fit_until_this_t_ref = fit_until_this_t
nrealiz = 2000
max_load_fwd = 50

simulationkey = True 

pathsave = ""


if simulationkey == True:
    os.chdir("/scratch/users/mhaerin1/generatePowerlawData/")
    alpha_ar = [0.0]#[-2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5 ,-0.4, -0.3 ,-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    gamma_ar = [0.0] #[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]
    D_ar = [0.2]
    beta_ar = [0.0]#[-2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5 ,-0.4, -0.3 ,-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for i in np.arange(len(alpha_ar)):
        for j in np.arange(len(beta_ar)):
            for k in np.arange(len(gamma_ar)):
                alpha = alpha_ar[i]
                gamma = gamma_ar[k]
                D = D_ar[0]
                beta = beta_ar[j]

                print('################################################')
                print('alpha = ', alpha, ' ,gamma = ', gamma, ',D = ', D,'beta  =', beta)
                print('################################################')
                PFP = PureFwdPowerlaw(fit_until_this_t,nrealiz,gamma,D,alpha,beta)
                PFP.run()

                # the c code saves trajectories in "num_trj.txt". Here we load and resave the data including a nametag to identify the TSA ensemble.
                trajs = np.loadtxt('num_trj.txt')
                np.savetxt(os.path.join(pathsave,"trajectories_gamma{}_D{}_alpha{}_beta{}_N{}.txt".format(int(gamma*10),int(D*10), int(alpha*10), int(beta*10), nrealiz)), trajs)

                '''
                # load output of c simulation and save
                # can also load the raw trajectories and more, for info look into c++ code. the 'num_...' files are the c outputs. their naming should already be telling as well.
                md = np.loadtxt('num_mean.txt')
                mv = np.loadtxt('num_var.txt')
                cov_ar=np.loadtxt("num_cov.txt")
                #hittingTime=np.loadtxt("hittingTime.txt")
                
                np.savetxt("meannegtwo_gamma{}_D{}_alpha{}_beta{}_N{}.txt".format(int(gamma*10),int(D*10), int(alpha*10), int(beta*10), nrealiz), md)
                np.savetxt("varnegtwo_gamma{}_D{}_alpha{}_beta{}_N{}.txt".format(int(gamma*10),int(D*10), int(alpha*10), int(beta*10), nrealiz), mv)
                np.savetxt("covnegtwo_gamma{}_D{}_alpha{}_beta{}_N{}.txt".format(int(gamma*10),int(D*10), int(alpha*10), int(beta*10), nrealiz), cov_ar)
                '''


