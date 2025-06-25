
import numpy as np
import scipy as sp
import time, sys, os
import pickle


'''
This script loads the trajectories of the specified TSA ensemble and computes the squared coefficient of variation (here called cv).
'''


### get external options
path_data, path_save, alpha, beta, gamma, D, dt, nrealiz, t_analyze = sys.argv[1:]

alpha = float(alpha)
gamma = float(gamma)
D = float(D)
beta = float(beta)
nrealiz = int(nrealiz)
dt = float(dt)
t_analyze = float(t_analyze) # compute covariance until this tau, often will be 3

### load data
datastring = "trajectories_gamma{}_D{}_alpha{}_beta{}_N{}.txt".format(int(gamma*10),int(D*10), int(alpha*10), int(beta*10), nrealiz)
loadts = np.loadtxt(os.path.join(path_data, datastring))
tsarr = loadts


mean = np.nanmean(tsarr, axis=0)
var = np.nanvar(tsarr, axis=0)

cvsquared = var/mean**2

cvtozero = cvsquared[int(t_analyze/dt)]


### save cv
np.savetxt(os.path.join(path_save, 'cv_data_gamma{}_D{}_alpha{}_beta{}_N{}.txt'.format(int(gamma*10),int(D*10), int(alpha*10), int(beta*10), nrealiz)), np.array([cvtozero, np.nan]))


## load like this:
# blub = np.loadtxt('thisisatest.csv')
# entropyval = blub[0]