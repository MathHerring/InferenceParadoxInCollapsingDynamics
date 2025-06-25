import numpy as np
import scipy as sp
import time, sys, os
import pickle


#### functions

def compute_corrmap(X,Y):
    #preallocation
    covdim = X.shape[1]
    
    corr=np.zeros((covdim,covdim))
    corrstderr=np.zeros((covdim,covdim))

    #compute covariance matrix
    for t1 in np.arange(covdim):
        for t2 in np.arange(covdim):
            x = X[:, t1]
            y = Y[:, t2]
            
            bad = ~np.logical_or(np.isnan(x), np.isnan(y))
            xnan = np.compress(bad, x)
            ynan = np.compress(bad, y)
            
            cov = np.mean((xnan-np.mean(xnan))*(ynan-np.mean(ynan)))
            corr[t1,t2] = cov/np.std(xnan)/np.std(ynan)
            corrstderr[t1,t2] = np.std((xnan-np.mean(xnan))*(ynan-np.mean(ynan)))/np.std(xnan)/np.std(ynan) / np.sqrt(np.size(xnan))
    
    return corr, corrstderr

def compute_covmap(X,Y):
    #preallocation
    covdim = X.shape[1]
    
    corr=np.zeros((covdim,covdim))

    #compute covariance matrix
    for t1 in np.arange(covdim):
        for t2 in np.arange(covdim):
            x = X[:, t1]
            y = Y[:, t2]
            
            bad = ~np.logical_or(np.isnan(x), np.isnan(y))
            xnan = np.compress(bad, x)
            ynan = np.compress(bad, y)
            
            cov = np.mean((xnan-np.mean(xnan))*(ynan-np.mean(ynan)))
            corr[t1,t2] = cov
    
    return corr


def entropy_from_eigen(evs):
    k = len(evs)
    norm = 0.5*k*np.log2(2*np.pi*np.exp(1))
    ent = 0.5*np.log2(np.prod(evs))
    return norm + ent



#################################### 
#################################### MAIN 


'''
This script loads previously calculated trajectories, calculates the two-time covariance and from that extracts the entropy.
Entropy values are saved.
'''


### get external options
path_data, path_save, type_covcorr, alpha, beta, gamma, D, dt, nrealiz, t_analyze = sys.argv[1:]
# path_data - path to trajectories
# path_save - path to save the entropy

alpha = float(alpha)
gamma = float(gamma)
D = float(D)
beta = float(beta)
nrealiz = int(nrealiz)
dt = float(dt)
t_analyze = float(t_analyze) # compute covariance until this tau

### load data
datastring = "trajectories_gamma{}_D{}_alpha{}_beta{}_N{}.txt".format(int(gamma*10),int(D*10), int(alpha*10), int(beta*10), nrealiz)
loadts = np.loadtxt(os.path.join(path_data, datastring))
tsarr = loadts


### want covariance to be 100x100
Ntimesteps = int(t_analyze/dt)
spacing = int(Ntimesteps/100)

ensemble = tsarr[:, :Ntimesteps][:2000, ::spacing]


### compute covariance/correlation
if type_covcorr == 'covariance':
    c_arr = compute_covmap(ensemble,ensemble)
elif type_covcorr == 'correlation':
    c_arr = compute_corrmap(ensemble,ensemble)
else:
    print('ERROR: type of computation neither covariance nor correlation')


eigen, eigenvec = np.linalg.eig(c_arr)
entr = np.real(entropy_from_eigen(eigen[:10]))


### save entropy
np.savetxt(os.path.join(path_save, 'entropy_data_type{}_gamma{}_D{}_alpha{}_beta{}_N{}.txt'.format(type_covcorr, int(gamma*10),int(D*10), int(alpha*10), int(beta*10), nrealiz)), np.array([entr, np.nan]))


## load like this:
# blub = np.loadtxt('thisisatest.csv')
# entropyval = blub[0]
