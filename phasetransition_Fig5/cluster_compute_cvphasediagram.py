import numpy as np
import scipy as sp
import time, sys, os
import pickle


'''
This script loads the saved cv values and bundles them into one array.
'''


path_data = '' # path to previsouly computed cv values
path_save = '' # path to save resulting arrays

alpha_ar = [-2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5 ,-0.4, -0.3 ,-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
gamma_ar = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]
#D_ar = [0.2]
beta_ar = [-2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5 ,-0.4, -0.3 ,-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
Nalp = len(alpha_ar)
Nbet = len(beta_ar)
Ngam = len(gamma_ar)


D = 0.2
nrealiz=2000


gamma = 1.0
phase_alphabeta = np.zeros((Nalp, Nbet))
phase_alphabeta[:] = np.nan
for i in np.arange(Nalp):
    for j in np.arange(Nbet):
        alpha=alpha_ar[i]
        beta=beta_ar[j]

        arr = np.loadtxt(os.path.join(path_data, 'cv_data_gamma{}_D{}_alpha{}_beta{}_N{}.txt'.format(int(gamma*10),int(D*10), int(alpha*10), int(beta*10), nrealiz)))
        entr = arr[0]

        phase_alphabeta[i,j] = entr


beta = 0.0
phase_alphagamma = np.zeros((Nalp, Ngam))
phase_alphagamma[:] = np.nan
for i in np.arange(Nalp):
    for j in np.arange(Ngam):
        alpha=alpha_ar[i]
        gamma=gamma_ar[j]

        arr = np.loadtxt(os.path.join(path_data, 'cv_data_gamma{}_D{}_alpha{}_beta{}_N{}.txt'.format(int(gamma*10),int(D*10), int(alpha*10), int(beta*10), nrealiz)))
        entr = arr[0]

        phase_alphagamma[i,j] = entr




np.savetxt(os.path.join(path_save, "phdg_cv_alpgam.csv"), phase_alphagamma, delimiter=",")
np.savetxt(os.path.join(path_save, "phdg_cv_alpbet.csv"), phase_alphabeta, delimiter=",")