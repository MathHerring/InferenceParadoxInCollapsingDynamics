
# Inference phasetransition using CV-squared, Entropy and Evidence


This directory contains several analysis of the inference phasetransition that were performed on target state aligned (TSA) ensemble data. Using pre-calculated raw TSA trajectories, the squared coefficient of variation (CV), the entropy, and the path-based evidence have been calculated and summarized in phasediagrams. The data arrays containing these values are included in this directory and the analysis based on this data can be found in the jupyter notebooks:
```
Analysis_EntropyPhasetransition.ipynb
```
and 
```
Analysis_CV_and_EvidencePhasetransition.ipynb
```
Both jupyter notbeooks contain all the plotting routines to replicate the figures of our manuscript. They also provide functions that cast our analytical results into python code for numerics-theory comparison.

## How to get the raw data

In case one wishes to calculate new phasediagrams, we here provide instructions how they can be generated from scratch.

The raw trajectories that are needed for calculating CV, entropy and evidence **are not included in this repository and have to be generated manually.** Since the trajectory data is very large, it was only saved on an HPC cluster. However, we do provide all necessary scripts to repeat the calculation of CV, entropy and evidence as well as code for trajectory generation.

For trajectory generation use
```
python GenerateTrajectories.py
```
An explanation of this script can be found in the README.md within the directory /generateTSAensemble/

The scripts `cluster_compute_cv.py` and `cluster_compute_cvphasediagram.py` contain code for how to calculate the cv-phasediagram. In similar fashion there are scripts for the entropy and evidence. 

These scripts have been concipated to be executed from bash scripts via slurm. Below we provide an example script which automatically distributes separate calls of the python script to different jobs on an HPC. See the slurm documentation for more details. Each job thus calculates the value for one parameter combination of the phasediagram, for example $\alpha$-$\beta$ or $\alpha$-$\gamma$. 

This example bash script specifies 
- the number of jobs (corresponding to the number of parameter combinations, here 961)
- then the arrays of the parameters we used to calculate the phasediagrams
- finally the call of the python script

To understand the arguments of the python call, have a look at the comments in the respective script. The only prerequisite for running all analysis is that the path to the pre-generated trajectories is given (naturally the trajectories must have been generated with the same parameters as here and saved according to the naming convection - see specific python script for details).
```
#SBATCH -a 0-960 #=num of jobs - 1 #This determines SLURM_ARRAY_TASK_ID

arr_alpha=(-2.0 -1.9 -1.8 -1.7 -1.6 -1.5 -1.4 -1.3 -1.2 -1.1 -1.0 -0.9 -0.8 -0.7 -0.6 -0.5 -0.4 -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

arr_gamma=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0)

arr_beta=(-2.0 -1.9 -1.8 -1.7 -1.6 -1.5 -1.4 -1.3 -1.2 -1.1 -1.0 -0.9 -0.8 -0.7 -0.6 -0.5 -0.4 -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

NumberOfParameters=31

python cluster_compute_entropy.py /directory/to/trajectory/data/ /directory/to/save/entropy/results/ covariance ${arr_alpha[$(($SLURM_ARRAY_TASK_ID/$NumberOfParameters))]} ${arr_beta[$(($SLURM_ARRAY_TASK_ID%$NumberOfParameters))]} 1.0 0.2 0.0005 2000 0.1
```
For the CV, an example including arguments is
```
python cluster_compute_cv.py /directory/to/trajectory/data/ /directory/to/save/entropy/results/ ${arr_alpha[$(($SLURM_ARRAY_TASK_ID/$NumberOfParameters))]} 0 ${arr_gamma[$(($SLURM_ARRAY_TASK_ID%$NumberOfParameters))]} 0.2 0.0005 2000 0.01
```

For the Evidence, an example including arguments is
```
python cluster_compute_evidencefromNS.py /directory/to/trajectory/data/ /directory/to/save/evidence/results/ random_walk ${arr_alpha[$(($SLURM_ARRAY_TASK_ID/$NumberOfParameters))]} ${arr_beta[$(($SLURM_ARRAY_TASK_ID%$NumberOfParameters))]} 1.0 0.2 0.0005 2000
```
To obtain the difference of the log-evidence between the "free-$\gamma$" and "pure diffusion" models, the evidence script must be executed twice - with the "random_walk" and "general_force" arguments respectively.

