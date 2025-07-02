
# Inference using analytical path-likelihood

This directory contains the code to perform inference using our analytical path-based likelihood function. All the code needed to reproduce Figure 6 in the manuscript are given in the jupyter notebooks
```
Paperplots_Examplemodels_TrajectoriesANDinference.ipynb
```
and
```
PlotOpaqueInference.ipynb
```

The first notebook additionally generates visualizations of trajectories for the 4 models used as examples throughout the manuscript. The data is contained in this directory. For the opaque evidence comparison we only provide the final result. Below we detail how to obtain these results if needed.

## How to obtain raw data for opaque evidence results

The evidence comparison data has been generated using the scripts `cluster_inference_opaquetransition.py` and `cluster_opaqueevidence.py`

The raw trajectories that are needed for calculating the evidence **are not included in this repository and have to be generated manually.** For trajectory generation use the scripts given in /phasetransition_Fig5/ or /generateTSAensemble/ which also contains a detailed explanation. 

The `cluster_inference_opaquetransition.py` scripts have been concipated to be executed from bash scripts via slurm. Below we provide an example script which automatically distributes separate calls of the python script to different jobs on an HPC. See the slurm documentation for more details. Each job thus calculates the value for one inference interval. 

This example bash script specifies 
- the number of jobs (corresponding to the number of parameter combinations, here 200)
- then the array of the parameters we used for calculation
- finally the call of the python script

To understand the arguments of the python call, have a look at the comments in the respective script. The only prerequisite for running all analysis is that the path to the pre-generated trajectories is given (naturally the trajectories must have been generated with the same parameters as here and saved according to the naming convection - see specific python script for details).
```
#SBATCH -a 0-199 #=num of jobs - 1 #This determines SLURM_ARRAY_TASK_ID

arr_starts=($(seq 0.01 0.01 2)) # interval length

python cluster_inference_opaquetransition.py 0.1 ${arr_starts[$(($SLURM_ARRAY_TASK_ID))]}
```
The two arguments are the $\tau$ values at which the inference interval starts (here 0.1) and the length of the inference interval being mapped out.

