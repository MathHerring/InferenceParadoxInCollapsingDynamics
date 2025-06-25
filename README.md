# InferenceParadoxInCollapsingDynamics

Code for paper "Inferring the Dynamics of Collapse" by Nicolas Lenner, Stephan Eule, Jörg Großhans, Fred Wolf and Matthias Häring

# Prerequisites
python dependencies:

- numpy
- scipy
- seaborn
- matplotlib
- dynesty
- sbi
- sdeint


# Contents

**Generating trajectories**

TSA trajectories can be generated using our C code - wrapped in python - in the folder /generateTSAensemble/

Alternatively /pathinference_Fig6/Paperplots_Examplemodels_TrajectoriesANDinference.ipynb contains native (much slower) python code.

**Simulation based inference**

/sbi_Fig2/ contains scripts for training SBI estimators as well as code for plotting and prediction. We also combine nested sampling with the sbi-likelihood estimator to compute the evidence. Pretrained estimators for likelihood and posterior are included.

**Theory figures**

Code for the Feller boundary classification plots, as well as code comparing reverse time theory with numerical simulations is given in /FellerClassification_Fig3/ and /TSAtheory_Fig4/

**Phasetransition**

Scripts for both entropy and squared coefficient of variation phasediagrams is given in /phasetransition_Fig5/ inlcuding most of the data needed to reproduce the plots. The scripts for cv and entropy generally only generate one datapoint of the respective diagram and have been evaluated on the cluster. 

The required trajectory data can be generated using the code given in /generateTSAensemble/.

**Reverse time inference**

In /pathinference_Fig6/ we showcase how to perform inference using our path inference formalism based on the reverse time theoretical framwork.


# Contributors

Matthias Häring - Göttingen Campus Institute For Dynamics Of Biological Networks

Nicolas Lenner - Institute For Advanced Study, Princeton

# Acknowledgements

For nested sampling inference we used dynesty: https://github.com/joshspeagle/dynesty

For simulation based inference we used sbi: https://github.com/sbi-dev/sbi
