# TSA core theory comparison with numerical simulations

This directory contains the code for generating plots of Figure 4, which compares the TSA theory with numerical simulations. The code is given in the jupyter notebook

```
Figure4_tsa_theory.ipynb
```

The numerical simulations have been performed with the parameters $\gamma=1, D=0.2, \alpha=0, \beta=1$ and results of the moments of the TSA ensemble are contained in this directory. The only file not included in this repository
is "trajs_gamma10_D2_alpha0_beta10_N10000_dt500.txt" which contains the raw trajectories, only needed for one plot. However, the trajectories can be generated using the code provided in /generateTSAEnsemble/ (see instructions 
given in the directory for details).

Additionally, the notebook contains the analytical results of the exact and core TSA dynamics in python-executable functions.
