# Simulation Based Inference

`/sbi_Fig2/` contains scripts for training SBI estimators as well as code for plotting and prediction. We also combine nested sampling with the sbi-likelihood estimator to compute the evidence. Pretrained estimators for likelihood and posterior are included.

## Posterior Plotting

This script performs parameter inference on stochastic process models using simulation-based inference (SBI). It uses externally compiled C code for simulating time series, loads results, and visualizes posterior distributions of model parameters.

### How to Use

1. **Requirements**
   - Libraries: `numpy`, `torch`, `matplotlib`, `seaborn`, `sbi`, `dynesty`, and available C simulation binary.
   - C simulation binary must be in `./generateTSAtrajectories/prog` and produce `num_mean.txt`, `num_var.txt`.

2. **Edit Parameters** *(optional)*
   - Modify simulation settings (e.g., prior ranges, simulation time, or `true_params`) as needed.

3. **Run the Script**
   - Execute via terminal:
     ```
     python SimulationBasedInference_posterior_plotting.py
     ```
   - Ensure all paths (C program, data files, posterior pickle) are correct.

4. **Results**
   - Pairplots of posterior samples are saved as PDF and PNG files (e.g., `pairplot_WF.png`).

**Note**  
- To perform new inference/training, you must generate a new posterior with SBI (`posterior__TSAalpbet_Nens5000_Ntrain10000_dt0005_untiltau02.pkl`).

**In short**
- Runs a C simulation for a stochastic model.
- Performs parameter inference using summary statistics and a pre-trained SBI posterior.
- Plots and saves posterior distributions.

## Posterior Saving

This script trains a neural posterior estimator for parameter inference using SBI (Simulation-Based Inference). It simulates time series trajectories via an external C executable, collects summary statistics, trains a neural posterior (using SNPE), and saves the posterior for later use.

### How to Use

1. **Requirements**
   - Packages: `numpy`, `torch`, `matplotlib`, `sbi`, `dynesty`, `pickle`
   - C simulation binary at `./generateTSAtrajectories/prog` able to create `num_mean.txt`, `num_var.txt` in the working directory.

2. **Edit Parameters** *(if needed)*
   - Set model parameter ranges in `prior_min`/`prior_max`.
   - Adjust simulation parameters at the top (`fit_until_this_t`, `dt`, `nrealiz`).

3. **Run the Script**
   ```
   python SimulationBasedInference_evidence_alphabetaFIXED_savinglikelihood.py
   ```
   - Ensure that the C program path is correct and it writes the expected files.

4. **Posterior**
   - The trained posterior is saved as  
     `posterior__TSAalpbet_Nens5000_Ntrain10000_dt0005_untiltau02.pkl`
   - This file can be loaded in a separate script for inference and plotting.

**In short**
- Calls a C simulation with varied parameters sampled from a prior.
- Collects the simulation results as summary statistics.
- Trains a neural posterior estimator with SNPE.
- Saves the posterior for use in later inference/visualization scripts.

## Evidence Saving

This script trains neural likelihood estimators via SNLE (Sequential Neural Likelihood Estimation) for two types of stochastic models (full model and random walk), using simulation-based inference (SBI).  
It calls an external C simulation to generate time series data, processes the output into summary statistics, and trains neural models that approximate the likelihood for each scenario.  
Trained estimators are saved as `.pkl` files for later use (e.g., inference, posterior sampling, or model comparison).

### How to Use

1. **Requirements**
   - Packages: `numpy`, `torch`, `matplotlib`, `sbi`, `dynesty`, `pickle`
   - The external C simulator at `./generateTSAtrajectories/prog` producing `num_mean.txt` and `num_var.txt` for each simulation.

2. **Edit Parameters** *(optional)*
   - Change prior bounds, number of realizations, and simulation counts as needed.
   - Make sure the simulator can handle the requested parameters and outputs required files.

3. **Run the Script**
   ```sh
   python SimulationBasedInference_evidence_alphabetaFIXED_savinglikelihood.py
   ```
   - Make sure all paths and simulation configurations are correct (the C binary is accessible and works as expected).

4. **Using Trained Likelihoods**
   - Load the `.pkl` files in other scripts to:
      - Evaluate likelihoods for inference and comparison.
      - Perform posterior sampling or nested sampling.

**In short**
- The script builds a likelihood estimator for two models by generating simulated data, summarizing it, and training neural estimators via SNLE.
- The resulting neural likelihoods are serialized as `.pkl` files for later (possibly Bayesian) inference.

## Evidence Calculation for 4 Example Cases

This script estimates the Bayesian model evidence for different models and numbers of trajectories by using nested sampling with neural likelihoods trained previously via SBI (SNLE). It compares a general model and a random walk (RW) model, using an external C simulator to generate observed data. Model evidences and errors are computed over multiple cases and observation sizes, and results are saved as CSV files.

### How to Use

1. **Requirements**
   - Packages: `numpy`, `torch`, `matplotlib`, `sbi`, `dynesty`, `pickle`
   - The external C simulator at `./generateTSAtrajectories/prog` producing `num_mean.txt` and `num_var.txt`.
   - Neural likelihood pickle files from previous SNLE training:
     - `likelihood_general_Nens1000_Ntrain40004000_dt0005_untiltau02.pkl`
     - `likelihood_RW_Nens1000_Ntrain40004000_dt0005_untiltau02.pkl`

2. **Edit Parameters** *(optional)*
   - Adjust `alptbet_4cases`, `Ntrajlist`, and `Nboot` for models, sample sizes, and bootstrap repetitions.
   - Make sure paths for simulation and likelihood pickles are correct.

3. **Run the Script**
   - From command line:
     ```
     python SimulationBasedInference_evidence_4cases_alphabetaFIXED_loadinglikelihood_nestedsampling.py
     ```
   - Ensure C simulator works and expected files are created for each simulation.

4. **Results**
   - Model evidences and their bootstrap errors are saved as CSV files:
     - `phdg_logevidence_VS_Ntraj_4cases_SBI_FIXEDbetalp_together_training40004000_tau02_dt0005_start3_Ntraj1000_Nboot1_longer.csv`
     - Similar naming for error and RW evidence files.

**In short**
- The script uses trained neural likelihoods and nested sampling to estimate Bayesian evidence for several models as a function of observation size.
- Calls a C simulation executable to generate "real" observation data for inference.
- Saves all estimated evidences and errors (general and RW) for further analysis or plotting.

## Evidence Phasediagram

This script computes the Bayesian model evidence for a wide range of model parameters (`alpha`, `beta`) using nested sampling with neural likelihoods (learned with SNLE and SBI). It compares two models (a general stochastic model and a random-walk variant). For each (`alpha`, `beta`) pair, it:

- Simulates observations using an external C program via the `generatePowerEnsemble` class
- Loads pre-trained neural likelihoods from `.pkl` files
- Uses nested sampling (via `dynesty`) to compute and store the model evidence (and errors) for both models
- Saves all result arrays as CSV files for further analysis or visualization

### How to Use

1. **Requirements**
   - Packages: `numpy`, `torch`, `matplotlib`, `sbi`, `dynesty`, `pickle`
   - Compiled C simulation at `./generateTSAtrajectories/prog` that writes `num_mean.txt`, `num_var.txt` upon each simulation run
   - Neural likelihood `.pkl` files:
     - `likelihood_general_Nens1000_Ntrain40004000_dt0005_untiltau02.pkl`
     - `likelihood_RW_Nens1000_Ntrain40004000_dt0005_untiltau02.pkl`

2. **Edit Parameters** *(if needed)*
   - Change `alpha_ar` and `beta_ar` to define the parameter grid you want to evaluate.
   - Adjust `fit_until_this_t`, `dt`, `nrealiz` for simulation settings.

3. **Run the Script**
   ```sh
   python SimulationBasedInference_evidence_phasediagram_alphabetaFIXED_loadinglikelihood_nestedsampling.py
   ```
   - Ensure the working directory contains the necessary files and C simulator.

4. **Outputs**
   - Log-evidence values and error estimates for each (alpha, beta) pair, for both models:
     - `phdg_logevidence_new_SBI_FIXEDbetalp_together_training40004000_tau02_dt0005_start3_Ntraj1000.csv`
     - `phdg_logevidence_new_error_SBI_FIXEDbetalp_together_training40004000_tau02_dt0005_Start3_Ntraj1000.csv`
     - `phdg_logevidence_new_SBI_FIXEDbetalp_RW_together_training40004000_tau02_dt0005_start3_Ntraj1000.csv`
     - `phdg_logevidence_new_error_SBI_FIXEDbetalp_RW_together_training40004000_tau02_dt0005_start3_Ntraj1000.csv`

**In short**
- Loops over a grid of (`alpha`, `beta`) model parameters.
- Simulates data for each parameter set using a C forward model.
- Uses neural likelihoods and nested sampling to compute and compare Bayesian evidence for two models.
- Saves evidence values and errors to CSV files for downstream usage.

