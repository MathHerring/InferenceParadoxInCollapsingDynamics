#!/usr/bin/env bash
#SBATCH -A cidbn_legacy
#SBATCH --exclude=amp052
#SBATCH --nodes=1                      # Request 1 node
#SBATCH -c 1
#SBATCH -t 2-00:00:00               # Request runtime of 30 minutes
#SBATCH -p cidbn                 # Run on sched_engaging_default partition
#SBATCH -o output_%j.txt          # Redirect output to output_JOBID.txt
#SBATCH -e error_%j.txt           # Redirect errors to error_JOBID.txt
#SBATCH -a 0-199 # 0-199 #X=num of jobs - 1 -> 0-7
#SBATCH --mail-type=BEGIN,END     # Mail when job starts and ends
#SBATCH --mail-user=matthias.haering@ds.mpg.de # Email recipient

shopt -s nullglob

#arr_starts=($(seq 0.01 0.005 1)) # start time
arr_starts=($(seq 0.01 0.01 2)) # interval length
LambdaLength=${#arr_starts[@]}

module use /usr/users/cidbn_sw/sw/modules
module load cidbn_miniforge3
source activate dynesty

#python /scratch/users/mhaerin1/evidence_phasetransition/inference_opaquetransition.py 0.25 0.15
#python /scratch/users/mhaerin1/evidence_phasetransition/inference_opaquetransition.py ${arr_starts[$(($SLURM_ARRAY_TASK_ID))]} 0.15 
python /scratch/users/mhaerin1/evidence_phasetransition/inference_opaquetransition.py 0.1 ${arr_starts[$(($SLURM_ARRAY_TASK_ID))]}
# ${arr_starts[$(($SLURM_ARRAY_TASK_ID/$LambdaLength))]} ${arr_beta[$(($SLURM_ARRAY_TASK_ID%$LambdaLength))]} 1.0 0.2 0.0005 2000
