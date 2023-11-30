#!/bin/sh
#SBATCH --job-name=sample_0
#SBATCH --nodes=1           
#SBATCH --partition=gpu
#SBATCH --exclusive 
#SBATCH -o sample_ddpm_true_pdist_independent_dim_len18_idx0.out
#SBATCH -e sample_ddpm_true_pdist_independent_dim_len18_idx0.err

cd /gpfs/home/itaneja/train_ml_models/18_pointmutation

module load pytorch 

python -u sample_ddpm_true_pdist_independent_dim.py 1 10 0 5000 15000 18 > sample_ddpm_true_pdist_independent_dim_len18_idx0.out 
