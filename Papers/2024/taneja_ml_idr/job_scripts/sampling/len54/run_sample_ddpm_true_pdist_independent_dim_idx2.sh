#!/bin/sh
#SBATCH --job-name=sample_2
#SBATCH --nodes=1           
#SBATCH --partition=gpu
#SBATCH --exclusive 
#SBATCH -o sample_ddpm_true_pdist_independent_dim_len54_idx2.out
#SBATCH -e sample_ddpm_true_pdist_independent_dim_len54_idx2.err

cd /gpfs/home/itaneja/train_ml_models/18_pointmutation

module load pytorch 

python -u sample_ddpm_true_pdist_independent_dim.py 1 50 2 5000 15000 54 100 > sample_ddpm_true_pdist_independent_dim_len54_idx2_n100.out 
