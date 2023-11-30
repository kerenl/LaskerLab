#!/bin/sh
#SBATCH --job-name=sample_2
#SBATCH --nodes=1           
#SBATCH --partition=gpu
#SBATCH --exclusive 
#SBATCH -o sample_ddpm_sampled_pdist_true_mean_pred_cov_indepedent_dim_len18_idx2.out
#SBATCH -e sample_ddpm_sampled_pdist_true_mean_pred_cov_independent_dim_len18_idx2.err

cd /gpfs/home/itaneja/train_ml_models/18_pointmutation

module load pytorch 

python -u sample_ddpm_sampled_pdist_independent_dim.py 1 10 2 5000 15000 1 0 18 > sample_ddpm_sampled_pdist_true_mean_pred_cov_independent_dim_len18_idx2.out 
