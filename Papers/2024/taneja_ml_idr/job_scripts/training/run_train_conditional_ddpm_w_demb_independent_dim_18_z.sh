#!/bin/sh   
#SBATCH --job-name=train_model_conditional
#SBATCH --nodes=1           
#SBATCH --partition=gpu
#SBATCH --exclusive 
#SBATCH -o train_model_conditional_w_emb_dim_18_z.out
#SBATCH -e train_model_conditional_w_emb_dim_18_z.err

cd /gpfs/home/itaneja/train_ml_models/18_pointmutation

module load pytorch 

python -u train_ddpm_w_demb_independent_dim.py 1 10 18 z > train_model_conditional_w_emb_dim_18_z.out 
