#!/bin/sh   
cat <<EoF
#!/bin/sh
#SBATCH --job-name=sample_$1
#SBATCH --nodes=1           
#SBATCH --partition=gpu
#SBATCH --exclusive 
#SBATCH -o sample_ddpm_sampled_pdist_pred_mean_pred_cov_independent_dim_len$2_idx$1.out
#SBATCH -e sample_ddpm_sampled_pdist_pred_mean_pred_cov_independent_dim_len$2_idx$1.err

cd /gpfs/home/itaneja/train_ml_models/18_pointmutation

module load pytorch 

python -u sample_ddpm_sampled_pdist_independent_dim.py 1 $4 $1 5000 $3 0 0 $2 > sample_ddpm_sampled_pdist_pred_mean_pred_cov_independent_dim_len$2_idx$1.out 
EoF
