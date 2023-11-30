import numpy as np 
import pickle 
from joblib import Parallel, delayed, dump, load

from pathlib import Path
import os
import os.path
import glob

num_residues = 54 
num_residues_str = 'len%d' % num_residues

if num_residues == 18:
    rel_bin_list = range(0,10)
else:
    rel_bin_list = [1,2,3,4,5]


home_dir = '/gpfs/home/itaneja/train_ml_models/18_pointmutation'


if num_residues == 18: #this is only relevant for 18 

    #########################

    save_dir = '%s/pairwise_dist_sampled_from_mean_cov/%s' % (home_dir, num_residues_str) 
    pairwise_dist_sampled_true_mean_true_cov = np.load('%s/pairwise_dist_sampled_true_mean_true_cov_df.npy' % save_dir)

    print(pairwise_dist_sampled_true_mean_true_cov.shape)

    save_dir = '%s/memmap_data/%s' % (home_dir, num_residues_str)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print('dumping pairwise_dist_sampled_true_mean_true_cov')

    pdist_filename_memmap =  '%s/memmap_data/%s/pairwise_dist_sampled_true_mean_true_cov_memmap' % (home_dir, num_residues_str)
    dump(pairwise_dist_sampled_true_mean_true_cov, pdist_filename_memmap)


    #########################

    save_dir = '%s/pairwise_dist_sampled_from_mean_cov/%s' % (home_dir, num_residues_str) 
    pairwise_dist_sampled_true_mean_pred_cov = np.load('%s/pairwise_dist_sampled_true_mean_pred_cov_df.npy' % save_dir)

    print(pairwise_dist_sampled_true_mean_pred_cov.shape)

    save_dir = '%s/memmap_data/%s' % (home_dir, num_residues_str) 
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print('dumping pairwise_dist_sampled_true_mean_pred_cov')

    pdist_filename_memmap =  '%s/memmap_data/%s/pairwise_dist_sampled_true_mean_pred_cov_memmap' % (home_dir, num_residues_str)
    dump(pairwise_dist_sampled_true_mean_pred_cov, pdist_filename_memmap)


    ########################

    save_dir = '%s/pairwise_dist_sampled_from_mean_cov/%s' % (home_dir, num_residues_str) 
    pairwise_dist_sampled_pred_mean_pred_cov = np.load('%s/pairwise_dist_sampled_pred_mean_pred_cov_df.npy' % save_dir)

    print(pairwise_dist_sampled_pred_mean_pred_cov.shape)

    save_dir = '%s/memmap_data/%s' % (home_dir, num_residues_str) 
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print('dumping pairwise_dist_sampled_pred_mean_pred_cov')

    pdist_filename_memmap =  '%s/memmap_data/%s/pairwise_dist_sampled_pred_mean_pred_cov_memmap' % (home_dir, num_residues_str)
    dump(pairwise_dist_sampled_pred_mean_pred_cov, pdist_filename_memmap)


    ######################


    save_dir = '%s/pairwise_dist_sampled_from_mean_cov/%s' % (home_dir, num_residues_str) 
    pairwise_dist_sampled_pred_mean_true_cov = np.load('%s/pairwise_dist_sampled_pred_mean_true_cov_df.npy' % save_dir)

    print(pairwise_dist_sampled_pred_mean_true_cov.shape)

    save_dir = '%s/memmap_data/%s' % (home_dir, num_residues_str) 
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print('dumping pairwise_dist_sampled_pred_mean_true_cov')

    pdist_filename_memmap =  '%s/memmap_data/%s/pairwise_dist_sampled_pred_mean_true_cov_memmap' % (home_dir, num_residues_str)
    dump(pairwise_dist_sampled_pred_mean_true_cov, pdist_filename_memmap)


########################

save_dir = '%s/pairwise_dist/%s' % (home_dir, num_residues_str) 

pairwise_dist_all_bin_train = []
pairwise_dist_all_bin_test = []

for bin_num in rel_bin_list:
    
    print(bin_num)  

    pairwise_dist_curr_bin_train = np.load('%s/pairwise_dist_train_bin_%d.npy' % (save_dir, bin_num))    
    pairwise_dist_all_bin_train.append(pairwise_dist_curr_bin_train)

    pairwise_dist_curr_bin_test = np.load('%s/pairwise_dist_test_bin_%d.npy' % (save_dir, bin_num))
    pairwise_dist_all_bin_test.append(pairwise_dist_curr_bin_test)

pairwise_dist_all_bin_train = np.vstack(pairwise_dist_all_bin_train)
print(pairwise_dist_all_bin_train.shape)

pairwise_dist_all_bin_train = pairwise_dist_all_bin_train.reshape(pairwise_dist_all_bin_train.shape[0]*pairwise_dist_all_bin_train.shape[1],pairwise_dist_all_bin_train.shape[2])
print(pairwise_dist_all_bin_train.shape)


pairwise_dist_all_bin_test = np.vstack(pairwise_dist_all_bin_test)
print(pairwise_dist_all_bin_test.shape)

pairwise_dist_all_bin_test = pairwise_dist_all_bin_test.reshape(pairwise_dist_all_bin_test.shape[0]*pairwise_dist_all_bin_test.shape[1],pairwise_dist_all_bin_test.shape[2])
print(pairwise_dist_all_bin_test.shape)


pdist_mean_data = np.mean(pairwise_dist_all_bin_train,axis=0)
pdist_std_data = np.std(pairwise_dist_all_bin_train,axis=0)

print('saving pdist_mean/std')

np.save('%s/pairwise_dist/%s/pdist_mean_data.npy' % (home_dir, num_residues_str), pdist_mean_data)
np.save('%s/pairwise_dist/%s/pdist_std_data.npy' % (home_dir, num_residues_str), pdist_std_data)


save_dir = '%s/memmap_data/%s' % (home_dir, num_residues_str) 
Path(save_dir).mkdir(parents=True, exist_ok=True)

pdist_filename_memmap =  '%s/memmap_data/%s/pdist_memmap' % (home_dir, num_residues_str)
dump(pairwise_dist_all_bin_test, pdist_filename_memmap)


######################


save_dir = '%s/xyz_coeff/%s' % (home_dir, num_residues_str)

xyz_coeff_all_bin_train = []
xyz_coeff_all_bin_test = []


for bin_num in rel_bin_list:

    print(bin_num)
    xyz_coeff_curr_bin_train = np.load('%s/xyz_coeff_train_bin_%d.npy' % (save_dir, bin_num))
    xyz_coeff_curr_bin_test = np.load('%s/xyz_coeff_test_bin_%d.npy' % (save_dir, bin_num))
    
    xyz_coeff_all_bin_train.append(xyz_coeff_curr_bin_train)
    xyz_coeff_all_bin_test.append(xyz_coeff_curr_bin_test)
    

xyz_coeff_all_bin_train = np.vstack(xyz_coeff_all_bin_train)
xyz_coeff_all_bin_test = np.vstack(xyz_coeff_all_bin_test)



#####



xyz_coeff_all_bin_train = xyz_coeff_all_bin_train.reshape(xyz_coeff_all_bin_train.shape[0]*xyz_coeff_all_bin_train.shape[1],xyz_coeff_all_bin_train.shape[2],xyz_coeff_all_bin_train.shape[3])
xyz_coeff_all_bin_train = xyz_coeff_all_bin_train.reshape(xyz_coeff_all_bin_train.shape[0],xyz_coeff_all_bin_train.shape[1]*xyz_coeff_all_bin_train.shape[2])
print(xyz_coeff_all_bin_train.shape)

xyz_coeff_all_bin_test = xyz_coeff_all_bin_test.reshape(xyz_coeff_all_bin_test.shape[0]*xyz_coeff_all_bin_test.shape[1],xyz_coeff_all_bin_test.shape[2],xyz_coeff_all_bin_test.shape[3])
xyz_coeff_all_bin_test = xyz_coeff_all_bin_test.reshape(xyz_coeff_all_bin_test.shape[0],xyz_coeff_all_bin_test.shape[1]*xyz_coeff_all_bin_test.shape[2])
print(xyz_coeff_all_bin_test.shape)



##########################


xyz_coeff_mean_data = np.mean(xyz_coeff_all_bin_train,axis=0)
xyz_coeff_std_data = np.std(xyz_coeff_all_bin_train,axis=0)

print('saving xyz_coeff_mean/std')

np.save('%s/xyz_coeff/%s/xyz_coeff_mean_data.npy' % (home_dir, num_residues_str), xyz_coeff_mean_data)
np.save('%s/xyz_coeff/%s/xyz_coeff_std_data.npy' % (home_dir, num_residues_str), xyz_coeff_std_data)

