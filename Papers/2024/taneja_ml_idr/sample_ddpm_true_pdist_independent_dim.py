from ddpm import *
from eval_ddpm_sample_functions import * 

import numpy as np  

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd

from pathlib import Path
import os
import os.path
import glob
import shutil
import math
import time

import sys 
import pickle

import gc  

import cpuinfo 
import joblib 
from joblib import Parallel, delayed, dump, load
import time
#from joblib.externals.loky import get_reusable_executor



######################

home_dir = '/gpfs/home/itaneja/train_ml_models/18_pointmutation'

arg1 = sys.argv[1]
arg2 = sys.argv[2]
arg3 = sys.argv[3]
arg4 = sys.argv[4]
arg5 = sys.argv[5]
arg6 = sys.argv[6]
arg7 = sys.argv[7]

sample_conditional = bool(int(arg1))
num_epochs = int(arg2)
bin_idx = int(arg3) #used to define seed to sample row 
num_samples = int(arg4)
num_repeat = int(arg5) #how many conformations to generate per sample 
num_residues = int(arg6)
num_instances_to_sample = str(arg7)

num_residues_str = 'len%d' % num_residues
sample_str = 'sample=%s' % num_instances_to_sample

if num_residues == 18:
    from load_metadata_len18 import *
elif num_residues == 36:
    from load_metadata_len36 import * 
elif num_residues == 54:
    from load_metadata_len54 import * 


####################

knot_vector_GS_mean = np.load('/gpfs/home/itaneja/train_ml_models/18_pointmutation/metadata/%s/knot_vector_GS_mean.npy' % num_residues_str)

#################### 

test_instances_df = pd.read_csv('%s/%s/%s/test_instances_df.csv' % (home_dir, 'train_test_info', num_residues_str)) 

print(test_instances_df)

if num_residues == 18:
    rel_bin_list = range(0,10)
else:
    rel_bin_list = [1,2,3,4,5]

bin_count_dict = {}
for i in rel_bin_list:
    bin_count_dict[i] = -1 
    
test_dir_idx_list = [] 

if num_residues == 18:
    for i in rel_bin_list:
        print(i)
        subset_test_instances_df = test_instances_df[test_instances_df['orig_seq_bin_num'] == i].reset_index()
        rand_row = subset_test_instances_df.sample(n=1, random_state=bin_idx)
        print(rand_row)
        rand_idx = int(rand_row['index'])
        test_dir_idx_list.append(rand_idx)
else:
    for i in rel_bin_list:
        print(i)
        subset_test_instances_df = test_instances_df[test_instances_df['orig_seq_bin_num'] == i].reset_index()
        rand_row = subset_test_instances_df.sample(n=1, random_state=bin_idx+10)
        rand_idx = int(rand_row['index'])
        test_dir_idx_list.append(rand_idx)
        if bin_idx == 3:
            break 
        rand_row = subset_test_instances_df.sample(n=1, random_state=bin_idx+20)
        rand_idx = int(rand_row['index'])
        test_dir_idx_list.append(rand_idx)
    
print('inital test_dir_idx_list')      
print(test_dir_idx_list)    

if sample_conditional:
    save_dir = '%s/ddpm_samples_true_pdist_conditional_independent_dim/%s/%s/num_samples=%d/num_repeat=%d' % (home_dir,num_residues_str,sample_str,num_samples,num_repeat)
else:
    save_dir = '%s/ddpm_samples_true_pdist_unconditional_independent_dim/%s/%s/num_samples=%d/num_repeat=%d' % (home_dir,num_residues_str,sample_str,num_samples,num_repeat)


print(save_dir)

test_dir_idx_list_unprocessed = []

for test_dir_idx in test_dir_idx_list:
    
    file_path = '%s/test_dir_idx=%d.pkl' % (save_dir,test_dir_idx)
    print(file_path)
    if (os.path.isfile(file_path)) == False: 
        test_dir_idx_list_unprocessed.append(test_dir_idx)

test_dir_idx_list = test_dir_idx_list_unprocessed
        
print('test_dir_idx_list without processed samples')      
print(test_dir_idx_list)
    
###################

if sample_conditional:
    model_name = 'sample_conditional_%d_epochs' % num_epochs
else:
    model_name = 'sample_unconditional_%d_epochs' % num_epochs



##load model_x##

model_save_dir = '%s/ddpm_model_w_demb_independent_dim/%s/x/%s' % (home_dir,num_residues_str,sample_str)
model_save_path = '%s/%s.pth' % (model_save_dir, model_name)


model_info_path = '%s/%s.pkl' % (model_save_dir, model_name)

with open(model_info_path, 'rb') as f:
    model_info_dict = pickle.load(f)


ndist = model_info_dict['ndist']
d_emb = model_info_dict['d_emb']
ncoords = model_info_dict['ncoords']
nout = model_info_dict['nout']
num_timesteps = model_info_dict['num_timesteps']
hidden_size = model_info_dict['hidden_size']
embedding_size = model_info_dict['embedding_size']
hidden_layers = model_info_dict['hidden_layers']
time_embedding =  model_info_dict['time_embedding']
input_embedding = model_info_dict['input_embedding']


model_x = MLP_dist_emb(ndist,d_emb,ncoords,nout,num_timesteps, 
    hidden_size=hidden_size,
    hidden_layers=hidden_layers,
    emb_size=embedding_size,
    time_emb=time_embedding,
    input_emb=input_embedding)


print(device)
print(str(device) == 'cpu')

if str(device) == 'cpu':
    model_x.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
else:
    model_x.load_state_dict(torch.load(model_save_path))

model_x.to(device)


##load model_y##

model_save_dir = '%s/ddpm_model_w_demb_independent_dim/%s/y/%s' % (home_dir,num_residues_str,sample_str)
model_save_path = '%s/%s.pth' % (model_save_dir, model_name)


model_info_path = '%s/%s.pkl' % (model_save_dir, model_name)

with open(model_info_path, 'rb') as f:
    model_info_dict = pickle.load(f)


ndist = model_info_dict['ndist']
d_emb = model_info_dict['d_emb']
ncoords = model_info_dict['ncoords']
nout = model_info_dict['nout']
num_timesteps = model_info_dict['num_timesteps']
hidden_size = model_info_dict['hidden_size']
embedding_size = model_info_dict['embedding_size']
hidden_layers = model_info_dict['hidden_layers']
time_embedding =  model_info_dict['time_embedding']
input_embedding = model_info_dict['input_embedding']


model_y = MLP_dist_emb(ndist,d_emb,ncoords,nout,num_timesteps, 
    hidden_size=hidden_size,
    hidden_layers=hidden_layers,
    emb_size=embedding_size,
    time_emb=time_embedding,
    input_emb=input_embedding)


print(device)
print(str(device) == 'cpu')

if str(device) == 'cpu':
    model_y.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
else:
    model_y.load_state_dict(torch.load(model_save_path))

model_y.to(device)


##load model_z##

model_save_dir = '%s/ddpm_model_w_demb_independent_dim/%s/z/%s' % (home_dir,num_residues_str,sample_str)
model_save_path = '%s/%s.pth' % (model_save_dir, model_name)


model_info_path = '%s/%s.pkl' % (model_save_dir, model_name)

with open(model_info_path, 'rb') as f:
    model_info_dict = pickle.load(f)


ndist = model_info_dict['ndist']
d_emb = model_info_dict['d_emb']
ncoords = model_info_dict['ncoords']
nout = model_info_dict['nout']
num_timesteps = model_info_dict['num_timesteps']
hidden_size = model_info_dict['hidden_size']
embedding_size = model_info_dict['embedding_size']
hidden_layers = model_info_dict['hidden_layers']
time_embedding =  model_info_dict['time_embedding']
input_embedding = model_info_dict['input_embedding']


model_z = MLP_dist_emb(ndist,d_emb,ncoords,nout,num_timesteps, 
    hidden_size=hidden_size,
    hidden_layers=hidden_layers,
    emb_size=embedding_size,
    time_emb=time_embedding,
    input_emb=input_embedding)


print(device)
print(str(device) == 'cpu')

if str(device) == 'cpu':
    model_z.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
else:
    model_z.load_state_dict(torch.load(model_save_path))

model_z.to(device)



########################


noise_scheduler = NoiseScheduler(
  num_timesteps=num_timesteps,
beta_schedule="linear")



#########################


save_dir = '%s/xyz_coeff/%s' % (home_dir,num_residues_str)
xyz_coeff_mean_data = np.load('%s/xyz_coeff_mean_data.npy' % save_dir)
xyz_coeff_std_data = np.load('%s/xyz_coeff_std_data.npy' % save_dir)


save_path = '%s/train_test_info/%s/train_instance_idx_mapping.pkl' % (home_dir,num_residues_str)
with open(save_path, 'rb') as f:
    train_instance_idx_mapping = pickle.load(f)

save_path = '%s/train_test_info/%s/test_instance_idx_mapping.pkl' % (home_dir,num_residues_str)
with open(save_path, 'rb') as f:
    test_instance_idx_mapping = pickle.load(f)



##########################

save_dir = '%s/memmap_data/%s' % (home_dir,num_residues_str)
pdist_filename_memmap =  '%s/pdist_memmap' % save_dir 
pairwise_dist_all_bin_test = load(pdist_filename_memmap, mmap_mode='r')
print(pairwise_dist_all_bin_test.shape)

####################


save_dir = '%s/pairwise_dist/%s' % (home_dir,num_residues_str)
pdist_mean_data = np.load('%s/pdist_mean_data.npy' % save_dir)
pdist_std_data = np.load('%s/pdist_std_data.npy' % save_dir)

pairwise_dist_all_bin_test_norm = apply_norm((pairwise_dist_all_bin_test), pdist_mean_data, pdist_std_data)

####################


minibatch_size_dict = {18: 100, 36: 50, 54: 25}
uniform_sample_minibatch_size = minibatch_size_dict[num_residues] 

for test_dir_idx in test_dir_idx_list: 

    metadata = {}
    metadata['cpu'] = cpuinfo.get_cpu_info()['brand_raw']
    metadata['num_cpu'] = joblib.cpu_count()
    metadata['gpu'] = torch.cuda.get_device_name(0)
    print(metadata)

    start_time = time.time() 
    metadata[test_dir_idx] = {}
    metadata[test_dir_idx]['inference_time'] = [] 

    print(test_dir_idx)

    test_dir_start_idx = test_instance_idx_mapping[test_dir_idx][0]
    test_dir_end_idx = test_dir_start_idx+num_samples
               
    target_idx = np.arange(test_dir_start_idx,test_dir_end_idx)
    
    print(target_idx)

    results_all = [] 

    minibatch_idx_list = np.arange(0,len(target_idx)+uniform_sample_minibatch_size,uniform_sample_minibatch_size)
    minibatch_idx_list[-1] = len(target_idx)
    print(minibatch_idx_list)

    for i in range(0,(len(minibatch_idx_list)-1)):
      
        print('%d/%d' % (i+1,len(minibatch_idx_list)-1))

        curr_target_idx_start = target_idx[minibatch_idx_list[i]]
        curr_target_idx_end = target_idx[minibatch_idx_list[i+1]-1] #-1 to avoid out of bounds error for last
        sample_minibatch_size = minibatch_idx_list[i+1]-minibatch_idx_list[i] #to account for last one because uneven spacing

        print(curr_target_idx_start)
        print(curr_target_idx_end)

        model_x.eval()
        model_y.eval()
        model_z.eval()
        timesteps = list(range(len(noise_scheduler)))[::-1]

        sample = torch.randn(sample_minibatch_size*num_repeat, nout).to(device, dtype=torch.float)
        sample_x = sample
        sample_y = sample
        sample_z = sample 

        #plus1 needed (curr_target_idx_end+1) because curr_target_idx_end = target_idx[minibatch_idx_list[i+1]-1]
        pdist_sample_norm = torch.from_numpy(np.repeat(pairwise_dist_all_bin_test_norm[curr_target_idx_start:(curr_target_idx_end+1),:], repeats=num_repeat, axis=0)).to(device, dtype=torch.float)

        start_inference_time = time.time()        

        for j, t in enumerate(timesteps):

            t_rep = torch.from_numpy(np.repeat(t, sample_minibatch_size*num_repeat)).to(device, dtype=torch.long)

            '''if sample_conditional: old
                model_input = torch.cat((sample,pdist_sample_norm), axis=1)
            else:
                model_input = sample 

            with torch.no_grad():
                residual = model.forward(model_input, t_rep)'''
            
            if sample_conditional:
                with torch.no_grad():
                    residual_x = model_x.forward(pdist_sample_norm, sample_x, t_rep)    
                    residual_y = model_y.forward(pdist_sample_norm, sample_y, t_rep)    
                    residual_z = model_z.forward(pdist_sample_norm, sample_z, t_rep)    
            else:
                with torch.no_grad():
                    residual_x = model_x.forward(sample_x, t_rep)   
                    residual_y = model_y.forward(sample_y, t_rep)    
                    residual_z = model_z.forward(sample_z, t_rep)    
 
            sample_x = noise_scheduler.step(residual_x, t_rep[0], sample_x)
            sample_y = noise_scheduler.step(residual_y, t_rep[0], sample_y)
            sample_z = noise_scheduler.step(residual_z, t_rep[0], sample_z)


        inference_time = time.time()-start_inference_time
        metadata[test_dir_idx]['inference_time'].append(inference_time) 

        #################        

        i = 0 
        idx_start = i*num_residues
        idx_end = (i*num_residues) + num_residues 

        sample_x = apply_reverse_norm(sample_x.cpu().numpy(), xyz_coeff_mean_data[idx_start:idx_end], xyz_coeff_std_data[idx_start:idx_end])
        sample_x = sample_x.reshape(sample_minibatch_size,num_repeat,nout)

        sample_filename_memmap =  '%s/memmap_data/%s/sample_test_dir_idx_%d_true_pdist_x' % (home_dir, num_residues_str, test_dir_idx)
        dump(sample_x, sample_filename_memmap)
        sample_x = load(sample_filename_memmap, mmap_mode='r')

        i = 1 
        idx_start = i*num_residues
        idx_end = (i*num_residues) + num_residues 

        sample_y = apply_reverse_norm(sample_y.cpu().numpy(), xyz_coeff_mean_data[idx_start:idx_end], xyz_coeff_std_data[idx_start:idx_end])
        sample_y = sample_y.reshape(sample_minibatch_size,num_repeat,nout)

        sample_filename_memmap =  '%s/memmap_data/%s/sample_test_dir_idx_%d_true_pdist_y' % (home_dir, num_residues_str, test_dir_idx)
        dump(sample_y, sample_filename_memmap)
        sample_y = load(sample_filename_memmap, mmap_mode='r')

        i = 2
        idx_start = i*num_residues
        idx_end = (i*num_residues) + num_residues 

        sample_z = apply_reverse_norm(sample_z.cpu().numpy(), xyz_coeff_mean_data[idx_start:idx_end], xyz_coeff_std_data[idx_start:idx_end])
        sample_z = sample_z.reshape(sample_minibatch_size,num_repeat,nout)

        sample_filename_memmap =  '%s/memmap_data/%s/sample_test_dir_idx_%d_true_pdist_z' % (home_dir, num_residues_str, test_dir_idx)
        dump(sample_z, sample_filename_memmap)
        sample_z = load(sample_filename_memmap, mmap_mode='r')

        sample_xyz = np.concatenate((sample_x,sample_y,sample_z), axis=2)



        #################

        #results outputs list of len 4, each list corresponding to some # of repeats. because we are processing in minibatches, we append to repeat_X_repeat_all and then
        #we have to select the best minibatch 
        
        #i = 0 
        #results = get_pdist_spline_min_error(sample_xyz, pairwise_dist_all_bin_test[curr_target_idx_start:(curr_target_idx_end+1),:], dij_ee_idx, dij_except_3_apart, dij_except_3_apart_idx, atom_id_matrix, atom_pair_id_list, knot_vector_GS_mean, i, sample_xyz.shape[0], sample_xyz.shape[2])
        #print(results)

        results = Parallel(n_jobs=-1,backend='loky')(delayed(get_pdist_spline_min_error)(num_residues, sample_xyz, pairwise_dist_all_bin_test[curr_target_idx_start:(curr_target_idx_end+1),:], dij_ee_idx, dij_except_3_apart, dij_except_3_apart_idx, atom_id_matrix, atom_pair_id_list, knot_vector_GS_mean, i, sample_xyz.shape[0], sample_xyz.shape[2]) for i in np.arange(curr_target_idx_start, curr_target_idx_end+1))

        #results is a list of lists of len sample_minibatch_size
        if len(results) > 0:
            results_all.append(results)
    
    total_time = time.time()-start_time   
    metadata[test_dir_idx]['total_time'] = total_time 

    if sample_conditional:
        save_dir = '%s/ddpm_samples_true_pdist_conditional_independent_dim/%s/%s/num_samples=%d/num_repeat=%d' % (home_dir,num_residues_str,sample_str,num_samples,num_repeat)
    else:
        save_dir = '%s/ddpm_samples_true_pdist_unconditional_independent_dim/%s/%s/num_samples=%d/num_repeat=%d' % (home_dir,num_residues_str,sample_str,num_samples,num_repeat)

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print(metadata)

    f = open('%s/test_dir_idx=%d.pkl' % (save_dir,test_dir_idx) ,"wb")
    pickle.dump(results_all,f)
    f.close()

    f = open('%s/test_dir_idx_metadata=%d.pkl' % (save_dir,test_dir_idx), "wb")
    pickle.dump(metadata,f)
    f.close() 


    sample_filename_memmap_x = '%s/memmap_data/%s/sample_test_dir_idx_%d_true_pdist_x' % (home_dir, num_residues_str, test_dir_idx)
    sample_filename_memmap_y = '%s/memmap_data/%s/sample_test_dir_idx_%d_true_pdist_y' % (home_dir, num_residues_str, test_dir_idx)
    sample_filename_memmap_z = '%s/memmap_data/%s/sample_test_dir_idx_%d_true_pdist_z' % (home_dir, num_residues_str, test_dir_idx)

    if os.path.isfile(sample_filename_memmap_x):
        os.remove(sample_filename_memmap_x)
    if os.path.isfile(sample_filename_memmap_y):
        os.remove(sample_filename_memmap_y)
    if os.path.isfile(sample_filename_memmap_z):
        os.remove(sample_filename_memmap_z)






