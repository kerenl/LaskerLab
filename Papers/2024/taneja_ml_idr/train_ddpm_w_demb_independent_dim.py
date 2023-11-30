from ddpm import *

import numpy as np  

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
 
import pandas as pd

from pathlib import Path
import os
import glob
import shutil
import math
import time

import pickle 
import sys 

#########
home_dir = '/gpfs/home/itaneja/train_ml_models/18_pointmutation'

arg1 = sys.argv[1]
arg2 = sys.argv[2]
arg3 = sys.argv[3]
arg4 = sys.argv[4]
arg5 = sys.argv[5]

sample_conditional = bool(int(arg1))
num_epochs = int(arg2)
num_residues = int(arg3) 
num_residues_str = 'len%d' % num_residues 
coeff_dim = str(arg4)
num_instances_to_sample = str(arg5)

#########



print('**************')
print(home_dir)
print(sample_conditional)
print(num_epochs)
print(num_residues) 
print('***************')

if num_residues == 18:
    from load_metadata_len18 import *
elif num_residues == 36:
    from load_metadata_len36 import * 
elif num_residues == 54:
    from load_metadata_len54 import * 

class CustomDataset(torch.utils.data.Dataset):
	
    def __init__(self, xyz_coeff, pairwise_dist=None):
        self.xyz_coeff = xyz_coeff
        self.pairwise_dist = pairwise_dist

    def __len__(self):
        return len(self.xyz_coeff)

    def __getitem__(self, index):

        x = self.xyz_coeff[index]

        if self.pairwise_dist is not None:
            y = self.pairwise_dist[index]
        else:
            y = 0

        return x,y



#####################

if num_residues == 18:
    rel_bin_num_list = range(0,10)
else: #update if necessary 
    rel_bin_num_list = [1,2,3,4,5]

######################

save_dir = '%s/xyz_coeff/%s' % (home_dir,num_residues_str)

xyz_coeff_all_bin_train = []
xyz_coeff_all_bin_test = []


for bin_num in rel_bin_num_list:
  
    xyz_coeff_curr_bin_train = np.load('%s/xyz_coeff_train_bin_%d.npy' % (save_dir, bin_num))
    xyz_coeff_curr_bin_test = np.load('%s/xyz_coeff_test_bin_%d.npy' % (save_dir, bin_num))
    
    xyz_coeff_all_bin_train.append(xyz_coeff_curr_bin_train)
    xyz_coeff_all_bin_test.append(xyz_coeff_curr_bin_test)
    

xyz_coeff_all_bin_train = np.vstack(xyz_coeff_all_bin_train)
xyz_coeff_all_bin_test = np.vstack(xyz_coeff_all_bin_test)

if num_instances_to_sample != 'all':
    num_instances_to_sample_int = int(num_instances_to_sample)
    xyz_coeff_all_bin_train = xyz_coeff_all_bin_train[0:num_instances_to_sample_int,:]


print(xyz_coeff_all_bin_train.shape)
print(xyz_coeff_all_bin_test.shape)




xyz_coeff_all_bin_train = xyz_coeff_all_bin_train.reshape(xyz_coeff_all_bin_train.shape[0]*xyz_coeff_all_bin_train.shape[1],xyz_coeff_all_bin_train.shape[2],xyz_coeff_all_bin_train.shape[3])
xyz_coeff_all_bin_train = xyz_coeff_all_bin_train.reshape(xyz_coeff_all_bin_train.shape[0],xyz_coeff_all_bin_train.shape[1]*xyz_coeff_all_bin_train.shape[2])
print(xyz_coeff_all_bin_train.shape)

xyz_coeff_all_bin_test = xyz_coeff_all_bin_test.reshape(xyz_coeff_all_bin_test.shape[0]*xyz_coeff_all_bin_test.shape[1],xyz_coeff_all_bin_test.shape[2],xyz_coeff_all_bin_test.shape[3])
xyz_coeff_all_bin_test = xyz_coeff_all_bin_test.reshape(xyz_coeff_all_bin_test.shape[0],xyz_coeff_all_bin_test.shape[1]*xyz_coeff_all_bin_test.shape[2])
print(xyz_coeff_all_bin_test.shape)


#####################


save_dir = '%s/pairwise_dist/%s' % (home_dir,num_residues_str) 

pairwise_dist_all_bin_train = []
pairwise_dist_all_bin_test = []

for bin_num in rel_bin_num_list:
  
    pairwise_dist_curr_bin_train = np.load('%s/pairwise_dist_train_bin_%d.npy' % (save_dir, bin_num))
    pairwise_dist_curr_bin_test = np.load('%s/pairwise_dist_test_bin_%d.npy' % (save_dir, bin_num))
    
    pairwise_dist_all_bin_train.append(pairwise_dist_curr_bin_train)
    pairwise_dist_all_bin_test.append(pairwise_dist_curr_bin_test)


pairwise_dist_all_bin_train = np.vstack(pairwise_dist_all_bin_train)
pairwise_dist_all_bin_test = np.vstack(pairwise_dist_all_bin_test)

if num_instances_to_sample != 'all':
    num_instances_to_sample_int = int(num_instances_to_sample)
    pairwise_dist_all_bin_train = pairwise_dist_all_bin_train[0:num_instances_to_sample_int,:]


print(pairwise_dist_all_bin_train.shape)
print(pairwise_dist_all_bin_test.shape)


pairwise_dist_all_bin_train = pairwise_dist_all_bin_train.reshape(pairwise_dist_all_bin_train.shape[0]*pairwise_dist_all_bin_train.shape[1],pairwise_dist_all_bin_train.shape[2])
print(pairwise_dist_all_bin_train.shape)

pairwise_dist_all_bin_test = pairwise_dist_all_bin_test.reshape(pairwise_dist_all_bin_test.shape[0]*pairwise_dist_all_bin_test.shape[1],pairwise_dist_all_bin_test.shape[2])
print(pairwise_dist_all_bin_test.shape)

####################


xyz_coeff_mean_data = np.mean(xyz_coeff_all_bin_train,axis=0)
xyz_coeff_std_data = np.std(xyz_coeff_all_bin_train,axis=0)

xyz_coeff_all_bin_train_norm = apply_norm((xyz_coeff_all_bin_train), xyz_coeff_mean_data, xyz_coeff_std_data)
xyz_coeff_all_bin_test_norm = apply_norm((xyz_coeff_all_bin_test), xyz_coeff_mean_data, xyz_coeff_std_data)

del xyz_coeff_all_bin_train
del xyz_coeff_all_bin_test

pdist_mean_data = np.mean(pairwise_dist_all_bin_train,axis=0)
pdist_std_data = np.std(pairwise_dist_all_bin_train,axis=0)

pairwise_dist_all_bin_train_norm = apply_norm((pairwise_dist_all_bin_train), pdist_mean_data, pdist_std_data)
pairwise_dist_all_bin_test_norm = apply_norm((pairwise_dist_all_bin_test), pdist_mean_data, pdist_std_data)

del pairwise_dist_all_bin_train
del pairwise_dist_all_bin_test

###################


if sample_conditional:
    train_set = CustomDataset(xyz_coeff_all_bin_train_norm,pairwise_dist_all_bin_train_norm)
    test_set = CustomDataset(xyz_coeff_all_bin_test_norm,pairwise_dist_all_bin_test_norm)
    ncoords = int(xyz_coeff_all_bin_train_norm.shape[1]/3) #per dimension  
    ndist = pairwise_dist_all_bin_train_norm.shape[1]
    nout = int(xyz_coeff_all_bin_train_norm.shape[1]/3)
    model_name = 'sample_conditional_%d_epochs' % num_epochs   
else:
    train_set = CustomDataset(xyz_coeff_all_bin_train_norm,None)
    test_set = CustomDataset(xyz_coeff_all_bin_test_norm,None)
    ncoords = int(xyz_coeff_all_bin_train_norm.shape[1]/3)
    nout = int(xyz_coeff_all_bin_train_norm.shape[1]/3)
    model_name = 'sample_unconditional_%d_epochs' % num_epochs


print(ncoords)
print(ndist)
print(nout)

  
torch.manual_seed(0)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, drop_last=False, shuffle=True)
torch.manual_seed(0)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, drop_last=False, shuffle=True)


#############

hidden_size = 128 
embedding_size = 16 
hidden_layers = 2
time_embedding =  'linear' #
input_embedding = 'identity'
d_emb = math.floor(ndist**.5) #10 

num_timesteps = 100

if coeff_dim == 'x':
    i = 0 
elif coeff_dim == 'y':
    i = 1
elif coeff_dim == 'z':
    i = 2


print('******************')
print('dim=%s' % coeff_dim)
print('******************')

idx_start = i*num_residues
idx_end = (i*num_residues) + num_residues 

model = MLP_dist_emb(ndist,d_emb,ncoords,nout,num_timesteps, 
    hidden_size=hidden_size,
    hidden_layers=hidden_layers,
    emb_size=embedding_size,
    time_emb=time_embedding,
    input_emb=input_embedding)

model.to(device)

noise_scheduler = NoiseScheduler(
  num_timesteps=num_timesteps,
beta_schedule="linear")


learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

model_info_dict = {}
model_info_dict['num_residues'] = num_residues 
model_info_dict['ndist'] = ndist
model_info_dict['d_emb'] = d_emb
model_info_dict['learning_rate'] = learning_rate 
model_info_dict['num_epochs'] = num_epochs
model_info_dict['ncoords'] = ncoords
model_info_dict['nout'] = nout
model_info_dict['hidden_size'] = hidden_size
model_info_dict['embedding_size'] = embedding_size
model_info_dict['hidden_layers'] = hidden_layers
model_info_dict['time_embedding'] = time_embedding
model_info_dict['input_embedding'] = input_embedding 
model_info_dict['num_timesteps'] = num_timesteps 
model_info_dict['coeff_dim'] = coeff_dim
model_info_dict['train_losses'] = []
model_info_dict['test_losses'] = [] 


print("Training model...")


for epoch in range(num_epochs):
   
    model.train()

    print('epoch = %d' % epoch)
    train_losses_curr_epoch = [] 

    for step, x in enumerate(train_loader):
      
        if step % 1000 == 0:
            print(step)

        xyz_coeff = x[0]
        pairwise_dist = x[1]
                        
        xyz_coeff = xyz_coeff.to(device, dtype=torch.float)
        pairwise_dist = pairwise_dist.to(device, dtype=torch.float)

        xyz_coeff_dimi = xyz_coeff[:,idx_start:idx_end]
                      
        noise = torch.randn(xyz_coeff_dimi.shape).to(device, dtype=torch.float)
        timesteps = torch.randint(0, noise_scheduler.num_timesteps, (xyz_coeff_dimi.shape[0],)).to(device, dtype=torch.long)

        noisy = noise_scheduler.add_noise(xyz_coeff_dimi, noise, timesteps)

        optimizer.zero_grad()

        if sample_conditional:
            noise_pred = model.forward(pairwise_dist, noisy, timesteps)
        else:
            noise_pred = model.forward(noisy, timesteps)

        loss = F.l1_loss(noise_pred, noise)
        loss.backward(loss)

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
   
        train_losses_curr_epoch.append(loss.detach().item()) 

    mean_train_loss_curr_epoch = np.mean(train_losses_curr_epoch)

    print('epoch %d train loss %.4f' % (epoch, mean_train_loss_curr_epoch))

    print('test')
    mean_test_loss_curr_epoch = eval_loss_w_demb_independent_dim(model, noise_scheduler, sample_conditional, test_loader, idx_start, idx_end)

    model_info_dict['train_losses'].append([epoch, mean_train_loss_curr_epoch])
    model_info_dict['test_losses'].append([epoch, mean_test_loss_curr_epoch])

  
    if epoch % 1 == 0:
        print('epoch %d train loss %.4f, test loss %.4f' % (epoch, mean_train_loss_curr_epoch, mean_test_loss_curr_epoch))


sample_str = 'sample=%s' % num_instances_to_sample
model_save_dir = '/gpfs/home/itaneja/train_ml_models/18_pointmutation/ddpm_model_w_demb_independent_dim/%s/%s/%s' % (num_residues_str,coeff_dim,sample_str)
Path(model_save_dir).mkdir(parents=True, exist_ok=True)

model_save_path = '%s/%s.pth' % (model_save_dir, model_name)
torch.save(model.state_dict(), model_save_path)
print('saving model at %s' % model_save_path)

model_info_path = '%s/%s.pkl' % (model_save_dir, model_name)
f = open(model_info_path,"wb")
pickle.dump(model_info_dict,f)
f.close()



     
