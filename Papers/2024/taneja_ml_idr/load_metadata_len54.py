import numpy as np 


home_dir = '/gpfs/home/itaneja/train_ml_models/18_pointmutation'
save_dir = '%s/metadata/len54' % home_dir

with open('%s/atom_id_matrix.npy' % save_dir, 'rb') as f:
    atom_id_matrix = np.load(f,allow_pickle=True)

atom_pair_id_list = []

for i in range(0,atom_id_matrix.shape[0]):
    atom_pair_id_list.append('%d_%d' % (atom_id_matrix[i,0],atom_id_matrix[i,1]))
    
for i in range(0,len(atom_id_matrix)):
    if atom_id_matrix[i][0] == 1 and atom_id_matrix[i][1] == 54:
        dij_ee_idx = i 

print(dij_ee_idx)
 
dij_except_3_apart = []
dij_except_3_apart_idx = []

for i in range(0,len(atom_id_matrix)):
    if abs(atom_id_matrix[i][0]-atom_id_matrix[i][1]) >= 4:
        dij_except_3_apart.append([int(atom_id_matrix[i][0]),int(atom_id_matrix[i][1])])
        dij_except_3_apart_idx.append(i)
      
print('loaded atom_id_matrix, ddpm_dij, and dij_except_3_apart')
