import numpy as np 
import pandas as pd  
import math
import sys 

from scipy.interpolate import splprep, splev

import numba


#xyz_res corresponds to xyz coordinates for values of t ranging between 0 and 1 spaced .001 apart 
#this is to avoid recomputing coordinates for a given t (which may occur because we explore different values of bond length) 
@numba.njit(fastmath=True,cache=True,nopython=True)
def get_res_t_dict_linear_precompute_coords_coarse_fine_nb(xyz_res, search_delta, max_t, num_residues):
  
  bond_len_ub_lb_delta = .01 #final bond_len can be within +- .02 (by construction) 
  
  final_bond_len_tolerance = .01 #final_bond_len can be within .01 of actual ;;; so if final determiend bond_len is .36, final can be .35 ;;; this is because we were discarding lots of conformations otherwise 
  
  epsilon = 1e-15
  
  #final coordinate can be placed no less than min_t, no more than max_t
  min_t = .95 
    
  initial_bond_len = .38 
  bond_len = initial_bond_len

  end = -1
  final_t_prev = -1
  
  iter_num = 0 

  res_t_list_prev = np.zeros((num_residues-1,3)) 

  while (end <= 1):
    
    res_t_list = np.zeros((num_residues-1,3))

    curr_res_num = 2 

    start = 0 
    end = search_delta

    while curr_res_num <= num_residues: #keep increasing search_delta until we hit bond_len

      end_candidates = np.arange(start+.01,min(max_t,start+.1),.01) #start with a coarse search [spaced .01 apart] for end_t to save time 
            
      start_idx = math.ceil(start/search_delta)
      end_candidates_idx = np.ceil(end_candidates/search_delta)
            
      for i in range(0,len(end_candidates_idx)):
      
        a = (xyz_res[start_idx,0] - xyz_res[int(end_candidates_idx[i]),0])**2
        b = (xyz_res[start_idx,1] - xyz_res[int(end_candidates_idx[i]),1])**2
        c = (xyz_res[start_idx,2] - xyz_res[int(end_candidates_idx[i]),2])**2
        curr_seg_dist = np.round((a+b+c)**.5,3)

        if curr_seg_dist >= (bond_len-epsilon): 
          if i > 0:
            end = np.round(end_candidates[i-1],3) #start searching for the optimal end_t at the previous end_candidate  
          break
            
      seg_dist = 0.
      prev_seg_dist = 0.
      prev_end = end 

      while (seg_dist <= bond_len) and (end <= max_t):

        start_idx = math.ceil(start/search_delta)
        end_idx = math.ceil(end/search_delta)
        
        a = (xyz_res[start_idx,0] - xyz_res[end_idx,0])**2
        b = (xyz_res[start_idx,1] - xyz_res[end_idx,1])**2
        c = (xyz_res[start_idx,2] - xyz_res[end_idx,2])**2
        seg_dist = np.round((a+b+c)**.5,3)

        if seg_dist <= (bond_len+epsilon):
          prev_seg_dist = seg_dist
          prev_end = end
          end += search_delta
          end = np.round(end,3)
                  
      res_t_list[curr_res_num-2,0] = prev_seg_dist
      res_t_list[curr_res_num-2,1] = start
      res_t_list[curr_res_num-2,2] = prev_end 
      
      curr_res_num += 1
      start = np.round(prev_end,3)
      end = start+search_delta
      start = np.round(start,3)
      end = np.round(end,3)
            

    final_t = res_t_list[-1,2]
    final_seg_dist = res_t_list[-1,0]
 
    #if on the first iteration, final_t > 1, that means we overshot so we need to decrease bond_len 
    #if on the first iteration, final_seg_dist < (bond_len-final_bond_len_tolerance)  that means we overshot. this can occur because end is not allowed to be more than max_t  
    #keep doing this as needed
    #last seg_dist needs to be close to determined bond_len 
    if iter_num == 0 and ((final_t > 1) or (final_seg_dist < (bond_len-final_bond_len_tolerance))):
      
      bond_len -= .01
      bond_len = np.round(bond_len,3)
      end = -1
      iter_num = 0 #this is to give us one more try 
            
      if bond_len < (initial_bond_len-bond_len_ub_lb_delta): #this is a 'bad' starting conformation 
        return (np.zeros((num_residues-1,3)), bond_len)
      
    else: #increase bond_len until final_t is at maximum
      
      if final_t >= 1:
        
        curr_delta = abs(final_t-1)
        prev_delta = abs(final_t_prev-1)
        
        if (final_seg_dist < (bond_len-final_bond_len_tolerance)):
          return (np.zeros((num_residues-1,3)), bond_len)
        
        if prev_delta < curr_delta: #last one is better
          if final_t_prev <= min_t:
            return (np.zeros((num_residues-1,3)), bond_len)
          else:
            return (res_t_list_prev, bond_len)
        else: #this is one is better
          if final_t <= max_t:
            return (res_t_list, bond_len)
          else:
            return (np.zeros((num_residues-1,3)), bond_len)
            
      else:
        
        res_t_list_prev = np.copy(res_t_list)
        final_t_prev = res_t_list_prev[-1,2]
        
      bond_len += .001
      bond_len = np.round(bond_len,3)
      iter_num += 1 
      
      if bond_len > (initial_bond_len+bond_len_ub_lb_delta): #this is a 'bad' starting conformation 
        return (np.zeros((num_residues-1,3)), bond_len)
      
    
  return



def get_pairwise_dist_spline_vectorized(tck, res_t_dict, atom_id_matrix, atom_pair_id_list=None, convert_to_square=False):
  
  if res_t_dict[0,0] == 0: 
    return [] 
  
  pairwise_dist_all_ij_wide = [] 
 
  splev_t_input_list = []
 
  for atom_pair in atom_id_matrix:
    
    res_i = int(atom_pair[0])
    res_j = int(atom_pair[1])
    
    if res_i == 1:
      res_i_t = 0 
    else:
      res_i_t = res_t_dict[res_i-2][2] #minus 2 because res_t_dict is from res 2 to num_residues
      
    if res_j == 1:
      res_j_t = 0 
    else:
      res_j_t = res_t_dict[res_j-2][2]
    
    splev_t_input_list.extend([res_i_t,res_j_t])

  #x/y/z_res is of shape len(splev_t_input_list)
  #np.array([x_res,y_res,z_res]) is of shape len(splev_t_input_list),3
  ##i.e [[x1,x2...xn],[y1,y2...yn],[z1,z2...zn]]
  ##transpose is [[x1,y1,z1],[x2,y2,z2],...[xn,yn,zn]]
  
  x_res, y_res, z_res = splev(splev_t_input_list, tck)
  xyz_res = np.array([x_res,y_res,z_res]).T
  xyz_diff = np.diff(xyz_res, axis=0) #take diff along rows 
  #from xyz_diff, we only want every other value because splev_t_input_list consists of pairs (i.e we want diff between [x2,y2,z2]/[x1,y1,z1] and [x4,y4,z4]/[x3,y3,z3] but not intermediate)
  xyz_diff = xyz_diff[::2]
  pairwise_dist_all_ij_wide = np.sqrt((xyz_diff ** 2).sum(axis=1))

    
  if convert_to_square:
    pairwise_dist_all_ij_square = convert_wide_to_square_matrix(np.array(pairwise_dist_all_ij_wide)[None,:], atom_pair_id_list, atom_id_matrix, False)
    return pairwise_dist_all_ij_square
  else:
    return pairwise_dist_all_ij_wide










def get_pdist_spline_min_error(num_residues, xyz_coeff_sampled, pairwise_dist_all_bin_test_subset, dij_ee_idx, dij_except_3_apart, dij_except_3_apart_idx, atom_id_matrix, atom_pair_id_list, knot_vector_GS_mean, i, dim1, dim2):

  #basic idea is to iterate across all samples generated for a fixed set of distance constraints and find the best one
  ####
  
  pairwise_dist_spline_min_pdist_error = [] 
  pairwise_dist_spline_square_min_pdist_error = []
  xyz_coeff_sampled_min_pdist_error = np.zeros((dim1,dim2))

  min_pdist_error_sample_all_repeat = []
  min_pdist_error_sample = [] 

  pairwise_dist_spline_subset_dij_list = [] 
  #print('num_sample=%d' % i)
  non_ignored_repeat_idx = [] 

  
  search_delta = .001
  max_t = 1.01
  ee_dist_tol_percentage = .2 
  
  num_ignored = []

  #i is the idx with respect to entire data;;; since we are evaluating in batches we have to take mod 
  mod_idx = (i % xyz_coeff_sampled.shape[0]) 
  target = pairwise_dist_all_bin_test_subset[mod_idx,dij_except_3_apart_idx] #target should only consist of non_ignored instances 
  target_ee_dist = pairwise_dist_all_bin_test_subset[mod_idx,dij_ee_idx]
  target_ee_dist_lb = (1-ee_dist_tol_percentage)*target_ee_dist
  target_ee_dist_ub = (1+ee_dist_tol_percentage)*target_ee_dist

  num_nonignored = 0 
  
  total_conformations_explored = xyz_coeff_sampled.shape[1]
  min_conformations_explored = xyz_coeff_sampled.shape[1]-5000
  min_conformations_generated = int(min_conformations_explored*.01) 
  
  for j in range(0,xyz_coeff_sampled.shape[1]):

    #if j % 100 == 0:
    #  print('num_repeat=%d, sample=%d' % (j,i))

    u_fine = np.linspace(0,1,100)

    tck = [np.array(knot_vector_GS_mean), xyz_coeff_sampled[mod_idx,j,:].reshape(3,num_residues), 5]

    t_range = np.arange(0,max_t+search_delta,search_delta).round(3) 
    x_res, y_res, z_res = splev(t_range, tck)
    xyz_res = np.array([x_res,y_res,z_res]).T
    
    ee_diff = xyz_res[-1,:] - xyz_res[0,:]
    sampled_ee_dist = np.sqrt((ee_diff ** 2).sum(axis=0))

    if (sampled_ee_dist >= target_ee_dist_lb) and (sampled_ee_dist <= target_ee_dist_ub):
        ee_within_tol = True
    else:
        ee_within_tol = False 

    #print('target %.3f' % target_ee_dist)
    #print('sampled_ee %.3f' % sampled_ee_dist)
  
    if ee_within_tol:
        res_t_dict, bond_len = get_res_t_dict_linear_precompute_coords_coarse_fine_nb(xyz_res, search_delta, max_t, num_residues)
        if res_t_dict[0,0] != 0:
          pairwise_dist_spline_subset_dij = get_pairwise_dist_spline_vectorized(tck, res_t_dict, dij_except_3_apart) #only subset of ij used 
          pairwise_dist_spline_subset_dij_list.append(pairwise_dist_spline_subset_dij)
          non_ignored_repeat_idx.append(j)
          num_ignored.append(0)
          num_nonignored += 1 
        else:
          num_ignored.append(1)
    else:
        num_ignored.append(1)  

    '''#early stopping condition -- 
    if (num_nonignored >= min_conformations_generated) and ((j+1) >= min_conformations_explored):
        break'''  
        
      

  num_ignored = np.array(num_ignored)

  if len(pairwise_dist_spline_subset_dij_list) == 0:
    return [] 

  
  pairwise_dist_spline_subset_dij_list = np.array(pairwise_dist_spline_subset_dij_list)
  abs_error = np.abs(target - pairwise_dist_spline_subset_dij_list)
  mean_abs_error = np.mean(abs_error,axis=1) #average error across cols for each dij

  out_all_repeat_num = []

  if total_conformations_explored == 20000:
    repeat_num_subset_list = [1000,5000,10000,15000,'total']
  elif total_conformations_explored == 15000:
    repeat_num_subset_list = [1000,2500,5000,10000,'total']
  elif total_conformations_explored == 10000:
    repeat_num_subset_list = [1000,2500,5000,7500,'total']

  for repeat_num_subset in repeat_num_subset_list:
    
    if repeat_num_subset != 'total':
      stop_idx = np.argmin(np.abs(np.array(non_ignored_repeat_idx) - repeat_num_subset)) #get closest idx to 1000,2000,3000,etc. ;; non_ignored_repeat_idx looks like [2,10,15,99,997,1002...etc]
      min_pdist_error_repeat = np.min(mean_abs_error[0:(stop_idx+1),]) #mean_abs_error has entry for each one in non_ignored_repeat_idx 
      min_pdist_error_repeat_idx = np.argmin(mean_abs_error[0:(stop_idx+1),])
      num_instances = repeat_num_subset 
      num_ignored_sum = np.sum(num_ignored[0:repeat_num_subset])
    else:
      min_pdist_error_repeat = np.min(mean_abs_error)
      min_pdist_error_repeat_idx = np.argmin(mean_abs_error)
      num_instances = len(num_ignored)
      num_ignored_sum = np.sum(num_ignored)

    ##### get pairwise_dist_spline_all_dij for minimum error ##### 

    xyz_coeff_sampled_min_pdist_error = np.copy(xyz_coeff_sampled[mod_idx,non_ignored_repeat_idx[min_pdist_error_repeat_idx],:]) #shape is 1,54 ;; we have to np.copy because too many memmap files open otherwise 
    tck = [np.array(knot_vector_GS_mean), xyz_coeff_sampled_min_pdist_error.reshape(3,num_residues), 5]

    t_range = np.arange(0,max_t+search_delta,search_delta).round(3) 
    x_res, y_res, z_res = splev(t_range, tck)
    xyz_res = np.array([x_res,y_res,z_res]).T
      
    res_t_dict, bond_len = get_res_t_dict_linear_precompute_coords_coarse_fine_nb(xyz_res, search_delta, max_t, num_residues)

    if res_t_dict[0,0] != 0:
      pairwise_dist_spline_all_dij_min_error = get_pairwise_dist_spline_vectorized(tck, res_t_dict, atom_id_matrix)  
    else:
      print('error')
      sys.exit()

    #####

    out = [min_pdist_error_repeat, i, non_ignored_repeat_idx[min_pdist_error_repeat_idx], pairwise_dist_spline_subset_dij_list[min_pdist_error_repeat_idx,], pairwise_dist_spline_all_dij_min_error, xyz_coeff_sampled_min_pdist_error, num_instances, num_ignored_sum, res_t_dict]
    out_all_repeat_num.append(out)

  return out_all_repeat_num 



def convert_wide_to_square_matrix(data, atom_pair_id_list, atom_id_matrix, index=None, include_df=False):

  tmp = data
    
  if index is not None:
    tmp = tmp[index:(index+1),:]
  
  df = pd.DataFrame(tmp, columns = atom_pair_id_list)
  dfm = df.melt().reset_index()
  dfm = dfm[['variable','value']]

  dfm['atom_id1'] = np.tile(atom_id_matrix[:,0]-1, tmp.shape[0])
  dfm['atom_id2'] = np.tile(atom_id_matrix[:,1]-1, tmp.shape[0])

  dfm['atom_id1'] = (dfm['atom_id1']).astype(int)
  dfm['atom_id2'] = (dfm['atom_id2']).astype(int)

  dfm_sq = dfm.pivot(index='atom_id2', columns='atom_id1', values='value')      

  dfm_sq_np = dfm_sq.to_numpy() #lower triangular 
  dfm_sq_np = np.tril(dfm_sq_np) + np.triu(dfm_sq_np.T, 1)
  
  if include_df:
    return(dfm_sq_np, dfm_sq)
  else:
    return dfm_sq_np



def get_xyz_spline(tck, num_residues, search_delta=.001, max_t=1.01, res_t_dict=None):
     
  if res_t_dict is None:
    res_t_dict, bond_len = get_res_t_dict_linear_precompute_coords_coarse_fine_nb(xyz_res, search_delta, max_t, num_residues)
  
  if res_t_dict[0,0] == 0:
    return [],[],[],{}
  
  x_all_res = [] 
  y_all_res = [] 
  z_all_res = [] 
  
  for res in range(1,num_residues+1):
    
    if res == 1:
      res_t = 0 
    else:
      res_t = res_t_dict[res-2][2]
                  
    x_res, y_res, z_res = splev([res_t], tck)
    
    x_all_res.append(x_res[0])
    y_all_res.append(y_res[0])
    z_all_res.append(z_res[0])
    
  return x_all_res,y_all_res,z_all_res,res_t_dict

