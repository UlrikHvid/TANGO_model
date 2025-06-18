from War_peace_functions_class_based import *
import matplotlib.pyplot as plt
savefig = False
import importlib
import War_peace_functions_class_based as War_peace_functions_class_based
import sys
importlib.reload(sys.modules['War_peace_functions_class_based'])
import scipy as sp
import torch
import networkx as nx

####################################################################################################################################
#Run simulation
####################################################################################################################################

N           = 170000
M           = 28
T           = 365*15
r_old_mean  = 0.2
pexp        = 1.55
lb          = 0.15
ub_list     = [300]
window      = 365
old_dist    = "gamma"
alpha       = 2
T_track     = None
n_couples   = 0

# Initialize peacetime parameters and run the simulation
for ub in ub_list:
    simulation = EpidemicSimulation(N,M,r_old_mean,pexp,old_dist,alpha,ub,lb)
    simulation.initialize_peace_time(T,window,progress=True)
    simulation.run_peace_time_simulation()
    
    ####################################################################################################################################
    # Export network
    ####################################################################################################################################
    
    indices = simulation.indices
    indices_sparse  = torch.sparse_coo_tensor(indices,torch.ones(indices.shape[1]))
    indices_sparse  = indices_sparse.coalesce()
    rows,cols       = indices_sparse.indices()
    strengths       = indices_sparse.values()
    
    network_data = np.column_stack((rows.numpy(),cols.numpy(),strengths.numpy()))
    
    network_header = 'rows,cols,strengths'
    
    np.savetxt(f'Network_N={N:.1e}_ub={ub}.csv', network_data, delimiter=',', header=network_header, comments='', fmt='%d')
    
    ####################################################################################################################################
    # Export connection log
    ####################################################################################################################################
    
    tot_log = simulation.tot_log
    tot_log_sparse  = torch.sparse_coo_tensor(tot_log,torch.ones(tot_log.shape[1]))
    tot_log_sparse  = tot_log_sparse.coalesce()
    rows,cols,t     = tot_log_sparse.indices()
    repeats         = tot_log_sparse.values()
    
    tot_log_data = np.column_stack((rows.numpy(),cols.numpy(),t.numpy(),repeats.numpy()))
    
    tot_log_header = 'rows,cols,t,repeats'
    
    np.savetxt(f'tot_log_N={N:.1e}_ub={ub}.csv', tot_log_data, delimiter=',', header=tot_log_header, comments='', fmt='%d')
    
    ####################################################################################################################################
    # Export member data
    ####################################################################################################################################
    
    tot_enc,tot_par = simulation.analyze_encounters()
    
    # Stack arrays horizontally to create a 2D array
    member_data = np.column_stack((simulation.seeking.numpy(), simulation.r_new.numpy(), simulation.r_old.numpy(), tot_enc.numpy(), tot_par.numpy()))
    
    # Define the header
    member_header = 'seeking,r_new,r_old,tot_enc,tot_par'
    
    fmt = ['%d', '%f', '%f', '%d', '%d']

    # Save to CSV
    np.savetxt(f'Member_data_N={N:.1e}_ub={ub}.csv', member_data, delimiter=',', header=member_header, comments='', fmt='%f')

