import sys
import os
import matplotlib.pyplot as plt
import importlib
# Get the directory where the current script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
# Define the data directory relative to the script directory
data_dir = os.path.join(parent_dir, "Saved_networks/")
serial_interval_dir = os.path.join(parent_dir, "Data/2022_epidemic/")

import tango_model as tango_model
from tango_model import *
import torch
import pandas as pd
from datetime import date

# # Function to load existing data if file exists
# def load_existing_data(file_name):
#     if os.path.exists(file_name):
#         return np.loadtxt(file_name, delimiter=',')
#     return None

# # Save or append new data
# def save_or_append_data(file_name, new_data, fmt):
#     existing_data = load_existing_data(file_name)
    
#     if existing_data is not None:
#         # Concatenate the existing data with the new data
#         updated_data = np.vstack([existing_data, new_data])
#     else:
#         # No existing data, just use the new data
#         updated_data = new_data
    
#     # Save the updated data (overwrite the file with the concatenated data)
#     np.savetxt(file_name, updated_data, delimiter=',', fmt=fmt)

def load_existing_data(file_name):
    if os.path.exists(file_name):
        return np.loadtxt(file_name, delimiter=',')
    return None

# Save or append new data
def save_or_append_data(file_name, new_data, fmt):
    # Make sure new_data is a (1 × N) row array
    new_data = np.atleast_2d(new_data)
    existing_data = load_existing_data(file_name)
    if existing_data is not None:
        # Stack as another row
        updated_data = np.vstack([existing_data, new_data])
    else:
        # First write: still a 2-D row
        updated_data = new_data
    # Overwrite with the correctly shaped 2-D array
    np.savetxt(file_name, updated_data, delimiter=",", fmt=fmt)

N           = 170000
M           = 28
r_old_mean  = 0.2
pexp        = 1.55 
lb          = 0.15
ub          = 300
old_dist    = "gamma"
alpha       = 2

if pexp == 1.55:
    network_sparse  = pd.read_csv(data_dir + f'Network_N={N:.1e}_ub={ub}.csv')
    member_data     = pd.read_csv(data_dir + f'Member_data_N={N:.1e}_ub={ub}.csv')
else:
    network_sparse  = pd.read_csv(data_dir + f'Network_N={N:.1e}_ub={ub}_pexp={pexp}.csv')
    member_data     = pd.read_csv(data_dir + f'Member_data_N={N:.1e}_ub={ub}_pexp={pexp}.csv')
seeking         = torch.tensor(member_data["seeking"])
r_new           = torch.tensor(member_data["r_new"])
r_old           = torch.tensor(member_data["r_old"])

rows            = torch.tensor(network_sparse["rows"].values)
cols            = torch.tensor(network_sparse["cols"].values)
indices         = torch.stack([rows,cols]).to(torch.int32)
strengths       = torch.tensor(network_sparse["strengths"].values)
size            = torch.Size([N,N])

#Now the log
tot_log_peace_df = pd.read_csv(data_dir + f'tot_log_N={N:.1e}_ub={ub}.csv')

# Convert each column of the DataFrame to a tensor and stack them to form a 3xN tensor
tot_log_peace_tensor = torch.tensor([tot_log_peace_df["rows"].values, 
                                     tot_log_peace_df["cols"].values, 
                                     tot_log_peace_df["t"].values])

# Extract the third row (time coordinates)
t = tot_log_peace_tensor[2, :]

# Get the sorted indices of the third row
sorted_indices = torch.sort(t, dim=0).indices

# Use the sorted indices to sort the entire tensor
tot_log_peace = tot_log_peace_tensor[:, sorted_indices]

# Transpose the time coordinates
max_t = tot_log_peace[2].max()
tot_log_peace[2] -= (max_t + 1)

#Epidemic parameters
p           = 0.48
TE          = 5 
n           = 2 
pre_vac     = 0.13
#n_seed      = int(3 * N/85000) #Number of initial imported cases - SCALES WITH N
n_seed      = 7 ###### !!!!!!! #######

import_date = date(2022, 5, 19)
vac_date    = date(2022, 8, 8)

# Input parameters
end_date = date(2022, 9, 1) #End of behavior change
t0_date = date(2022, 7, 6) #Date of maximum decrease (fitted)
k = 2.45/30.5 #Fitted and converted from month to day

name        = "Counterfactual"  ###### !!!!!!! #######
if name == "Counterfactual":
    TI_changes  = None
    TI0         = 18
else:
    # Generate sigmoid values
    TSvals      = generate_sigmoid_values(import_date, end_date, t0_date, k)
    TSvals      = np.array([val[1] for val in TSvals])
    TIvals      = 2*(TSvals - TE)
    TI0         = TIvals[0]
    TI_changes  = torch.tensor([np.arange(1,len(TIvals)),TIvals[1:]])

Tmax        = 1000 #Generous guess at the maximum possible durations of an epidemic
num_runs    = 20

# Define file names
if TI_changes is not None:
    daily_file  = os.path.join(script_dir, f'daily_' + name + f'_N={N}_p={p}_pexp={pexp}_n_seed={n_seed}_t_import={import_date.day}-{import_date.month}_ub={ub}_TIs=sigmoid.csv')
    Reff_file   = os.path.join(script_dir, f'Reff_' + name + f'_N={N}_p={p}_pexp={pexp}_n_seed={n_seed}_t_import={import_date.day}-{import_date.month}_ub={ub}_TIs=sigmoid.csv')
else:
    daily_file  = os.path.join(script_dir, f'daily_' + name + f'_N={N}_p={p}_pexp={pexp}_n_seed={n_seed}_t_import={import_date.day}-{import_date.month}_ub={ub}_TIs=None.csv')
    Reff_file   = os.path.join(script_dir, f'Reff_' + name + f'_N={N}_p={p}_pexp={pexp}_n_seed={n_seed}_t_import={import_date.day}-{import_date.month}_ub={ub}_TIs=None.csv')

for run in range(num_runs):
    daily_arr   = torch.zeros((Tmax), dtype=torch.long)
    Reff_arr    = torch.zeros((Tmax))
    simulation  = EpidemicSimulation(N,M,r_old_mean,pexp,old_dist,alpha,ub,lb)
    simulation.load_state(indices,seeking,r_new,r_old,tot_log_peace,t=0)
    simulation.initialize_war_time(TE,TI0,n,p,TI_changes,pre_vac)
    simulation.iterate_war_time(t_inf=0,n_seed=n_seed,run=run,Tmax=Tmax)
    time_points,Reff_values         = simulation.moving_window_reff(window_size=10)
    S,E,I,R                         = simulation.S,simulation.E,simulation.I,simulation.R
    dur                             = len(E)
    _,daily_arr_temp                = simulation.calculate_new_cases()
    daily_arr[:dur]                 = daily_arr_temp[:dur].round().to(torch.long)
    Reff_arr[:len(Reff_values)] = Reff_values
    # Save or append IE data
    save_or_append_data(daily_file, daily_arr.cpu().numpy(), fmt='%d')
    # Save or append Reff data
    save_or_append_data(Reff_file, Reff_arr.numpy(), fmt='%f')

