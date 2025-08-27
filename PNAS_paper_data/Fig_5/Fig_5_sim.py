import sys
import os
import matplotlib.pyplot as plt
import importlib
import numpy as np
import torch
import pandas as pd
from datetime import date, timedelta

# Get the directory where the current script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(parent_dir)

# Define the data directory relative to the script directory
data_dir = os.path.join(parent_dir, "Saved_networks/")

import tango_model as tango_model
from tango_model import *

N = 170000
M = 28
r_old_mean = 0.2
pexp = 1.55
lb = 0.15
ub = 300
window = 52
old_dist = "gamma"
alpha = 2

network_sparse = pd.read_csv(data_dir + f'Network_N={N:.1e}_ub={ub}.csv')
member_data = pd.read_csv(data_dir + f'Member_data_N={N:.1e}_ub={ub}.csv')
seeking = torch.tensor(member_data["seeking"])
r_new = torch.tensor(member_data["r_new"])
r_old = torch.tensor(member_data["r_old"])

rows = torch.tensor(network_sparse["rows"].values)
cols = torch.tensor(network_sparse["cols"].values)
indices = torch.stack([rows, cols]).to(torch.int32)
strengths = torch.tensor(network_sparse["strengths"].values)
size = torch.Size([N, N])

# Now the log
tot_log_peace_df = pd.read_csv(data_dir + f'tot_log_N={N:.1e}_ub={ub}.csv')
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

# Epidemic parameters
p           = 0.48
TE          = 5 
# TI0         = 18
# TI1         = 16
# TI2         = 8
# TI3         = 4
n           = 2
pre_vac     = 0.13
n_seed      = int(6 * N/85000) # Number of initial imported cases - SCALE WITH N


spain_date  = date(2022, 5, 12)
time_deltas = [0, 7, 14, 21, 28]
vac_date    = date(2022, 8, 8)
dur_vac     = 35
N_vac       = 0
strategy    = "random"
threshold   = 20
Tmax        = 1000 # Generous guess at the maximum possible durations of an epidemic
num_runs    = 50

#Generate TI values from sigmoid function
def sigmoid(t, k, t0):
    TSmax = 14
    TSmin = 6.5
    return TSmax - (TSmax - TSmin) / (1 + np.exp(-k * (t - t0)))

def generate_sigmoid_values(start_date, end_date, t0_date, k):
    date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    t0 = (t0_date - start_date).days

    results = []
    for date in date_range:
        t = (date - start_date).days
        value = sigmoid(t, k, t0)
        results.append((date.strftime("%Y-%m-%d"), value))

    return results

end_date = date(2022, 9, 1) #End of behavior change
t0_date = date(2022, 7, 6) #Date of maximum decrease (fitted)
k = 2.45/30.5 #Fitted and converted from month to day

# Define the path to the CSV file
csv_file_path = os.path.join(script_dir, f'outbreak_size_arr_N={N:.1e}_p={p}_ub={ub}_TIs=sigmoid.csv')

for i, delta in enumerate(time_deltas):
    outbreak_size_arr = torch.zeros(num_runs)
    print(f"delta ={delta}")
    for run in range(num_runs):
        import_date = spain_date + timedelta(days=delta)
        #Tvac = ((vac_date - import_date).days)
        #Tvac = [(Tvac + i) for i in range(dur_vac)] # For gradual vaccination
        Tvac    = None
        #Tmitig1 = 0 if import_date > mitig1_date else ((mitig1_date - import_date).days)
        #Tmitig2 = ((mitig2_date-import_date).days)
        #Tmitig3 = ((mitig3_date-import_date).days)
        # TI_changes  = torch.tensor([[Tmitig1,Tmitig2,Tmitig3],[TI1,TI2,TI3]])
        TSvals      = generate_sigmoid_values(import_date, end_date, t0_date, k)
        TSvals      = np.array([val[1] for val in TSvals])
        TIvals      = 2*(TSvals - TE)
        TI0         = TIvals[0]
        TI_changes  = torch.tensor([np.arange(1,len(TIvals)),TIvals[1:]])
        simulation = EpidemicSimulation(N, M, r_old_mean, pexp, old_dist, alpha, ub, lb)
        simulation.load_state(indices, seeking, r_new, r_old, tot_log_peace, t=0)
        simulation.initialize_war_time(TE,TI0,n,p,TI_changes,pre_vac,Tvac,N_vac) #(TE, TI0, n, p, TI2, TI3, Tmitig1, Tmitig2, pre_vac, Tvac, N_vac)
        simulation.iterate_war_time(t_inf=0, n_seed=n_seed, run=run)
        
        R = simulation.R
        outbreak_size_arr[run] = R.max()
    
    # Convert current `time_delta` data to DataFrame
    delta_df = pd.DataFrame(outbreak_size_arr.numpy(), columns=['outbreak_size'])
    delta_df.insert(0, 'time_delta', delta)  # Insert time_delta as the first column
    
    # Append to CSV or create if it doesn't exist
    if os.path.exists(csv_file_path):
        delta_df.to_csv(csv_file_path, mode='a', header=False, index=False)
    else:
        delta_df.to_csv(csv_file_path, mode='w', header=True, index=False)
