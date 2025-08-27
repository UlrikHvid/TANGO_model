import sys
import os
import matplotlib.pyplot as plt
import importlib
import matplotlib.ticker as mticker  # <-- Added import
# Get the directory where the current script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
# Define the data directory relative to the script directory
data_dir = os.path.join(parent_dir, "Saved_networks/")
serial_interval_dir = os.path.join(parent_dir, "Data/2022_epidemic/")
from scipy.optimize import curve_fit
import numpy as np
from scipy.special import expit

import MPox.tango_model as tango_model
from MPox.tango_model import *
import torch
import pandas as pd
from datetime import date

def compute_weighted_mean(data_arr, weight_arr, Tmax_plot):
    """
    Compute, for each t in [0..Tmax_plot-1], the weighted mean of data_arr[:, t],
    using weight_arr[:, t] as weights—but only over those repeats where data_arr is finite.
    If, after removing NaNs, the total weight is zero, return np.nan for that t.

    Parameters:
        data_arr:   np.ndarray of shape (num_repeats, Tmax_plot)
        weight_arr: same shape, nonnegative weights (e.g. number of cases)
        Tmax_plot:  integer length of the time‐axis

    Returns:
        wm: 1D array of length Tmax_plot, where
            wm[t] = (Σᵢ data_arr[i, t] * weight_arr[i, t]) / (Σᵢ weight_arr[i, t]),
            summing only over i where data_arr[i, t] is not NaN.
            If no valid i or Σ weights = 0, wm[t] = np.nan.
    """
    data_arr   = np.asarray(data_arr,   dtype=float)
    weight_arr = np.asarray(weight_arr, dtype=float)
    wm = np.full(Tmax_plot, np.nan, dtype=float)

    for t in range(Tmax_plot):
        d = data_arr[:, t]
        w = weight_arr[:, t]

        # Keep only those repeats where d is finite
        valid = ~np.isnan(d)
        if not np.any(valid):
            # no finite data at this t ⇒ leave wm[t] = nan
            continue

        d_valid = d[valid]
        w_valid = w[valid]
        total_w = np.sum(w_valid)

        if total_w > 0:
            wm[t] = np.sum(d_valid * w_valid) / total_w
        else:
            # all weights are zero ⇒ undefined ⇒ nan
            wm[t] = np.nan

    return wm

def estimate_k_flat(inf_times_flat, sec_infs_flat, ti, tf):
    """
    Stand‐alone helper: 
    inf_times_flat: 1‐D numpy or 1‐D torch (length = num_repeats * N)
    sec_infs_flat: 1‐D numpy or 1‐D torch (same length)
    ti, tf: integers (window boundaries, inclusive of ti, exclusive of tf)
    """
    # Convert to NumPy if needed
    if isinstance(inf_times_flat, torch.Tensor):
        inf = inf_times_flat.cpu().numpy()
        sec = sec_infs_flat.cpu().numpy()
    else:
        inf = inf_times_flat
        sec = sec_infs_flat

    # Mask = agents (across all repeats) whose infection time ∈ [ti, tf]
    # (Original code used `>= ti & <= tf`; here we'll do >= ti and < tf+1
    #  to mirror “include tf”. In practice either ≥ti and ≤tf is fine.)
    mask = (inf >= ti) & (inf <= tf)
    if not np.any(mask):
        return np.nan

    selected_offspring = sec[mask]
    # R0 = mean of those offspring counts
    R0 = selected_offspring.mean()
    var = selected_offspring.var(ddof=1)  # sample variance

    # If var == R0, formula would divide by zero → return nan
    if np.isclose(var, R0):
        return np.nan

    k_val = R0**2 / (var - R0)
    return k_val


def moving_window_k_all(inf_times_all, sec_infs_all, Tmax, window_size):
    """
    Compute a 1‐D array of k(t) by sliding a window of width `window_size`
    over t=0..Tmax-1, pooling all `inf_times_all` and `sec_infs_all` from 
    every repeat.

    inf_times_all, sec_infs_all: 2‐D arrays of shape (num_repeats, N)
    Tmax: integer
    window_size: integer

    Returns:
      - time_points: 1‐D np.ndarray of ints 0..Tmax-1
      - k_values:    1‐D np.ndarray of length Tmax, containing k(t) or np.nan
    """
    # Convert to NumPy if they're Torch tensors
    if isinstance(inf_times_all, torch.Tensor):
        inf_all = inf_times_all.cpu().numpy()
        sec_all = sec_infs_all.cpu().numpy()
    else:
        inf_all = inf_times_all
        sec_all = sec_infs_all

    # Flatten so that length = (num_repeats * N)
    inf_flat = inf_all.reshape(-1)
    sec_flat = sec_all.reshape(-1)

    time_points = np.arange(0, Tmax)
    k_values    = np.full(Tmax, np.nan, dtype=float)

    half_w = window_size // 2

    for t in time_points:
        ti = max(1, t - half_w)
        tf = min(Tmax - 1, t + half_w)  # inclusive window-end
        k_val = estimate_k_flat(inf_flat, sec_flat, ti, tf)
        if not np.isnan(k_val):
            k_values[t] = k_val

    return time_points, k_values

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

network_sparse  = pd.read_csv(data_dir + f'Network_N={N:.1e}_ub={ub}.csv')
member_data     = pd.read_csv(data_dir + f'Member_data_N={N:.1e}_ub={ub}.csv')
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
n_seed      = int(3.5 * N/85000) #Number of initial imported cases - SCALES WITH N

import_date = date(2022, 5, 19)
# Input parameters
end_date = date(2022, 9, 1) #End of behavior change

name        = "Europe_poolk"

###############################################################
# Import data for serial interval #
###############################################################

# Data from the table

SI_dates = [date(2022, 5, 15), date(2022, 6, 15), date(2022, 7, 15), date(2022, 8, 15)]
x_values_fit = [(SI_date - import_date).days + 1 for SI_date in SI_dates] # Units of days from import date (first value negative)

# Mean and standard deviation (sd) values (for TS)
mean = [13.95, 12.92, 9.00, 6.88]
N_pairs = [41, 31, 34, 15]  # Number of contact tracing pairs in each datapoint
sd = [11.39, 11.39, 7.07, 4.93]
sd = [sd[i] / np.sqrt(N_pairs[i]) for i in range(len(N_pairs))]

TE = 5  # given in the problem

k = 2.5/30.5 # Units: /day

def sigmoid3(x, t0, TSmax, TSmin): # Fitting function
    return TSmax - (TSmax - TSmin) / (1 + np.exp(-k * (x - t0)))

p0 = [7.0, 14, 7] # Days

params3, cov3 = curve_fit(
    sigmoid3,
    x_values_fit,    
    mean,            
    p0=p0,
    sigma=sd,        
    absolute_sigma=True
)

param_means3 = np.array(params3)   # [t0_hat, TSmax_hat, TSmin_hat]
param_cov3   = cov3                # shape (3,3)

daily_file  = os.path.join(script_dir, f'daily_' + name + f'_MC_samp_k={k*30.5:.2f}_N={N}_p={p}_n_seed={n_seed}_t_import={import_date.day}-{import_date.month}_ub={ub}_TIs=sigmoid.csv')
Reff_file   = os.path.join(script_dir, f'Reff_' + name + f'_MC_samp_k={k*30.5:.2f}_N={N}_p={p}_n_seed={n_seed}_t_import={import_date.day}-{import_date.month}_ub={ub}_TIs=sigmoid.csv')
params_file = os.path.join(
    script_dir,
    f'params_{name}_MC_samp_k={k*30.5:.2f}_N={N}_p={p}_n_seed={n_seed}_'
    f't_import={import_date.day}-{import_date.month}_ub={ub}_TIs=sigmoid.csv'
)
# --- New file paths for generation, serial, and dispersion (k) outputs ---
Gen_file = os.path.join(
    script_dir,
    f'Gen_{name}_MC_samp_k={k*30.5:.2f}_N={N}_p={p}_n_seed={n_seed}_'
    f't_import={import_date.day}-{import_date.month}_ub={ub}_TIs=sigmoid.csv'
)
Ser_file = os.path.join(
    script_dir,
    f'Ser_{name}_MC_samp_k={k*30.5:.2f}_N={N}_p={p}_n_seed={n_seed}_'
    f't_import={import_date.day}-{import_date.month}_ub={ub}_TIs=sigmoid.csv'
)
k_file = os.path.join(
    script_dir,
    f'k_{name}_MC_samp_k={k*30.5:.2f}_N={N}_p={p}_n_seed={n_seed}_'
    f't_import={import_date.day}-{import_date.month}_ub={ub}_TIs=sigmoid.csv'
)

if not os.path.exists(params_file):
    with open(params_file, 'w') as f:
        f.write(f'run,k,t0_s ({param_means3[0]:.2f}),TSmax_s ({param_means3[1]:.2f}),TSmin_s ({param_means3[2]:.2f})\n')
###############################################################
# Set up run #
###############################################################

# Simulation parameters
Tmax        = 120   # max duration
num_runs    = 47    # outer repetitions
num_repeats = 20    # inner repetitions per parameter set

for run in range(num_runs):
    # Containers for mean outputs
    daily_arr = torch.zeros(Tmax)
    Reff_arr  = torch.zeros(Tmax)

    # Inner arrays per run (add gen, ser, and k)
    daily_arr_in    = torch.zeros((num_repeats, Tmax))
    Reff_arr_in     = torch.zeros((num_repeats, Tmax))
    gen_arr_in      = torch.zeros((num_repeats, Tmax))
    ser_arr_in      = torch.zeros((num_repeats, Tmax))
    k_arr_in        = torch.zeros((num_repeats, Tmax))
    cases_arr_in    = torch.zeros((num_repeats, Tmax))

    # —––– NEW: pre‐allocate 2‐D arrays to hold all repeats’ infection_times & secondary_infs ––––
    #   Shape = (num_repeats, N)
    infection_times_all = torch.full((num_repeats, N), -1.0)  # initialize to -1 (never infected)
    secondary_infs_all  = torch.zeros((num_repeats, N))      # initialize to 0

    # --- sample epidemic parameters ---
    t0_s, TSmax_s, TSmin_s = \
        np.random.multivariate_normal(param_means3, param_cov3)
    TSvals = sigmoid3(np.arange(Tmax), t0_s, TSmax_s,TSmin_s)
    TIvals      = 2*(TSvals - TE)
    TI0         = TIvals[0]
    TI_changes  = torch.tensor([np.arange(0,len(TIvals)),TIvals[0:]])
    TI_changes[TI_changes < 0] = 0  # No negative TI allowed

    for rep in range(num_repeats):
        # run the simulation
        simulation = EpidemicSimulation(N, M, r_old_mean, pexp, old_dist, alpha, ub, lb)
        simulation.load_state(indices, seeking, r_new, r_old, tot_log_peace, t=0)
        simulation.initialize_war_time(TE, TI0, n, p, TI_changes, pre_vac)
        simulation.iterate_war_time(t_inf=0, n_seed=n_seed, run=run, Tmax=Tmax)

        # collect outputs
        time_points, Reff_values = simulation.moving_window_reff(window_size=10)
        _, daily_values = simulation.calculate_new_cases()

        # collect generation & serial intervals, and k in the same window
        _, gen_vals, ser_vals = simulation.moving_window_gen_ser(window_size=10)
        _, k_vals = simulation.moving_window_k(window_size=10)
        _, cases_vals = simulation.moving_window_cases(window_size=10)

        # dur_Reff  = len(Reff_values)
        # dur_daily = len(daily_values)
        # dur_gen   = len(gen_vals)
        # dur_ser   = len(ser_vals)
        # dur_k     = len(k_vals)

        daily_arr_in[rep,:] = torch.tensor(daily_values, dtype=torch.float)
        Reff_arr_in[rep,:]  = torch.tensor(Reff_values, dtype=torch.float)
        gen_arr_in[rep,:]   = torch.tensor(gen_vals, dtype=torch.float)
        ser_arr_in[rep,:]   = torch.tensor(ser_vals, dtype=torch.float)
        k_arr_in[rep,:]     = torch.tensor(k_vals, dtype=torch.float)
        cases_arr_in[rep,:] = torch.tensor(cases_vals, dtype=torch.float)

        # —––– NEW: extract and store this repeat’s infection_times & secondary_infs ––––
        #    We assume sim.infection_times is a 1‐D tensor length N,
        #    and sim.secondary_infs is a 1‐D tensor length N.
        infection_times_all[rep, :]  = simulation.infection_times.float()
        secondary_infs_all[rep, :]   = simulation.secondary_infs.float()

    # —––– Convert torch→numpy for computing the “pooled k(t)” ––––
    # (We only need to flatten for k; no change to how we do daily, Reff, etc.)
    inf_times_all_np = infection_times_all.cpu().numpy()
    sec_infs_all_np  = secondary_infs_all.cpu().numpy()

    # —––– Compute pooled‐over‐repeats moving‐window k(t) ––––
    #     This returns time_points (0..Tmax-1) and a 1‐D array k_values of length Tmax
    window_size = 10
    _, pooled_k_values = moving_window_k_all(
        inf_times_all_np,
        sec_infs_all_np,
        Tmax=Tmax,
        window_size=window_size
    )

    # Convert all torch tensors to NumPy once:
    daily_np = daily_arr_in.numpy()    # daily new‐cases per repeat per t
    Reff_np  = Reff_arr_in.numpy()
    gen_np   = gen_arr_in.numpy()
    ser_np   = ser_arr_in.numpy()
    cases_np = cases_arr_in.numpy()    # number of new cases per repeat per t

    # 1) daily incidence: still take a simple (unweighted) mean across repeats,
    #    because weighting daily counts by number of cases in the window would
    #    effectively re‐weight them by themselves
    mean_daily = np.nanmean(daily_np, axis=0)

    # 2) for everything else, use the case counts as weights:
    mean_Reff = compute_weighted_mean(Reff_np,  cases_np, Tmax)
    mean_gen  = compute_weighted_mean(gen_np,   cases_np, Tmax)
    mean_ser  = compute_weighted_mean(ser_np,   cases_np, Tmax)

    # store mean back into outer arrays
    daily_arr[:] = torch.tensor(mean_daily)
    Reff_arr[:]  = torch.tensor(mean_Reff)

    # save as CSV rows (append new run)
    save_or_append_data(daily_file, daily_arr.numpy(), fmt='%f')
    save_or_append_data(Reff_file, Reff_arr.numpy(), fmt='%f')
    save_or_append_data(Gen_file,   mean_gen,             fmt='%f')
    save_or_append_data(Ser_file,   mean_ser,             fmt='%f')
    save_or_append_data(k_file,     pooled_k_values,               fmt='%f')

    with open(params_file, 'a') as f:
        f.write(f"{run},{k:.6f},{t0_s:.6f},{TSmax_s:.6f},{TSmin_s:.6f}\n")