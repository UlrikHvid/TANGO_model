import sys
import os
# Directory setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '../..'))
data_dir = os.path.join(parent_dir, 'Saved_networks/')
sys.path.append(parent_dir)

import matplotlib.pyplot as plt
import importlib
import scipy as sp
import torch
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns

import tango_model as tango_model
from tango_model import EpidemicSimulation

################################################################################################
# Helper function for appending results
################################################################################################
def append_single_run_data(df: pd.DataFrame, filename: str) -> None:
    """
    Appends a single-run DataFrame row to a CSV file. Creates the file with headers if it doesn't exist.
    If file exists, appends without header.
    """
    if not os.path.exists(filename):
        df.to_csv(filename, index=False, mode='w')  # Create new file, include header
    else:
        df.to_csv(filename, index=False, mode='a', header=False)  # Append to existing, no header

def append_run_results(
    results_run_df: pd.DataFrame,
    inf_prob_run_df: pd.DataFrame,
    inf_prob_partners_run_df: pd.DataFrame,
    results_filename: str,
    inf_prob_filename: str,
    inf_prob_partners_filename: str
) -> None:
    """
    Appends one run's DataFrames to their respective CSV files.
    """
    append_single_run_data(results_run_df, results_filename)
    append_single_run_data(inf_prob_run_df, inf_prob_filename)
    append_single_run_data(inf_prob_partners_run_df, inf_prob_partners_filename)

################################################################################################
# Main parameters
################################################################################################
N           = 170000
M           = 28  
r_old_mean  = 0.2
pexp        = 1.55
lb          = 0.15
window      = 365
old_dist    = "gamma"
alpha       = 2
T_track     = None
n_couples   = 0

################################################################################################
# Epidemic parameters
################################################################################################
TE          = 5   # 6 days (divided by 7 to convert to weeks if needed)
TI          = 18 
n           = 2
pre_vac     = 0.13
n_seed      = 10    # Number of initial imported cases
t_inf       = 0     # Time to introduce infection

################################################################################################
# Variable parameters
################################################################################################
ub          = 300
p           = 0.48
quantiles   = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float32)
num_runs    = 30

# Import network data
if pexp == 1.55:
    network_data    = pd.read_csv(os.path.join(data_dir, f'Network_N={N:.1e}_ub={ub}.csv'))
    member_data     = pd.read_csv(os.path.join(data_dir, f'Member_data_N={N:.1e}_ub={ub}.csv'))
else:
    network_data    = pd.read_csv(os.path.join(data_dir, f'Network_N={N:.1e}_ub={ub}_pexp={pexp}.csv'))
    member_data     = pd.read_csv(os.path.join(data_dir, f'Member_data_N={N:.1e}_ub={ub}_pexp={pexp}.csv'))

# Extract member data
seeking = torch.tensor(member_data["seeking"].values, dtype=torch.uint8)
r_new   = torch.tensor(member_data["r_new"].values, dtype=torch.float32)
r_old   = torch.tensor(member_data["r_old"].values, dtype=torch.float32)

# Load network indices and strengths
rows        = torch.tensor(network_data["rows"].values, dtype=torch.int32)
cols        = torch.tensor(network_data["cols"].values, dtype=torch.int32)
strengths   = torch.tensor(network_data["strengths"].values, dtype=torch.int32)

# Reconstruct indices by repeating edges according to strengths
indices_rows    = rows.repeat_interleave(strengths)
indices_cols    = cols.repeat_interleave(strengths)
indices         = torch.stack([indices_rows, indices_cols])

# Load tot_log_peace data
tot_log_peace_df    = pd.read_csv(os.path.join(data_dir, f'tot_log_N={N:.1e}_ub={ub}.csv'))
tot_log_rows        = torch.tensor(tot_log_peace_df["rows"].values, dtype=torch.int32)
tot_log_cols        = torch.tensor(tot_log_peace_df["cols"].values, dtype=torch.int32)
tot_log_t           = torch.tensor(tot_log_peace_df["t"].values, dtype=torch.int32)
repeats             = torch.tensor(tot_log_peace_df["repeats"].values, dtype=torch.int32)

# Reconstruct tot_log by expanding repeats
expanded_rows   = tot_log_rows.repeat_interleave(repeats)
expanded_cols   = tot_log_cols.repeat_interleave(repeats)
expanded_t      = tot_log_t.repeat_interleave(repeats)
tot_log_peace   = torch.stack([expanded_rows, expanded_cols, expanded_t])

# Adjust time coordinates to negative times
max_t = tot_log_peace[2].max()
tot_log_peace[2] -= (max_t + 1)

# Create simulation object and load the state
simulation = EpidemicSimulation(N, M, r_old_mean, pexp, old_dist, alpha, ub, lb)
if M == 0:
    indices = torch.tensor([[], []])  # No network

simulation.load_state(
    indices=indices,
    seeking=seeking,
    r_new=r_new,
    r_old=r_old,
    tot_log=tot_log_peace,
    t=0  # Start at time 0
)

# N-length tensor with number of partners in last year before epidemic
_, tot_partners = simulation.analyze_encounters()

n_bins = 100

# Prepare output filenames
addendum = "_no_network" if M == 0 else ""
results_filename           = f'results_N={N}_ub={ub}_p={p}{addendum}.csv'
inf_prob_filename          = f'inf_prob_vals_N={N}_ub={ub}_p={p}{addendum}.csv'
inf_prob_partners_filename = f'inf_prob_partners_vals_N={N}_ub={ub}_p={p}{addendum}.csv'

################################################################################################
# Simulation loop
################################################################################################
for run in range(num_runs):
    # Update progress
    sys.stdout.write(f"\rProcessing run {run+1}/{num_runs} for ub={ub}, p={p}")
    sys.stdout.flush()
    
    # Reset the simulation state to ensure independence between runs
    simulation.load_state(
        indices=indices,
        seeking=seeking,
        r_new=r_new,
        r_old=r_old,
        tot_log=tot_log_peace,
        t=0  # Reset time to 0
    )
    
    # Initialize war-time simulation
    simulation.initialize_war_time(
        TE=TE,
        TI=TI,
        pre_vac=pre_vac,
        n=n,
        p=p
    )
    # Run the war-time simulation
    simulation.iterate_war_time(t_inf=t_inf, n_seed=n_seed, run=run)
    
    # Estimate R0 and k in the specified period
    R0_estimate = simulation.estimate_r0(t_start=t_inf+14, t_end=t_inf+50)
    k_estimate  = simulation.estimate_k(tf=t_inf+50)
    
    # Compute prior partners within 21 and 90 days before infection
    priorpartners21 = simulation.partners_prior_to_infectious(T=21)
    priorpartners90 = simulation.partners_prior_to_infectious(T=90)

    final_size = simulation.R[-1]/(N*0.58)
    
    # Probability by r_new and by # partners
    bin_centers, probabilities = simulation.infection_probability_by_r_new(n_bins=n_bins)
    bin_centers_partners, probabilities_partners = simulation.infection_probability_by_partners(
        n_bins=n_bins, tot_partners=tot_partners
    )
    
    # Exclude non-infected and initial infections
    infected_mask    = (simulation.infection_times != -1) & (simulation.infection_times != t_inf)
    priorpartners21  = priorpartners21[infected_mask]
    priorpartners90  = priorpartners90[infected_mask]
    
    # Compute quantiles
    if len(priorpartners21) > 0:
        lq21, median21, uq21 = torch.quantile(priorpartners21.float(), quantiles)
    else:
        lq21, median21, uq21 = 0.0, 0.0, 0.0
    if len(priorpartners90) > 0:
        lq90, median90, uq90 = torch.quantile(priorpartners90.float(), quantiles)
    else:
        lq90, median90, uq90 = 0.0, 0.0, 0.0
    
    #gen_time, _ = simulation.compute_generation_and_serial_intervals()
    #mean_gen_time = gen_time[gen_time > -10000].float().mean().item()  # scalar
    
    # --------------------------------------------------------------------------------------------
    # Build DataFrames (single row each) for the results of this run
    # --------------------------------------------------------------------------------------------

    # 1) Main results (R0, k, quantiles, final size, generation time)
    results_dict = {
        'R0_vals':          [R0_estimate],
        'k_vals':           [k_estimate],
        'lq21_vals':        [lq21.item() if isinstance(lq21, torch.Tensor) else lq21],
        'median21_vals':    [median21.item() if isinstance(median21, torch.Tensor) else median21],
        'uq21_vals':        [uq21.item() if isinstance(uq21, torch.Tensor) else uq21],
        'lq90_vals':        [lq90.item() if isinstance(lq90, torch.Tensor) else lq90],
        'median90_vals':    [median90.item() if isinstance(median90, torch.Tensor) else median90],
        'uq90_vals':        [uq90.item() if isinstance(uq90, torch.Tensor) else uq90],
        'final_size':       [final_size],
        #'gen_time_vals':    [mean_gen_time]
    }
    results_run_df = pd.DataFrame(results_dict)

    # 2) Infection probability by r_new
    #    We label the columns generically or using bin centers if desired
    inf_prob_run_df = pd.DataFrame([probabilities], 
                                   columns=[f'bin_{i}_{bin_centers[i]:.4f}' for i in range(n_bins)])

    # 3) Infection probability by # of partners
    inf_prob_partners_run_df = pd.DataFrame([probabilities_partners], 
                                            columns=[f'bin_{i}_{bin_centers_partners[i]:.4f}' for i in range(n_bins)])

    # --------------------------------------------------------------------------------------------
    # Use our append function
    # --------------------------------------------------------------------------------------------
    append_run_results(
        results_run_df=results_run_df,
        inf_prob_run_df=inf_prob_run_df,
        inf_prob_partners_run_df=inf_prob_partners_run_df,
        results_filename=results_filename,
        inf_prob_filename=inf_prob_filename,
        inf_prob_partners_filename=inf_prob_partners_filename
    )

# Final message
print("\nAll runs completed and data appended to CSV files.")
