import numpy as np
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import sys
import torch
from torch import tensor, zeros
import concurrent.futures
from datetime import timedelta

class EpidemicSimulation:
    def __init__(self, N, M, r_old_mean, pexp, old_dist="exponential", alpha=0, ub=None, lb=1):
        self.N = N
        self.t = 0  # Simulation time

        # Initialize common variables
        self.seeking = torch.zeros(self.N, dtype=torch.uint8)
        self.indices = torch.empty((2, 0), dtype=torch.int32) # A list of all current connections
        self.encounters_today = torch.empty((3, 0), dtype=torch.int32)
        self.tot_log = torch.empty((3, 0), dtype=torch.int32)
        self.log = False  # Whether to log encounters

        # Initialize variables for steady partnerships
        self.track_steady = False  # Set this to True when needed

        # Initialize data structures for logging steady partnerships
        self.logged_couples = []  # List to store logged steady partnerships
        self.logged_couple_strengths = {}  # Dictionary to store strengths over time
        self.logged_couple_steady_status = {}  # Dictionary to store steady status over time
        self.logged_couple_active_status = {}  # Dictionary to track if a partnership is active

        # Initialize parameters common to both simulations
        self.M = M  # Mean partnership duration in days
        self.r_old_mean = r_old_mean  # Mean seeking rate for old partners per day
        self.pexp = pexp
        self.old_dist = old_dist
        self.alpha = alpha
        self.ub = ub  # Upper bound in per year units
        self.lb = lb  # Lower bound in per year units
    
    def load_state(self, indices, seeking, r_new, r_old, tot_log, t=0):
        """
        Load the simulation state from saved data.

        Parameters:
            indices (torch.Tensor): Edge indices of the network.
            seeking (torch.Tensor): Seeking status of individuals.
            r_new (torch.Tensor): New partner seeking rates.
            r_old (torch.Tensor): Old partner seeking rates.
            tot_log (torch.Tensor): Total encounter log.
            t (int): Current simulation time.
        """
        self.indices = indices
        self.seeking = seeking
        self.r_new = r_new
        self.r_old = r_old
        self.tot_log = tot_log
        self.t = t

        if len(r_new) != self.N:
            raise(ValueError("N does not fit imported data"))

    @staticmethod
    def Poisson(lams):
        # Using the Knuth algorithm in a vectorized form
        k = torch.zeros_like(lams, dtype=torch.int16)
        p = torch.ones_like(lams, dtype=torch.float32)
        L = torch.exp(-lams)
        minus_matrix = (lams != 0).short()  # Ensure that lambda = 0 gives the values 0, not -1.
        while (p > L).sum() > 0:  # While there are still p-values greater than L
            k[p > L] += 1
            p *= torch.rand(*lams.shape)
        return (k - minus_matrix).to(torch.int32)

    def update_seeking(self):
        random_numbers = torch.rand(self.N)
        self.seeking = torch.where((random_numbers < self.r_old), 1, self.seeking) #Where number lower, make seeking, otherwise leave at previous value

    def make_old_connects(self):
        seeking_indices = torch.where(self.seeking)[0] # Everyone who is seeking
        seeking_set = set(seeking_indices.tolist()) # Trick for speed
        mask_row = torch.tensor([(val in seeking_set) for val in self.indices[0].tolist()], dtype=torch.bool) 
        mask_col = torch.tensor([(val in seeking_set) for val in self.indices[1].tolist()], dtype=torch.bool)
        combined_mask = mask_row & mask_col # Both must be seeking
        potential_encs = self.indices[:, combined_mask] # They must be connected

        while potential_encs.shape[1] > 0:
            selected = torch.randint(0, potential_encs.shape[1], (1,))

            # Strengthen the selected connection
            row_index, col_index = potential_encs[0, selected], potential_encs[1, selected]
            tensor_to_append = torch.tensor([row_index, col_index, self.t]).unsqueeze(1)
            self.encounters_today = torch.cat((self.encounters_today, tensor_to_append), dim=1)

            # Satisfy members
            self.seeking[row_index], self.seeking[col_index] = 0, 0

            # Remove connections
            prune_mask = (potential_encs != row_index) & (potential_encs != col_index)
            potential_encs = potential_encs[:, prune_mask.all(dim=0)]

    def make_new_connects(self, new_seek_counts):
        while (new_seek_counts > 0).sum() > 1:
            seeking_indices = torch.nonzero(new_seek_counts).squeeze(1)
            all_seeking = seeking_indices[torch.randperm(seeking_indices.size(0))]
            half_length = all_seeking.size(0) // 2
            while (all_seeking[half_length - 1] == all_seeking[half_length]):
                if all_seeking[-1] == all_seeking[half_length]:
                    half_length -= 1
                else:
                    half_length += 1
            row_indices = all_seeking[:half_length]
            col_indices = all_seeking[half_length:]
            row_indices = row_indices.repeat_interleave(new_seek_counts[row_indices])
            col_indices = col_indices.repeat_interleave(new_seek_counts[col_indices])
            row_indices = row_indices[torch.randperm(row_indices.size(0))]
            col_indices = col_indices[torch.randperm(col_indices.size(0))]

            min_length = min(row_indices.size(0), col_indices.size(0))
            row_indices = row_indices[:min_length]
            col_indices = col_indices[:min_length]

            tensor_to_append = torch.stack([row_indices, col_indices, torch.full((min_length,), self.t, dtype=torch.long)], dim=0)
            self.encounters_today = torch.cat((self.encounters_today, tensor_to_append), dim=1)

            new_seek_counts -= torch.bincount(row_indices, minlength=new_seek_counts.size(0))
            new_seek_counts -= torch.bincount(col_indices, minlength=new_seek_counts.size(0))

    def connection_decay(self):
        if self.M:
            random_numbers = torch.rand(self.indices.shape[1])
            mask = (random_numbers > 1 / self.M)
            self.indices = self.indices[:, mask]
        else:  # Clear memory
            self.indices = torch.empty((2, 0), dtype=torch.int32)

    def assign_counts(self):
        new_seek_counts = self.Poisson(self.r_new)
        return new_seek_counts

    def housekeeping_peace(self):
        self.indices = torch.cat((self.indices, self.encounters_today[:-1]), dim=1)
        if self.log:
            self.tot_log = torch.cat((self.tot_log, self.encounters_today), dim=1)
        self.encounters_today = torch.empty((3, 0), dtype=torch.int32)
        self.t += 1
    
        # Only update after T_track
        if self.T_track is not None and self.t >= self.T_track:
            # Always update logged couples
            self.update_logged_couples()
    
            # Call find_new_steady_partnerships only if tracking is enabled
            if self.track_steady:
                self.find_new_steady_partnerships()


    def compute_current_steady_partnerships(self):
        # Get edges from indices
        indices = self.indices
    
        # Get sorted edges so that i <= j
        edges_sorted = torch.sort(indices, dim=0)[0]  # 2 x E
    
        # Now, get unique edges and counts
        unique_edges, inverse_indices, counts = torch.unique(
            edges_sorted, dim=1, return_inverse=True, return_counts=True
        )
    
        # unique_edges: 2 x E_unique
        # counts: E_unique
    
        # Now, create node_indices, neighbor_indices, strengths_cat
        edges = unique_edges
        strengths = counts.float()
    
        node_indices = torch.cat([edges[0], edges[1]])
        neighbor_indices = torch.cat([edges[1], edges[0]])
        strengths_cat = torch.cat([strengths, strengths])
    
        # Compute total strengths
        total_strengths = torch.zeros(self.N)
        total_strengths.scatter_add_(0, node_indices.long(), strengths_cat)
    
        # Find maximum strengths and corresponding neighbors
        sorted_indices = torch.argsort(node_indices)
        node_indices_sorted = node_indices[sorted_indices].long()
        neighbor_indices_sorted = neighbor_indices[sorted_indices].long()
        strengths_sorted = strengths_cat[sorted_indices]
    
        # Get unique nodes and their counts
        unique_nodes, counts_nodes = torch.unique_consecutive(
            node_indices_sorted, return_counts=True
        )
    
        max_strengths = torch.zeros(self.N)
        favorite = torch.full((self.N,), -1, dtype=torch.long)
    
        # Manually track start indices for each node
        start_idx = 0
        for idx, node in enumerate(unique_nodes):
            # The end index is determined by the count of the current node
            end_idx = start_idx + counts_nodes[idx]
    
            # Get the values for this node's connections
            s_values = strengths_sorted[start_idx:end_idx]
            n_values = neighbor_indices_sorted[start_idx:end_idx]
    
            # Find the strongest connection
            max_idx = torch.argmax(s_values)
    
            # Assign max strength and favorite
            max_strengths[node] = s_values[max_idx]
            favorite[node] = n_values[max_idx]
    
            # Update start index for the next node
            start_idx = end_idx
    
        # Compute rest_strengths
        rest_strengths = total_strengths - max_strengths
    
        # Check if max_strengths > rest_strengths and strength >= 2
        mask_favorite = (max_strengths > rest_strengths) & (max_strengths >= 2)
    
        # Update favorite to -1 where condition is not met
        favorite[~mask_favorite] = -1
    
        # Find steady partnerships
        mask_valid = favorite >= 0
        favorite_of_favorite = torch.full((self.N,), -1, dtype=torch.long)
        indices_range = torch.arange(self.N, dtype=torch.long)
        valid_indices = indices_range[mask_valid]
        favorite_of_favorite[valid_indices] = favorite[favorite[valid_indices]]
    
        steady_mask = (favorite_of_favorite == indices_range) & (favorite >= 0) & (indices_range < favorite)
    
        steady_pairs = torch.stack([indices_range[steady_mask], favorite[steady_mask]], dim=1)
    
        # Create edge_strengths dictionary
        edge_strengths = {}
        for i in range(edges.shape[1]):
            edge_tuple = (edges[0, i].item(), edges[1, i].item())
            edge_strengths[edge_tuple] = strengths[i].item()
    
        return steady_pairs, edge_strengths

    def find_new_steady_partnerships(self):
        # Compute current steady partnerships
        steady_pairs, _ = self.compute_current_steady_partnerships()

        # Build set of current steady partnerships
        current_steady_pairs = set([tuple(pair.tolist()) for pair in steady_pairs])

        # Check for new steady partnerships and add to log if tracking is enabled
        for pair in current_steady_pairs:
            if pair not in self.logged_couples:
                if len(self.logged_couples) < self.n_couples:
                    self.logged_couples.append(pair)
                    self.logged_couple_strengths[pair] = []
                    self.logged_couple_steady_status[pair] = []
                    self.logged_couple_active_status[pair] = True  # Partnership is active
                    if len(self.logged_couples) == self.n_couples:
                        self.track_steady = False
                else:
                    self.track_steady = False
                    print(f"Done seeking couples, t = {self.t}")
                    break

    def get_steady_partners_mask(self):
        """
        Generates a boolean mask indicating network members who are currently
        in a steady partnership.
    
        Returns:
            steady_mask (torch.Tensor): A boolean tensor of shape (N,), where N is
                                        the number of nodes in the network. True indicates
                                        the node is in a steady partnership.
        """
        # Compute current steady partnerships
        steady_pairs, _ = self.compute_current_steady_partnerships()
    
        # Initialize the mask with False values
        steady_mask = torch.zeros(self.N, dtype=torch.bool)
    
        # Mark nodes that are in steady partnerships
        if steady_pairs.numel() > 0:
            nodes_in_steady = torch.unique(steady_pairs)
            steady_mask[nodes_in_steady] = True
    
        return steady_mask

    def update_logged_couples(self):
        # Compute current edge strengths and steady partnerships
        _, edge_strengths = self.compute_current_steady_partnerships()      
        # Build set of current steady partnerships
        steady_pairs, _ = self.compute_current_steady_partnerships()
        current_steady_pairs = set([tuple(pair.tolist()) for pair in steady_pairs])     
        # Update logged couples
        for couple in self.logged_couples:
            # Skip updates for inactive (dissolved) partnerships
            if not self.logged_couple_active_status[couple]:
                continue  # Skip to the next couple     
            node_i, node_j = couple
            edge = (min(node_i, node_j), max(node_i, node_j))
            strength = edge_strengths.get(edge, 0)
            is_steady_partner = couple in current_steady_pairs
            self.logged_couple_strengths[couple].append(strength)
            self.logged_couple_steady_status[couple].append(is_steady_partner)
            if strength == 0:
                self.logged_couple_active_status[couple] = False

    def get_partnership_durations(self, include_non_steady=False):
        durations = []
        for couple in self.logged_couples:
            strengths_list = self.logged_couple_strengths[couple]
            steady_status_list = self.logged_couple_steady_status[couple]

            if include_non_steady:
                # Total duration of the connection (active periods)
                duration = len(strengths_list)
            else:
                # Total duration as a steady partnership
                duration = sum(steady_status_list)

            durations.append(duration)
        return durations
    
    def initialize_peace_time(self, T, Twindow, progress=True, n_couples=0, all_conn=False, margin=1, T_track=None):
        # All time parameters are now in days
        self.T = T
        self.Twindow = Twindow
        self.progress = progress
        self.n_couples = n_couples
        self.all_conn = all_conn
        self.margin = margin
        self.T_track = T_track  # Time to start tracking steady partnerships

        self.indices = torch.empty((2, 0), dtype=torch.int32) # A list of all current connections
        self.encounters_today = torch.empty((3, 0), dtype=torch.int32)
        self.seeking = torch.zeros(self.N, dtype=torch.uint8)
        self.conn_ages = torch.zeros(self.N, self.N, dtype=torch.int16) if all_conn else None
        self.ss_index = 0
        self.conn_durations = torch.tensor([])
        self.log = False
        # self.tot_log = torch.empty((3, 0), dtype=torch.int32)
        self.mean_arr = torch.zeros(T)

        if self.M:
            self.steady_state = False
        else:
            self.steady_state = True

        self.looplist = tqdm(range(T)) if progress else range(T)
        self.tracking = 0
        # Initialize rates
        self.r_new, self.r_old = self.initialize_rates()

    def initialize_rates(self):
        if self.old_dist == "exponential":
            self.r_old = torch.zeros(self.N)
            if self.r_old_mean > 0:
                self.r_old.exponential_(1 / self.r_old_mean)
        elif self.old_dist == "gamma":
            dist = torch.distributions.gamma.Gamma(self.alpha, self.alpha / self.r_old_mean)
            self.r_old = dist.rsample(torch.Size([self.N]))
        elif self.old_dist == "constant":
            self.r_old = torch.ones(self.N) * self.r_old_mean
        else:
            raise ValueError("Unknown distribution")

        if isinstance(self.pexp, float):
            self.r_new = self.Kdist()
            self.r_new /= 365  # Convert from per year to per day
        elif isinstance(self.pexp, torch.Tensor):
            if len(self.pexp) != self.N:
                raise ValueError("Length of 'pexp' tensor does not match 'N'")
            else:
                self.r_new = self.pexp / 365  # From per year to per day
        else:
            raise ValueError("'pexp' has invalid type")

        return self.r_new, self.r_old

    def Kdist(self, round=False):
        Karr = torch.zeros(self.N)
        if self.ub is None:
            ub = self.N
        else:
            ub = self.ub
        expo = -self.pexp + 1
        for i in range(self.N):
            ri = np.random.random()
            Karr[i] = (ri * (ub ** expo - self.lb ** expo) + self.lb ** expo) ** (1 / expo)
        if round:
            return Karr.sort(descending=True).values.round().astype(int)
        else:
            return Karr.sort(descending=True).values

    def iterate_peace_time(self):
        self.update_seeking()
        self.make_old_connects()

        new_seek_counts = self.assign_counts()
        self.make_new_connects(new_seek_counts)

        self.connection_decay()
        self.housekeeping_peace()

    def run_peace_time_simulation(self):
        for t in self.looplist:
            if self.T_track is not None and t == self.T_track:
                self.track_steady = True  # Start tracking steady partnerships
            if t >= (self.T - self.Twindow):
                self.log = True
            self.iterate_peace_time()

    def initialize_war_time(self, TE, TI, n, p, TI_changes=None, pre_vac=0.12, Tvac=None, N_vac=0):
        # Time parameters are in days
        self.encounters_today = torch.empty((3, 0), dtype=torch.int32)
        self.states = self.smallpox_vaccination(pre_vac)
        self.TE = TE
        self.TI = TI
        self.Tvac = Tvac
        self.N_vac = N_vac
        self.infection_times = torch.ones(self.N, dtype=torch.int16) * (-1)
        self.infectious_times = torch.ones(self.N, dtype=torch.int16) * (-1)  # Initialize infectious_times
        self.primary_cases = torch.ones(self.N, dtype=torch.long) * (-1)     # Initialize primary_cases
        self.secondary_infs = torch.zeros(self.N, dtype=torch.int16)
        self.log = True
        self.S = torch.empty(0, dtype=torch.uint8)
        self.E = torch.empty(0, dtype=torch.uint8)
        self.I = torch.empty(0, dtype=torch.uint8)
        self.R = torch.empty(0, dtype=torch.uint8)
        self.t = 0  # Reset simulation time
        self.n = n
        self.p = p  # Transmission probability
        if TI_changes is not None:
            # Ensure TI_changes is a tensor of shape (2, n)
            if TI_changes.shape[0] != 2:
                raise ValueError("TI_changes should be a tensor of shape (2, n)")
            # Convert to lists
            Tmitigs = TI_changes[0, :].tolist()
            TIs = TI_changes[1, :].tolist()
            # Sort the Tmitigs and TIs according to Tmitigs
            Tmitigs_TIs = sorted(zip(Tmitigs, TIs))
            self.TI_changes_list = Tmitigs_TIs
            self.current_TI_index = 0
        else:
            self.TI_changes_list = []
            self.current_TI_index = 0

    def smallpox_vaccination(self, pre_vac):
        states = torch.zeros(self.N, dtype=torch.int8)
        random_numbers = torch.rand(self.N)
        states[random_numbers < pre_vac] = -1
        return states

    def iterate_war_time(self, t_inf=0, n_seed=1, vac_threshold=1, strategy="from_above", run=0,Tmax = np.inf):
        self.Tmax = Tmax
        while ((len(self.E) == 0) or (self.E[-1] + self.I[-1] != 0) or (self.t <= t_inf)) & (self.t < self.Tmax):
            if self.current_TI_index < len(self.TI_changes_list): # Update the time-changing infectious period
                Tmitig_next, TI_next = self.TI_changes_list[self.current_TI_index]
                if self.t == Tmitig_next:
                    self.TI = TI_next
                    self.current_TI_index += 1
            if self.t == t_inf:
                self.states, self.infection_times = self.infect_random(n_seed)
            if (self.Tvac is not None) and (self.t in np.array(self.Tvac)) and (self.N_vac > 0):
                self.states = self.mpox_vaccination(self.N_vac, strategy, vac_threshold)
            self.update_values()
    
            self.update_seeking()
            self.make_old_connects()
            
            new_seek_counts = self.assign_counts()
            self.make_new_connects(new_seek_counts)

            if self.TI > 0: # Taking behavioral change into account, negative values might be input
                self.states, self.infection_times, self.secondary_infs = self.disease_transmission()
    
            self.states = self.disease_progression()
    
            self.connection_decay()
    
            self.housekeeping_war(run)

    def disease_transmission(self):
        states_new = self.states.clone()
        # Variable length lists
        susceptible_names = torch.where(states_new == 0)[0]
        susceptible_set = set(susceptible_names.tolist())
        infectious_names = torch.where((states_new > self.n) & (states_new <= 2 * self.n))[0]
        infectious_set = set(infectious_names.tolist())

        # Length N masks
        susceptible_mask = torch.tensor([[(element.item() in susceptible_set) for element in row] for row in self.encounters_today[:-1]], dtype=torch.bool)
        infectious_mask = torch.tensor([[(element.item() in infectious_set) for element in row] for row in self.encounters_today[:-1]], dtype=torch.bool)

        lefttoright_mask = infectious_mask[0] & susceptible_mask[1] # Where left may infect right
        righttoleft_mask = infectious_mask[1] & susceptible_mask[0] # Where right may infect left
        infectors = torch.cat((self.encounters_today[0][lefttoright_mask], self.encounters_today[1][righttoleft_mask]), dim=0)
        infectees = torch.cat((self.encounters_today[1][lefttoright_mask], self.encounters_today[0][righttoleft_mask]), dim=0)
        random_numbers = torch.rand(len(infectors))
        successful_mask = random_numbers < self.p
        infectors, infectees = infectors[successful_mask], infectees[successful_mask] 
        self.infection_times[infectees] = self.t
        states_new[infectees] = 1  # Exposed state
        ones = torch.ones_like(infectors, dtype=torch.int16)
        self.secondary_infs.scatter_add_(0, infectors.long(), ones) # Zeroth dimension, mask, add one

        # Update primary_cases for new infections
        self.primary_cases[infectees.long()] = infectors.long()

        return states_new, self.infection_times, self.secondary_infs


    def disease_progression(self):
        previous_states = self.states.clone()  # Keep track of previous states
        random_numbers = torch.rand(self.N)
        Eprogression = (self.states > 0) & (self.states <= self.n) & (random_numbers < (self.n / self.TE))
        if self.TI > 0: # Taking behavioral change into account, negative values might be input
            Iprogression = (self.states > self.n) & (self.states <= 2 * self.n) & (random_numbers < (self.n / self.TI))
            self.states[Iprogression] += 1
        else: # Immediate progression
            Iprogression = (self.states > self.n) & (self.states <= 2 * self.n)
            self.states[Iprogression] = 2*self.n + 1 # Straight to recovered
        self.states[Eprogression] += 1  # DO NOT PLACE BEFORE DEFINITION OF IPROGRESSION!
        #self.states[self.states == 2 * self.n + 1] = 2 * self.n + 1  # Remain in recovered state

        # Identify individuals who just became infectious
        became_infectious = (previous_states == self.n) & (self.states == self.n + 1)
        self.infectious_times[became_infectious] = self.t

        return self.states

    def update_values(self):
        S_now = (self.states == 0).sum()
        E_now = sum([(self.states == i).sum() for i in range(1, self.n + 1)])
        I_now = sum([(self.states == i).sum() for i in range(self.n + 1, 2 * self.n + 1)])
        R_now = (self.states == 2 * self.n + 1).sum()
        self.S = torch.cat((self.S, torch.tensor([S_now])), dim=0)
        self.E = torch.cat((self.E, torch.tensor([E_now])), dim=0)
        self.I = torch.cat((self.I, torch.tensor([I_now])), dim=0)
        self.R = torch.cat((self.R, torch.tensor([R_now])), dim=0)

    # def infect_random(self, n_seed):
    #     mask = (self.states == 0)
    #     indices_temp = torch.multinomial(self.r_new[mask], num_samples=n_seed, replacement=False)
    #     infected = torch.nonzero(mask, as_tuple=True)[0][indices_temp]
    #     self.states[infected] = self.n + 1  # Infectious state
    #     self.infection_times[infected]  = self.t 
    #     self.infectious_times[infected] = self.t #Infected and infectious at the same time

    #     # Assign a special value to primary_cases for initial infections
    #     self.primary_cases[infected] = -2  # -2 indicates initial infection source

    #     return self.states, self.infection_times

    def infect_random(self, n_seed): #Different version
        mask      = (self.states == 0)
        sus_idx   = torch.nonzero(mask, as_tuple=True)[0] #Susceptible (non-vaccinated)
        sel       = torch.multinomial(self.r_new[mask], num_samples=n_seed, replacement=False)
        infected  = sus_idx[sel]
    
        # decide E vs I
        p_E     = self.TE / (self.TE + self.TI)
        is_E    = torch.bernoulli(torch.full((n_seed,), p_E, device=infected.device)).bool() 
    
        # allocate with the same dtype as self.states
        states_new = torch.empty(
            (n_seed,),
            dtype=self.states.dtype,
            device=infected.device
        )
    
        # fill E-substates: 1..n
        n_E = is_E.sum().item() 
        if n_E > 0:
            states_new[is_E] = torch.randint(
                low=1,
                high=self.n + 1,
                size=(n_E,),
                dtype=self.states.dtype,
                device=infected.device
            )
    
        # fill I-substates: n+1..2n
        n_I = n_seed - n_E
        if n_I > 0:
            states_new[~is_E] = torch.randint(
                low=self.n + 1,
                high=2*self.n + 1,
                size=(n_I,),
                dtype=self.states.dtype,
                device=infected.device
            )
    
        # assign
        self.states[infected]           = states_new
        self.infection_times[infected]  = -2 # Separate category for seeded cases
        self.infectious_times[infected] = -2
        self.primary_cases[infected]    = -2
    
        return self.states, self.infection_times

    def mpox_vaccination(self, N_vac, strategy, vac_threshold):
        mask = ((self.states == 0) | (self.states == -1))
        if strategy == "from_above":
            vaccinated = torch.arange(self.N)[mask][:N_vac]
            self.states[vaccinated] = -2
        elif strategy == "proportional":
            vaccinated = torch.multinomial(self.r_new * (mask.float()), N_vac, replacement=False)
            self.states[vaccinated] = -2
        elif strategy == "random":
            mask = mask & (self.r_new * 365 > vac_threshold)
            if N_vac > mask.sum():
                print("Too many vaccinations for target group")
                N_vac = mask.sum()
            vaccinated = torch.multinomial(mask.float(), N_vac, replacement=False)
            self.states[vaccinated] = -2
        else:
            raise ValueError("Undefined strategy")
        return self.states

    def housekeeping_war(self, run):
        self.indices = torch.cat((self.indices, self.encounters_today[:-1]), dim=1)
        if self.log:
            self.tot_log = torch.cat((self.tot_log, self.encounters_today), dim=1)
        self.encounters_today = torch.empty((3, 0), dtype=torch.int32)
        sys.stdout.write(f'\r Run: {run}, t: {self.t}, I: {self.I[-1]/(self.N)*100:.2f}%, R: {self.R[-1]/(self.N)*100:.2f}% (including inactive), TI = {self.TI:.2f} days, p = {self.p}')
        sys.stdout.flush()
        self.t += 1

    def estimate_r0(self, t_start=0, t_end=None):
        """
        Estimates R0 as the average number of secondary infections
        for individuals infected between t_start and t_end.

        Parameters:
            t_start (int): Start time of the window (inclusive).
            t_end (int): End time of the window (exclusive). If None, uses the current simulation time self.t.

        Returns:
            R0estimate (float): Estimated R0 value.
        """
        if t_end is None:
            t_end = self.t

        # Exclude infections before t_start and after t_end
        infected_indices = (self.infection_times >= t_start) & (self.infection_times < t_end)

        # Get the secondary infections for those indices
        secondary_infections_in_window = self.secondary_infs[infected_indices]

        # Calculate the mean of these secondary infections
        if secondary_infections_in_window.numel() == 0:
            R0estimate = 0.0  # No infections in window
        else:
            R0estimate = secondary_infections_in_window.float().mean().item()

        return R0estimate

    def reff_in_period(self, ti, tf):
        """
        Calculates the effective reproduction number (Reff) over a specified time period.

        Parameters:
            ti (int): Start time of the period (inclusive).
            tf (int): End time of the period (inclusive).

        Returns:
            R0 (float): The estimated effective reproduction number over the period.
        """
        infected_in_period = (self.infectious_times >= ti) & (self.infectious_times <= tf) # Names of all infected in the period
        if infected_in_period.sum() > 0:
            R0 = self.secondary_infs[infected_in_period].float().mean().item() #Mean number of secondary infections
        else:
            R0 = 0.0
        return R0
    
    def moving_window_reff(self, window_size):
        """
        Computes the effective reproduction number (Reff) over time using a moving window.

        Parameters:
            window_size (int): Size of the window over which to calculate Reff.

        Returns:
            time_points (numpy.ndarray): Array of time points.
            Reff_values (numpy.ndarray): Array of Reff values corresponding to time points.
        """
        #T_max = int(self.infectious_times[self.infectious_times >= 0].max().item())
        time_points = torch.arange(0, self.Tmax)
        Reff_values = torch.zeros(len(time_points))
        for i, t in enumerate(time_points):
            # Define window bounds, ensuring they remain within valid range [0, T_max]
            ti = max(0, t - window_size // 2)
            tf = max(0, min(self.Tmax, t + window_size // 2))
            Reff = self.reff_in_period(ti, tf)
            Reff_values[i] = Reff
        return time_points, Reff_values
    
    def moving_window_cases(self, window_size):
        """
        Computes the number of new cases in a moving time window. For use in weighted averages in analysis

        Parameters:
            window_size (int): Size of the window over which to calculate Reff.

        Returns:
            time_points (numpy.ndarray): Array of time points.
            cases_values (numpy.ndarray): Array of Reff values corresponding to time points.
        """
        #T_max = int(self.infectious_times[self.infectious_times >= 0].max().item())
        time_points = torch.arange(0, self.Tmax)
        cases_values = torch.zeros(len(time_points))
        for i, t in enumerate(time_points):
            # Define window bounds, ensuring they remain within valid range [0, T_max]
            ti = max(0, t - window_size // 2)
            tf = max(0, min(self.Tmax, t + window_size // 2))
            infected_in_period = (self.infection_times >= ti) & (self.infection_times <= tf) # Names of all infected in the period
            cases_values[i] = infected_in_period.sum()
        return time_points, cases_values
    
    def estimate_k(self, ti=1, tf=None): #Dispersion parameter
        """
        Estimates dispersion parameter in a time period
        """
        if tf is None:
            tf = self.t

        # Exclude infections before t_start and after t_end
        infected_indices = (self.infection_times >= ti) & (self.infection_times <= tf)

        # Get the secondary infections for those indices
        secondary_infections_in_window = self.secondary_infs[infected_indices]

        # Calculate R0
        if secondary_infections_in_window.numel() == 0:
            k   = np.nan  # Dispersion parameter is undefined
        else:
            R0      = secondary_infections_in_window.float().mean().item()
            sigma   = secondary_infections_in_window.float().std().item()
            var     = sigma**2
            if var != R0:
                k = R0**2 / (sigma**2 - R0)
            else:
                k = np.nan
        return k
    
    def moving_window_k(self, window_size):
        """
        Computes the dispersion parameter (k) over time using a moving window.

        Parameters:
            window_size (int): Size of the window (in time units) over which to estimate k.

        Returns:
            time_points (torch.Tensor): Array of time points.
            k_values (torch.Tensor): Array of k estimates corresponding to each time point.
        """
        # Determine the maximum infection time (assuming infection_times ≥ 0)
        #T_max = int(self.infection_times[self.infection_times >= 0].max().item())

        time_points = torch.arange(0, self.Tmax)
        # Initialize k_values to NaN by default
        k_values = torch.full((len(time_points),), float('nan'))

        for i, t in enumerate(time_points):
            # Center a window of width `window_size` around t
            half_w = window_size // 2
            ti = max(0, t - half_w)
            tf = min(self.Tmax, t + half_w)

            # estimate_k expects t_end to be exclusive, so add +1 to include tf
            k_est = self.estimate_k(ti=ti, tf=tf)

            # Only store it if it's not NaN
            if k_est == k_est:  # True if k_est is not NaN
                k_values[i] = k_est

        return time_points, k_values
    
    # def compute_generation_and_serial_intervals(self):
    #     """
    #     Computes the generation times and serial intervals for all infected individuals.

    #     Returns:
    #         generation_times (torch.Tensor): Tensor of length N containing the generation times.
    #                                          Set to -10000 for individuals not infected or without a valid primary case.
    #         serial_intervals (torch.Tensor): Tensor of length N containing the serial intervals.
    #                                          Set to -1000 for individuals not infected or without a valid primary case.
    #     """
    #     N = self.N
    #     # Initialize tensors with -10000 for individuals not infected or without a valid primary case
    #     generation_times = torch.full((N,), -10000, dtype=torch.int16)
    #     serial_intervals = torch.full((N,), -10000, dtype=torch.int16)

    #     # Identify valid cases where the individual is infected and has a valid primary case
    #     valid_mask = (self.infection_times >= 0) & (self.primary_cases >= 0)

    #     # Get indices of valid secondary cases
    #     secondary_cases = torch.nonzero(valid_mask).squeeze()
    #     # Get primary cases corresponding to valid secondary cases
    #     primary_cases = self.primary_cases[valid_mask].long()

    #     # Compute generation times
    #     generation_times[secondary_cases] = self.infection_times[secondary_cases] - self.infection_times[primary_cases]

    #     # Compute serial intervals
    #     serial_intervals[secondary_cases] = self.infectious_times[secondary_cases] - self.infectious_times[primary_cases]

    #     return generation_times, serial_intervals

    def gen_ser_in_period(self, ti, tf):
        """
        Computes mean generation time and mean serial interval for individuals whose
        infection time falls within [ti, tf].

        Parameters:
            ti (int): Start time of the window (inclusive).
            tf (int): End time of the window (inclusive).

        Returns:
            mean_gen (float): Mean generation time over the window (nan if no valid cases).
            mean_ser (float): Mean serial interval over the window (nan if no valid cases).
        """
        # “Valid” = infected (infection_times ≥ 0) AND has a valid primary case (primary_cases ≥ 0)
        valid_mask = (self.infection_times >= 0) & (self.primary_cases >= 0)
        # “In window” = infection_time between ti and tf
        in_window = (self.infection_times >= ti) & (self.infection_times <= tf) # The time coordinate corresponds to the secondary case
        mask = valid_mask & in_window

        secondary_idxs = torch.nonzero(mask).squeeze()
        if secondary_idxs.numel() > 0:
            # Primary indices for those secondaries
            primary_idxs = self.primary_cases[mask].long()

            # generation time = infection_time(secondary) – infection_time(primary)
            gen_intervals = (
                self.infection_times[secondary_idxs]
                - self.infection_times[primary_idxs]
            )
            # serial interval = infectious_time(secondary) – infectious_time(primary)
            ser_intervals = (
                self.infectious_times[secondary_idxs]
                - self.infectious_times[primary_idxs]
            )

            mean_gen = gen_intervals.float().mean().item()
            mean_ser = ser_intervals.float().mean().item()
        else:
            mean_gen = float('nan')
            mean_ser = float('nan')

        return mean_gen, mean_ser


    def moving_window_gen_ser(self, window_size):
        """
        Computes mean generation time and mean serial interval over time using a moving window.

        Parameters:
            window_size (int): Width of the window (in time units).

        Returns:
            time_points (torch.Tensor): Tensor of time points (0 .. T_max).
            mean_gen_vals (torch.Tensor): Tensor of mean generation times at each time point.
            mean_ser_vals (torch.Tensor): Tensor of mean serial intervals at each time point.
        """
        # Find the latest infection time among all infected individuals
        #T_max = int(self.infection_times[self.infection_times >= 0].max().item())
        time_points = torch.arange(0, self.Tmax)

        # Initialize output tensors with NaN
        mean_gen_vals = torch.full((len(time_points),), float('nan'))
        mean_ser_vals = torch.full((len(time_points),), float('nan'))

        half_w = window_size // 2
        for i, t in enumerate(time_points):
            # Center the window of width `window_size` around t,
            # making sure it stays within [0, T_max].
            ti = max(0, t - half_w)
            tf = min(self.Tmax, t + half_w)

            mg, ms = self.gen_ser_in_period(ti, tf)

            # Only store if not NaN
            if mg == mg:  # True iff mg is not NaN
                mean_gen_vals[i] = mg
            if ms == ms:  # True iff ms is not NaN
                mean_ser_vals[i] = ms

        return time_points, mean_gen_vals, mean_ser_vals

    def encounters_prior_to_infectious(self, T):
        """
        Computes the number of encounters each infected member had in T days prior to infection.

        Parameters:
            T (int): Number of days prior to infection to consider.

        Returns:
            interactions_count (torch.Tensor): Tensor containing the number of encounters for each member.
                                                Uninfected members have a value of -1.
        """
        N = self.N
        interactions_count = torch.ones(N, dtype=torch.int64) * (-1)
        
        # Filter tot_log for interactions within T days prior to infection
        for i in range(N):
            infectious_time = self.infectious_times[i]
            if infectious_time == -1:
                continue  # Member was never infected
            # Find interactions involving this member
            interactions = (self.tot_log[0] == i) | (self.tot_log[1] == i)
            interaction_times = self.tot_log[2][interactions]
            # Calculate time differences
            time_diffs = infectious_time - interaction_times
            # Count interactions within the T-day window
            prior_interactions = (time_diffs > 0) & (time_diffs <= T)
            interactions_count[i] = prior_interactions.sum().item()
        return interactions_count

    def partners_prior_to_infectious(self, T):
        """
        Computes the number of unique partners each infected member had in T days prior to infection.   
        Parameters:
            T (int): Number of days prior to infection to consider. 
        Returns:
            unique_partners_count (torch.Tensor): Tensor containing the number of unique partners for each member.
                                                  Uninfected members have a value of -1.
        """
        N = self.N
        unique_partners_count = torch.ones(N, dtype=torch.int32) * (-1) #Set to -1 if never infected
        
        for member in range(N):
            infectious_time = self.infectious_times[member]
            if infectious_time == -1:
                continue  # Member was never infected
            # Define the time window
            start_time = infectious_time - T
            # Find interactions within the time window
            time_mask = (self.tot_log[2] >= start_time) & (self.tot_log[2] <= infectious_time)
            interactions_in_window = self.tot_log[:, time_mask]
            # Find interactions involving this member
            member_mask = (interactions_in_window[0] == member) | (interactions_in_window[1] == member) #All members' encounters
            partners = interactions_in_window[0:2, member_mask]
            partners = partners.flatten()
            partners = partners[partners != member]  # Remove self
            unique_partners_count[member] = torch.unique(partners).numel()
        return unique_partners_count
    
    def analyze_encounters(self,tot_log = None):
        """
        Analyzes the encounter log to compute the total encounters and total partners
        for each node.

        Returns:
            tot_encounters (torch.Tensor): Tensor of shape (N,) containing the total
                                           number of encounters for each node.
            tot_partners (torch.Tensor): Tensor of shape (N,) containing the total
                                         number of unique partners for each node.
        """
        N = self.N  # Number of nodes
        if tot_log == None:
            tot_log = self.tot_log  # Shape (3, num_encounters)

        # Get interactions in both directions to make the adjacency matrix symmetric
        interactions_i = torch.cat([tot_log[0, :], tot_log[1, :]]).long()
        interactions_j = torch.cat([tot_log[1, :], tot_log[0, :]]).long()
        
        tot_encounters = torch.bincount(interactions_i, minlength=N)

        # Build indices for the sparse adjacency matrix
        indices = torch.stack([interactions_i, interactions_j], dim=0)  # Shape (2, num_interactions)

        # Create a sparse adjacency matrix
        values = torch.ones(indices.shape[1], dtype=torch.float32)
        adjacency_matrix = torch.sparse_coo_tensor(indices, values, size=(N, N))

        # Coalesce to sum duplicate entries (number of interactions between pairs)
        adjacency_matrix = adjacency_matrix.coalesce()

        # Set all values to 1 to represent existence of a connection
        adjacency_matrix_binary = torch.sparse_coo_tensor(
            adjacency_matrix.indices(),
            torch.ones_like(adjacency_matrix.values()),
            size=(N, N)
        )

        # Use torch.sparse.sum to count the number of unique partners for each node
        tot_partners = torch.sparse.sum(adjacency_matrix_binary, dim=1).to_dense().long().flatten()

        return tot_encounters, tot_partners
    
    def calculate_new_cases(self, period='daily'):
        """
        Calculates daily or weekly new cases and ensures the length of the new cases array
        matches the length of the simulation (i.e., the length of self.I).

        Parameters:
            period (str): 'daily' or 'weekly' for new cases calculation. Default is 'daily'.

        Returns:
            tuple: new_cases_time_axis (torch.Tensor), new_cases (torch.Tensor)
        """
        infection_times = self.infection_times #Log when agent becomes infection/symptomatic
        infection_mask = infection_times >= 0 #Uninfected have -1 - exclude them
        infection_times = infection_times[infection_mask].long()

        # Get the length of the simulation based on self.I
        #simulation_length = len(self.I)
        simulation_length = self.Tmax

        if period == 'daily':
            daily_new_cases = torch.zeros(simulation_length, dtype=torch.int32)
            if infection_times.numel() > 0:
                counts = torch.bincount(infection_times)
                daily_new_cases[:len(counts)] = counts  # Fill in the daily new cases
            new_cases_time_axis = torch.arange(simulation_length)
            new_cases = daily_new_cases
        elif period == 'weekly':
            week_numbers = (infection_times // 7).long()
            max_week = (simulation_length + 6) // 7  # Ensure weekly length covers the simulation
            weekly_new_cases = torch.zeros(max_week, dtype=torch.int32)
            if week_numbers.numel() > 0:
                counts = torch.bincount(week_numbers)
                weekly_new_cases[:len(counts)] = counts  # Fill in the weekly new cases
            new_cases_time_axis = torch.arange(max_week)
            new_cases = weekly_new_cases
        else:
            raise ValueError("period should be 'daily' or 'weekly'")

        return new_cases_time_axis, new_cases

    def plot_epidemic(self, TE=None, window_size=7, period='daily', figsize=(10, 15)):
        """
        Plots the epidemic data in three separate panes arranged vertically:
        1. Cumulative cases (E + I + R) over time.
        2. Daily or weekly new cases over time.
        3. Effective reproduction number over time.

        Parameters:
            TE (int, optional): Latency period. If None, uses self.TE.
            window_size (int): Window size for Reff calculation. Default is 7.
            period (str): 'daily' or 'weekly' for new cases plot. Default is 'daily'.
            figsize (tuple): Figure size in inches (width, height). Default is (10, 15).
        """
        if TE is None:
            TE = self.TE

        # Compute cumulative cases over time
        cumulative_cases = self.I + self.R #Not counting presymptomatic
        time_axis = torch.arange(len(cumulative_cases))

        # Calculate daily or weekly new cases
        new_cases_time_axis, new_cases = self.calculate_new_cases(period)

        # Set labels and titles for plotting
        if period == 'daily':
            x_label_new_cases = 'Time (Days)'
            title_new_cases = 'Daily New Cases'
        elif period == 'weekly':
            x_label_new_cases = 'Time (Weeks)'
            title_new_cases = 'Weekly New Cases'

        # Compute Reff over time
        time_points, Reff_values = self.moving_window_reff(window_size)

        # Plotting
        plt.figure(figsize=figsize)

        # First subplot: cumulative cases
        plt.subplot(3, 1, 1)
        plt.plot(time_axis.numpy(), cumulative_cases.numpy(), label='Cumulative Cases')
        plt.xlabel('Time')
        plt.ylabel('Number of Cases')
        plt.title('Cumulative Cases Over Time')
        plt.legend()
        plt.grid(True)

        # Second subplot: daily or weekly new cases
        plt.subplot(3, 1, 2)
        plt.bar(new_cases_time_axis.numpy(), new_cases.numpy(), color='orange')
        plt.xlabel(x_label_new_cases)
        plt.ylabel('Number of New Cases')
        plt.title(title_new_cases)
        plt.grid(True)

        # Third subplot: Reff over time
        plt.subplot(3, 1, 3)
        plt.plot(time_points, Reff_values, label='Effective Reproduction Number', color='green')
        plt.xlabel('Time')
        plt.ylabel('Reff')
        plt.title('Effective Reproduction Number Over Time')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
    
    def infection_probability_by_r_new(self, n_bins):
        """
        Calculates the probability for members in each bin of having reached the R-state,
        assuming they did not have smallpox vaccination.

        Parameters:
            n_bins (int): Number of bins to divide the members into based on r_new values.

        Returns:
            bin_centers (torch.Tensor): The center values of the bins.
            probabilities (torch.Tensor): The probability of being in R-state for each bin.
        """
        # Get r_new values (per day) for all members
        r_new = self.r_new

        # Get vaccination status (-1 indicates vaccinated)
        is_vaccinated = (self.states == -1)

        # Get recovery status (2 * n + 1 indicates R-state)
        is_recovered = (self.states == 2 * self.n + 1)

        # Only consider non-vaccinated members
        non_vaccinated_mask = ~is_vaccinated

        # Filter r_new and recovery status for non-vaccinated members
        r_new_nv = r_new[non_vaccinated_mask]
        is_recovered_nv = is_recovered[non_vaccinated_mask]

        # Define bin edges linearly over the range of r_new
        min_r_new = r_new.min()
        max_r_new = r_new.max()
        bin_edges = torch.linspace(min_r_new, max_r_new, n_bins + 1)

        # Assign each r_new_nv to a bin index
        bin_indices = torch.bucketize(r_new_nv, bin_edges, right=False) - 1
        # Ensure bin indices are within [0, n_bins - 1]
        bin_indices = bin_indices.clamp(0, n_bins - 1)

        # Initialize tensors to hold counts
        total_counts = torch.zeros(n_bins, dtype=torch.int32)
        recovered_counts = torch.zeros(n_bins, dtype=torch.int32)

        # Accumulate counts for each bin
        for i in range(n_bins):
            in_bin = (bin_indices == i)
            total_counts[i] = in_bin.sum()
            recovered_counts[i] = is_recovered_nv[in_bin].sum()

        # Calculate probabilities, avoiding division by zero
        probabilities = torch.where(
            total_counts > 0,
            recovered_counts.float() / total_counts.float(),
            torch.tensor(float('nan'))
        )

        # Calculate bin centers for plotting or analysis
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        return bin_centers, probabilities

    def infection_probability_by_partners(self, n_bins, tot_partners=None):
        """
        Calculates the probability for members in each bin of having reached the R-state,
        stratified by total number of partners.
    
        Parameters:
            n_bins (int): Number of bins to divide the members into based on tot_partners values.
            tot_partners (torch.Tensor, optional): Pre-computed total partners tensor. If None,
                                                   it will be calculated using self.analyze_encounters().
    
        Returns:
            bin_centers (torch.Tensor): The center values of the bins.
            probabilities (torch.Tensor): The probability of being in R-state for each bin.
        """
        # If tot_partners is not provided, compute it
        if tot_partners is None:
            _, tot_partners = self.analyze_encounters()
        
        if not isinstance(tot_partners, torch.Tensor):
            tot_partners = torch.tensor(tot_partners, dtype=torch.float32)
        else:
            tot_partners = tot_partners.clone().detach().float()
    
        # Get vaccination status (-1 indicates vaccinated)
        is_vaccinated = (self.states == -1)
    
        # Get recovery status (2 * n + 1 indicates R-state)
        is_recovered = (self.states == 2 * self.n + 1)
    
        # Only consider non-vaccinated members
        non_vaccinated_mask = ~is_vaccinated
    
        # Filter tot_partners and recovery status for non-vaccinated members
        tot_partners_nv = tot_partners[non_vaccinated_mask]
        is_recovered_nv = is_recovered[non_vaccinated_mask]
    
        # Define bin edges linearly over the range of tot_partners_nv
        min_partners = tot_partners_nv.min()
        max_partners = tot_partners_nv.max()
        bin_edges = torch.linspace(min_partners, max_partners, n_bins + 1)
    
        # Assign each tot_partners_nv to a bin index
        bin_indices = torch.bucketize(tot_partners_nv, bin_edges, right=False) - 1
        # Ensure bin indices are within [0, n_bins - 1]
        bin_indices = bin_indices.clamp(0, n_bins - 1)
    
        # Initialize tensors to hold counts
        total_counts = torch.zeros(n_bins, dtype=torch.int32)
        recovered_counts = torch.zeros(n_bins, dtype=torch.int32)
    
        # Accumulate counts for each bin
        for i in range(n_bins):
            in_bin = (bin_indices == i)
            total_counts[i] = in_bin.sum()
            recovered_counts[i] = is_recovered_nv[in_bin].sum()
    
        # Calculate probabilities, avoiding division by zero
        probabilities = torch.where(
            total_counts > 0,
            recovered_counts.float() / total_counts.float(),
            torch.tensor(float('nan'))
        )
    
        # Calculate bin centers for plotting or analysis
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
        return bin_centers, probabilities

    ####################################################################################
    # Debugging #
    ####################################################################################
    def plot_sanity_check(self,tot_log = None):
        """
        Plots the relationship between individual seeking rates and the number of encounters and partners.
        """
        # Get total encounters and total partners (over Twindow)
        if tot_log == None:
            tot_encounters, tot_partners = self.analyze_encounters()
        else:
            tot_encounters, tot_partners = self.analyze_encounters(tot_log)

        # Get r_new and r_old (per day)
        r_new = self.r_new
        r_old = self.r_old

        # Convert r_new and r_old to per week rates
        r_new_per_week = r_new * 7
        r_old_per_week = r_old * 7

        # Calculate average encounters and partners per week (over Twindow)
        avg_encounters_per_week = tot_encounters / (self.Twindow / 7)
        avg_partners_per_week = tot_partners / (self.Twindow / 7)

        # Plotting
        _, axes = plt.subplots(2, 2, figsize=(10, 8))

        # Plot 1: r_new vs avg_encounters_per_week
        axes[0, 0].plot(r_new_per_week, avg_encounters_per_week, ".")
        axes[0, 0].plot([0, r_new_per_week.max()], [0, r_new_per_week.max()], 'r')
        axes[0, 0].set_xlabel('r_new (per week)')
        axes[0, 0].set_ylabel('Average encounters per week')

        # Plot 2: r_old vs avg_encounters_per_week
        axes[0, 1].plot(r_old_per_week, avg_encounters_per_week, ".")
        axes[0, 1].plot([0, r_old_per_week.max()], [0, r_old_per_week.max()], 'r')
        axes[0, 1].set_xlabel('r_old (per week)')
        axes[0, 1].set_ylabel('Average encounters per week')

        # Plot 3: r_new + r_old vs avg_encounters_per_week
        total_r_per_week = r_new_per_week + r_old_per_week
        axes[1, 0].plot(total_r_per_week, avg_encounters_per_week, ".")
        axes[1, 0].plot([0, total_r_per_week.max()], [0, total_r_per_week.max()], 'r')
        axes[1, 0].set_xlabel('r_new + r_old (per week)')
        axes[1, 0].set_ylabel('Average encounters per week')

        # Plot 4: r_new vs avg_partners_per_week
        axes[1, 1].plot(r_new_per_week, avg_partners_per_week, ".")
        axes[1, 1].plot([0, r_new_per_week.max()], [0, r_new_per_week.max()], 'r')
        axes[1, 1].set_xlabel('r_new (per week)')
        axes[1, 1].set_ylabel('Average partners per week')

        plt.tight_layout()
        plt.show()
    
    def plot_encounters_over_time(self,xlim = None):
        """
        Plots the number of encounters each day over time, based on tot_log.
        """
        # Extract time values from tot_log
        t_values = self.tot_log[2]
        
        # Find minimum and maximum t-values
        t_min = t_values.min().item()
        t_max = t_values.max().item()
        
        # Shift t_values to start from zero to handle negative times
        t_values_shifted = t_values - t_min
        
        # Count the number of encounters per day using torch.bincount
        counts = torch.bincount(t_values_shifted)
        
        # Create time axis corresponding to original t_values
        time_axis = torch.arange(t_min, t_max + 1)
        
        # Plot the counts over time
        plt.figure(figsize=(10, 6))
        plt.plot(time_axis.numpy(), counts.numpy(), marker='o')
        plt.xlabel('Time')
        plt.ylabel('Number of Encounters')
        plt.title('Number of Encounters per Day Over Time')
        if xlim is not None:
            plt.xlim(xlim)
        plt.grid(True)
        plt.show()

#Generate TI values from sigmoid function
def sigmoid(t, k, t0, TSmax = 14, TSmin = 6.5):
    return TSmax - (TSmax - TSmin) / (1 + np.exp(-k * (t - t0)))

def generate_sigmoid_values(start_date, end_date, t0_date, k, TSmax = 14,TSmin = 6.5):
    date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    t0 = (t0_date - start_date).days

    results = []
    for date in date_range:
        t = (date - start_date).days
        value = sigmoid(t, k, t0, TSmax,TSmin)
        results.append((date.strftime("%Y-%m-%d"), value))

    return results