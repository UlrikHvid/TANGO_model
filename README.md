# Agent-Based Model of Sexual Pair Formation and STI Spread

This repository contains the code, data, and documentation for an agent-based model (ABM) simulating sexual pair formation and the spread of sexually transmitted infections (STIs) on pre-generated networks.

---

## Table of Contents

1. [Overview](#overview)

2. [Repository Structure](#repository-structure)

3. [Installation](#installation)

4. [Usage](#usage)

   - [Running Pre-saved Networks (Epidemic Simulations)](#running-pre-saved-networks-epidemic-simulations)
   - [Generating New Networks (Peacetime Analyses)](#generating-new-networks-peacetime-analyses)

5. [Data and File Formats](#data-and-file-formats)

6. [Licensing](#licensing)

7. [Citing This Work](#citing-this-work)

8. [Contact](#contact)

---

## Overview

This project implements an agent-based model to study:

- Formation and dynamics of sexual partnerships on social networks.
- Transmission of STIs across dynamic partnerships.

Core components include:

- **Network generation**: scripts to build synthetic sexual networks until steady state.
- **Epidemic simulation**: notebooks for running disease-spread experiments on saved networks.
- **Analysis tools**: notebooks for exploring partnership patterns without disease dynamics.

---

## Repository Structure

```
/                  # Root of repository
├── Saved_networks/         # Pre-generated networks and encounter logs
│   ├── network_N=XXX_ub=YYY.csv   # Edge lists: (agent_i, agent_j, tie strength)
│   ├── member_data_N=XXX_ub=YYY.csv  # Agent attributes and constant variables
│   └── tot_log_N=XXX_ub=YYY.csv    # Encounter logs over final simulated year
│
├── wartime_notebook.ipynb  # Epidemic simulations on pre-saved networks
├── peacetime_notebook.ipynb # Partnership dynamics analyses (no disease)
│
├── network_generator.py    # Generates and saves new networks to /Saved_networks
├── tango_model.py          # Core ABM functions and classes
├── requirements.txt        # Python dependencies
└── README.md               # Project overview and instructions
```

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/UlrikHvid/TANGO_model.git
   cd TANGO_model
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional)** Generate or update `requirements.txt`

   ```bash
   pip freeze > requirements.txt
   ```

---

## Usage

### Running Pre-saved Networks (Epidemic Simulations)

1. Open `wartime_notebook.ipynb` in Jupyter or VS Code.
2. Modify any simulation parameters (e.g., transmission probability, initial prevalence) at the top.
3. Execute cells sequentially to load a pre-saved network from `Saved_networks/`, run the epidemic, and visualize metrics.

### Generating New Networks (Peacetime Analyses)

1. Open `peacetime_notebook.ipynb`.
2. Set network size (`N`) and partnership-formation parameters in the configuration block.
3. Run all cells to simulate partnership dynamics and analyze steady-state properties.
4. To save a new network for later epidemic runs, use the `network_generator.py` script.

---

## Data and File Formats

- ``: CSV of network edges. Each row has three values: `agent_i`, `agent_j`, and `strength`, representing a tie between those agents. Example: `network_N=4.0e+04_ub=300.csv`.

- ``: Agent metadata table. Columns include:

  - `agent_id`
  - constant agent-specific rates: `r_new` (alpha), `r_old` (beta)
  - counts over the final simulated year: `num_partners`, `num_encounters` Example: `member_data_N=4.0e+04_ub=300.csv`.

- ``: Detailed encounter log. Columns include:

  - `agent_id`
  - `partner_id`
  - `timestamp` (time of each partnership encounter) Example: `tot_log_N=4.0e+04_ub=300.csv`.

To load these in Python:

```python
import pandas as pd

edges = pd.read_csv(
    'Saved_networks/network_N=4.0e+04_ub=300.csv',
    header=None,
    names=['agent_i', 'agent_j', 'strength']
)

member_data = pd.read_csv(
    'Saved_networks/member_data_N=4.0e+04_ub=300.csv'
)

encounters = pd.read_csv(
    'Saved_networks/tot_log_N=4.0e+04_ub=300.csv'
)
```

---

## Licensing

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details. You are free to use, modify, and distribute this code, provided you include attribution.

---

## Citing This Work

If you use this model in your research, please cite:

> **Author(s)**, *Title of Your Model Paper*, Journal Name, Year.

And if you archive the code via Zenodo, cite the DOI:

> `DOI:10.5281/zenodo.YOUR_DOI`

---

## Contact

For questions or contributions, please open an issue or contact:

- **Name:** Ulrik Hvid
- **Email:** [ulrik.hvid@nbi.ku.dk](mailto\:ulrik.hvid@nbi.ku.dk)

