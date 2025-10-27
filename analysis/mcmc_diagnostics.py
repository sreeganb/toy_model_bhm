"""MCMC convergence diagnostics including R-hat calculation"""

import numpy as np
import h5py
import os
from typing import Dict, List, Any

def run_mcmc_diagnostics(
    stage_output: str,
    stage_name: str,
    chain_dirs: List[str],
    rmf_files: List[str]
) -> Dict[str, Any]:
    """
    Run comprehensive MCMC diagnostics on multiple chains
    
    Args:
        stage_output: Directory containing all chain outputs
        stage_name: Name of the sampling stage
        chain_dirs: List of chain directory names
        rmf_files: List of RMF3 file paths
    
    Returns:
        Dictionary containing R-hat values, ESS, and other diagnostics
    """
    results = {
        'stage_name': stage_name,
        'n_chains': len(chain_dirs),
        'rhat': {},
        'ess': {},
        'summary_stats': {}
    }
    
    # Extract sigma trajectories from all chains
    sigma_chains = extract_sigma_trajectories(stage_output, chain_dirs)
    
    if not sigma_chains:
        print("    Warning: No sigma trajectories found!")
        return results
    
    # Calculate R-hat for each sigma parameter
    print(f"\n    Calculating R-hat values...")
    for param_name in sigma_chains[0].keys():
        # Collect this parameter across all chains
        param_chains = [chain[param_name] for chain in sigma_chains]
        
        # Calculate R-hat
        rhat = calculate_rhat(param_chains)
        results['rhat'][param_name] = float(rhat)
        
        # Calculate effective sample size
        ess = calculate_ess(param_chains)
        results['ess'][param_name] = float(ess)
        
        # Summary statistics
        all_samples = np.concatenate(param_chains)
        results['summary_stats'][param_name] = {
            'mean': float(np.mean(all_samples)),
            'std': float(np.std(all_samples)),
            'median': float(np.median(all_samples)),
            'min': float(np.min(all_samples)),
            'max': float(np.max(all_samples))
        }
    
    # create the pdf plots 
    print(f"\n    Generating diagnostic plots...")
    plot_diagnostics(stage_output, chain_dirs)
    
    return results

def extract_sigma_trajectories(stage_output: str, chain_dirs: List[str]) -> List[Dict[str, np.ndarray]]:
    """Extract sigma parameter trajectories from H5 files"""
    all_chains = []
    
    for chain_dir in chain_dirs:
        h5_file = os.path.join(stage_output, chain_dir, "trajectory.h5")
        
        if not os.path.exists(h5_file):
            continue
        
        chain_sigmas = {}
        
        try:
            with h5py.File(h5_file, 'r') as f:
                if 'trajectory' not in f:
                    continue
                
                traj_group = f['trajectory']
                state_names = sorted([k for k in traj_group.keys() if k.startswith('state_')])
                
                if not state_names:
                    continue
                
                # Get sigma parameter names from first state
                first_state = traj_group[state_names[0]]
                if 'sigma' not in first_state:
                    continue
                
                sigma_group = first_state['sigma']
                param_names = list(sigma_group.attrs.keys())
                
                # Initialize arrays for each parameter
                for param in param_names:
                    chain_sigmas[param] = []
                
                # Extract values from each state
                for state_name in state_names:
                    state = traj_group[state_name]
                    if 'sigma' in state:
                        sigma_vals = state['sigma']
                        for param in param_names:
                            if param in sigma_vals.attrs:
                                chain_sigmas[param].append(sigma_vals.attrs[param])
                
                # Convert to numpy arrays
                for param in param_names:
                    chain_sigmas[param] = np.array(chain_sigmas[param])
                
                all_chains.append(chain_sigmas)
                
        except Exception as e:
            print(f"    Warning: Failed to read {h5_file}: {e}")
    
    return all_chains

def calculate_rhat(chains: List[np.ndarray], burn_in_frac: float = 0.5) -> float:
    """
    Calculate Gelman-Rubin R-hat statistic for convergence
    
    Args:
        chains: List of numpy arrays, one per chain
        burn_in_frac: Fraction of samples to discard as burn-in
    
    Returns:
        R-hat value (should be < 1.1 for convergence)
    """
    if not chains or len(chains) < 2:
        return np.nan
    
    # Remove burn-in
    chains_trimmed = [chain[int(len(chain) * burn_in_frac):] for chain in chains]
    
    # Check for minimum length
    n_samples = min(len(chain) for chain in chains_trimmed)
    if n_samples < 2:
        return np.nan
    
    n_chains = len(chains_trimmed)
    
    # Trim all chains to same length
    chains_array = np.array([chain[:n_samples] for chain in chains_trimmed])
    
    # Check for zero variance (all values identical)
    if np.var(chains_array) == 0:
        return 1.0  # Perfect convergence if all chains agree
    
    # Calculate between-chain variance (B)
    chain_means = np.mean(chains_array, axis=1)
    overall_mean = np.mean(chain_means)
    B = n_samples * np.var(chain_means, ddof=1)
    
    # Calculate within-chain variance (W)
    chain_vars = np.var(chains_array, axis=1, ddof=1)
    W = np.mean(chain_vars)
    
    # Handle case where W is zero or very small
    if W < 1e-10:
        # If within-chain variance is tiny but between-chain variance exists
        if B > 1e-10:
            return np.inf  # Chains haven't mixed at all
        else:
            return 1.0  # All chains are identical
    
    # Calculate pooled variance
    var_plus = ((n_samples - 1) * W + B) / n_samples
    
    # Calculate R-hat with safety check
    rhat = np.sqrt(var_plus / W)
    
    # Return NaN if result is invalid
    if not np.isfinite(rhat):
        return np.nan
    
    return rhat

def calculate_ess(chains: List[np.ndarray], burn_in_frac: float = 0.5) -> float:
    """
    Calculate effective sample size
    
    Args:
        chains: List of numpy arrays, one per chain
        burn_in_frac: Fraction to discard as burn-in
    
    Returns:
        Effective sample size
    """
    # Remove burn-in and combine chains
    chains_trimmed = [chain[int(len(chain) * burn_in_frac):] for chain in chains]
    all_samples = np.concatenate(chains_trimmed)
    
    n_total = len(all_samples)
    
    # Calculate autocorrelation
    mean = np.mean(all_samples)
    var = np.var(all_samples)
    
    if var == 0:
        return n_total
    
    # Calculate autocorrelation up to lag of n_total/2
    max_lag = min(n_total // 2, 1000)
    autocorr = np.zeros(max_lag)
    
    for lag in range(max_lag):
        autocorr[lag] = np.mean(
            (all_samples[:-lag or None] - mean) * (all_samples[lag:] - mean)
        ) / var
    
    # Sum autocorrelation until it becomes negative
    rho_sum = 1.0  # Start with lag 0
    for rho in autocorr[1:]:
        if rho < 0:
            break
        rho_sum += 2 * rho
    
    ess = n_total / rho_sum
    
    return ess

#------------------------------------------------------------
# Function to plot the time series of sigma parameters, AA, AB and BC
# clubbed together for each chain in a single figure, plotted together in the same
# axis, saved as pdfpages in the directory stage_output/diagnostics created if not
# already present. then pdfpages adds one page per chain for the sigma parameters, 
# not only that, one page per chain for scores vs step as well, including the pair score
# components, exvol, tet, oct, pair, total score. at the end of this pdf document I need
# to add a summary page with the R-hat values for each sigma parameter, and the ESS values
# for each sigma parameter as well. 
#------------------------------------------------------------
def plot_diagnostics(stage_output: str, chain_dirs: List[str]) -> None:
    """Plot sigma parameter trajectories and scores for diagnostics"""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    diag_dir = os.path.join(stage_output, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    pdf_path = os.path.join(diag_dir, "mcmc_diagnostics.pdf")
    
    with PdfPages(pdf_path) as pdf:
        for chain_dir in chain_dirs:
            h5_file = os.path.join(stage_output, chain_dir, "trajectory.h5")
            if not os.path.exists(h5_file):
                continue
            
            try:
                with h5py.File(h5_file, 'r') as f:
                    if 'trajectory' not in f:
                        continue
                    
                    traj_group = f['trajectory']
                    state_names = sorted([k for k in traj_group.keys() if k.startswith('state_')])
                    
                    if not state_names:
                        continue
                    
                    # Extract sigma parameters and scores
                    sigma_params = {}
                    scores = {
                        'total_score': [],
                        'pair_score': [],
                        'exvol_score': [],
                        'tet_score': [],
                        'oct_score': []
                    }
                    
                    for state_name in state_names:
                        state = traj_group[state_name]
                        
                        # Sigma parameters
                        if 'sigma' in state:
                            sigma_vals = state['sigma']
                            for param in sigma_vals.attrs.keys():
                                if param not in sigma_params:
                                    sigma_params[param] = []
                                sigma_params[param].append(sigma_vals.attrs[param])
                        
                        # Scores
                        for score_key in scores.keys():
                            if score_key in state.attrs:
                                scores[score_key].append(state.attrs[score_key])
                            else:
                                scores[score_key].append(0.0)
                    
                    # Convert to numpy arrays
                    for param in sigma_params.keys():
                        sigma_params[param] = np.array(sigma_params[param])
                    for score_key in scores.keys():
                        scores[score_key] = np.array(scores[score_key])
                    
                    # Plot sigma parameters
                    plt.figure(figsize=(10, 6))
                    for param, values in sigma_params.items():
                        plt.plot(values, label=param)
                    plt.title(f"Sigma Parameters - Chain: {chain_dir}")
                    plt.xlabel("Step")
                    plt.ylabel("Sigma Value")
                    plt.legend()
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()
                    # Plot scores
                    plt.figure(figsize=(10, 6))
                    for score_key, values in scores.items():
                        plt.plot(values, label=score_key)
                    plt.title(f"Scores - Chain: {chain_dir}")
                    plt.xlabel("Step")
                    plt.ylabel("Score Value")
                    plt.legend()
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close() 
            except Exception as e:
                print(f"    Warning: Failed to plot diagnostics for {h5_file}: {e}")
            