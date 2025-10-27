# analysis.py
import numpy as np
import h5py
import os
import glob
from typing import Dict, List, Any
import json

def load_trajectories(stage_output: str) -> Dict[str, np.ndarray]:
    """Load all trajectory files from a stage output directory"""
    trajectories = {}
    
    # Find all chain directories
    chain_dirs = glob.glob(os.path.join(stage_output, "chain_*"))
    
    for chain_dir in sorted(chain_dirs):
        chain_id = os.path.basename(chain_dir)
        
        # Find trajectory file in this chain
        h5_files = glob.glob(os.path.join(chain_dir, "*.h5"))
        if h5_files:
            trajectory_file = h5_files[0]  # Take first .h5 file
            
            with h5py.File(trajectory_file, 'r') as f:
                # Load sigma trajectories
                if 'sigma' in f:
                    trajectories[chain_id] = {}
                    for sigma_key in f['sigma'].keys():
                        trajectories[chain_id][sigma_key] = f['sigma'][sigma_key][:]
    
    return trajectories

def calculate_rhat(chains: List[np.ndarray], burnin_frac: float = 0.5) -> float:
    """Calculate R-hat statistic for convergence assessment"""
    n_chains = len(chains)
    
    if n_chains < 2:
        return 1.0
    
    # Remove burnin
    chains_burned = [chain[int(len(chain) * burnin_frac):] for chain in chains]
    n_samples = min(len(chain) for chain in chains_burned)
    
    if n_samples < 2:
        return 1.0
    
    # Truncate to same length
    chains_burned = [chain[:n_samples] for chain in chains_burned]
    
    # Convert to array
    chains_array = np.array(chains_burned)  # shape: (n_chains, n_samples)
    
    # Calculate within-chain variance
    W = np.mean(np.var(chains_array, axis=1, ddof=1))
    
    # Calculate between-chain variance
    chain_means = np.mean(chains_array, axis=1)
    overall_mean = np.mean(chain_means)
    B = n_samples * np.var(chain_means, ddof=1)
    
    # Calculate R-hat
    var_est = ((n_samples - 1) / n_samples) * W + (1 / n_samples) * B
    rhat = np.sqrt(var_est / W) if W > 0 else 1.0
    
    return rhat

def calculate_effective_sample_size(chains: List[np.ndarray], burnin_frac: float = 0.5) -> float:
    """Calculate effective sample size"""
    # Simple implementation - can be made more sophisticated
    n_chains = len(chains)
    chains_burned = [chain[int(len(chain) * burnin_frac):] for chain in chains]
    total_samples = sum(len(chain) for chain in chains_burned)
    
    # Rough approximation - should implement autocorrelation analysis
    return total_samples / (2 * n_chains)  # Conservative estimate

def run_mcmc_diagnostics(stage_output: str, stage_name: str) -> Dict[str, Any]:
    """Run comprehensive MCMC diagnostics"""
    
    # Load all trajectories
    trajectories = load_trajectories(stage_output)
    
    if not trajectories:
        return {"error": "No trajectories found"}
    
    # Get all sigma parameters
    first_chain = list(trajectories.values())[0]
    sigma_params = list(first_chain.keys())
    
    results = {
        "stage_name": stage_name,
        "n_chains": len(trajectories),
        "n_parameters": len(sigma_params),
        "rhat": {},
        "effective_sample_size": {},
        "convergence_summary": {}
    }
    
    # Calculate diagnostics for each parameter
    for param in sigma_params:
        # Extract parameter chains
        param_chains = []
        for chain_data in trajectories.values():
            if param in chain_data:
                param_chains.append(chain_data[param])
        
        if len(param_chains) >= 2:
            # Calculate R-hat
            rhat = calculate_rhat(param_chains)
            results["rhat"][param] = float(rhat)
            
            # Calculate effective sample size
            ess = calculate_effective_sample_size(param_chains)
            results["effective_sample_size"][param] = float(ess)
            
            # Convergence assessment
            results["convergence_summary"][param] = {
                "converged": rhat < 1.1,
                "rhat": float(rhat),
                "ess": float(ess)
            }
    
    # Overall convergence assessment
    all_rhat = list(results["rhat"].values())
    results["overall_convergence"] = {
        "all_converged": all(r < 1.1 for r in all_rhat),
        "max_rhat": max(all_rhat) if all_rhat else 1.0,
        "mean_rhat": np.mean(all_rhat) if all_rhat else 1.0
    }
    
    return results