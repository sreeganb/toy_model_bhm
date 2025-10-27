# pipeline.py - Integrated with replica exchange support
from typing import List, Dict, Any, Optional, Tuple
import os
import json
import time
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import random
import h5py
from sklearn.mixture import GaussianMixture
import warnings

from analysis.mcmc_diagnostics import run_mcmc_diagnostics
from core.state import SystemState
from core.sigma import create_sigma_prior, get_default_sigma_ranges, initialize_sigma_dict, DEFAULT_SIGMA_RANGES
from core.system import SystemBuilder
from core.parameters import SystemParameters


class SamplerPipeline:
    """
    Run a sequence of samplers in a pipeline with analysis between stages.
    
    Key Design:
    -----------
    - Stage 0: Initialize from scratch (random/ideal)
    - Stage N>0: Each chain randomly picks one chain from stage N-1
                 Inherits BOTH positions AND sigma from that chain's last frame
    """
    
    def __init__(self, initial_state: SystemState, base_seed: int = 1234, 
                 init_mode: str = "random", prior_type: str = "uniform"):
        self.initial_state = initial_state
        self.stages = []
        self.base_seed = int(base_seed)
        self.init_mode = init_mode
        self.prior_type = prior_type
                    
    def add_stage(self, sampler_fn, n_steps, save_freq=100, **kwargs):
        """Add a sampling stage - extract components for replica exchange"""
        
        # Extract module and function info
        sampler_name = sampler_fn.__name__
        sampler_module = sampler_fn.__module__
        
        # Import the sampler module to get its components
        import importlib
        module = importlib.import_module(sampler_module)
        
        # Get the neg_log_posterior function from the module
        score_fn = getattr(module, 'neg_log_posterior', None)
        
        # Get proposal functions and move probabilities based on sampler type
        if 'pair' in sampler_name.lower():
            from core.movers import propose_particle_move, propose_sigma_move
            propose_fns = {
                'position': propose_particle_move,
                'sigma': propose_sigma_move
            }
            move_probs = {'position': 0.7, 'sigma': 0.3}
            
        elif 'tetramer' in sampler_name.lower():
            from core.movers import propose_particle_move, propose_sigma_move, propose_tetramer_move
            propose_fns = {
                'tetramer': propose_tetramer_move,
                'position': propose_particle_move,
                'sigma': propose_sigma_move
            }
            move_probs = {'tetramer': 0.60, 'position': 0.25, 'sigma': 0.15}
            
        elif 'octet' in sampler_name.lower():
            from core.movers import propose_particle_move, propose_sigma_move, propose_tetramer_move, propose_octet_move
            propose_fns = {
                'octet': propose_octet_move,
                'tetramer': propose_tetramer_move,
                'position': propose_particle_move,
                'sigma': propose_sigma_move
            }
            move_probs = {'octet': 0.50, 'tetramer': 0.20, 'position': 0.20, 'sigma': 0.10}
            
        else:
            raise ValueError(f"Unknown sampler type: {sampler_name}")
        
        if score_fn is None:
            raise ValueError(f"Could not find neg_log_posterior in {sampler_module}")
        
        self.stages.append({
            'name': sampler_name.replace('run_', ''),
            'function': sampler_fn,
            'score_fn': score_fn,  # Direct reference to neg_log_posterior
            'propose_fns': propose_fns,
            'move_probs': move_probs,
            'n_steps': n_steps,
            'kwargs': {'save_freq': save_freq, **kwargs}
        })
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def _seed_everything(self, seed: int) -> None:
        """Seed all RNGs for reproducibility"""
        np.random.seed(seed)
        random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except:
            pass

    # =========================================================================
    # SIGMA LOADING/SAVING
    # =========================================================================
    
    def _load_sigma_from_trajectory(self, h5_file: str) -> Tuple[Dict[str, float], bool]:
        """
        Load sigma dict from the last frame of trajectory.h5.
        
        Returns:
            (sigma_dict, success)
        """
        if not h5_file or not os.path.exists(h5_file):
            return {}, False
        
        try:
            with h5py.File(h5_file, 'r') as f:
                if 'trajectory' not in f:
                    return {}, False
                
                traj = f['trajectory']
                state_names = sorted([k for k in traj.keys() if k.startswith('state_')])
                
                if not state_names:
                    return {}, False
                
                last_state = traj[state_names[-1]]
                
                if 'sigma' not in last_state:
                    return {}, False
                
                sigma_grp = last_state['sigma']
                result = {str(key): float(sigma_grp.attrs[key]) 
                         for key in sigma_grp.attrs.keys()}
                
                return result, True
                
        except Exception:
            return {}, False

    def _extract_sigma_samples(self, h5_file: str, burn_in: float = 0.5) -> Dict[str, np.ndarray]:
        """
        Extract all sigma samples from H5 trajectory after burn-in.
        
        Used for fitting GMM posterior for next stage.
        """
        if not os.path.exists(h5_file):
            return {}
        
        sigma_samples = {}
        
        try:
            with h5py.File(h5_file, 'r') as f:
                if 'trajectory' not in f:
                    return {}
                
                traj = f['trajectory']
                state_names = sorted([k for k in traj.keys() if k.startswith('state_')])
                
                # Apply burn-in
                start_idx = int(len(state_names) * burn_in)
                state_names = state_names[start_idx:]
                
                # Extract sigma from each frame
                for state_name in state_names:
                    state = traj[state_name]
                    if 'sigma' not in state:
                        continue
                    
                    sigma_grp = state['sigma']
                    for key in sigma_grp.attrs.keys():
                        if key not in sigma_samples:
                            sigma_samples[key] = []
                        sigma_samples[key].append(float(sigma_grp.attrs[key]))
            
            return {k: np.array(v) for k, v in sigma_samples.items()}
            
        except Exception:
            return {}

    # =========================================================================
    # STATE INITIALIZATION
    # =========================================================================
    
    def _create_initial_states(self, stage_idx: int, n_chains: int, 
                               prev_stage_output: Optional[str]) -> List[SystemState]:
        """
        Create initial states for a stage.
        
        Stage 0: Random or ideal initialization
        Stage N>0: Each chain randomly picks a chain from previous stage
        
        Returns:
            List of states with ._init_traj_file attribute pointing to source
        """
        states = []
        params = SystemParameters()
        params.box_size = self.initial_state.box_size
        
        for chain_id in range(n_chains):
            if stage_idx == 0:
                # First stage: initialize from scratch
                source = self.init_mode if self.init_mode in ["ideal", "random"] else "random"
                builder = SystemBuilder(
                    params=params,
                    sampler_sequence=self.initial_state.sampler_sequence,
                    current_sampler=self.stages[stage_idx]['name'],
                    source=source
                )
                traj_file = None
                
            else:
                # Later stages: inherit from random previous chain
                chain_dirs = [d for d in os.listdir(prev_stage_output) 
                             if d.startswith('chain_') and 
                             os.path.isdir(os.path.join(prev_stage_output, d))]
                
                selected_chain = np.random.choice(chain_dirs)
                traj_file = os.path.join(prev_stage_output, selected_chain, "trajectory.h5")
                
                builder = SystemBuilder(
                    params=params,
                    sampler_sequence=self.initial_state.sampler_sequence,
                    current_sampler=self.stages[stage_idx]['name'],
                    source="trajectory",
                    trajectory_file=traj_file,
                    frame=-1  # Last frame
                )
            
            state = builder.build()
            state._init_traj_file = traj_file  # Remember source for sigma initialization
            states.append(state)
        
        return states

    def _initialize_sigma_for_stage(
        self,
        states: List[SystemState],
        stage_idx: int,
        prev_stage_output: Optional[str]
    ) -> None:
        """
        Initialize sigma values and attach sigma_prior to each state.
        
        Sigma Initialization Strategy:
        ------------------------------
        Stage 0:
            - Sample from prior (uniform/gamma/inv_gamma)
        
        Stage N>0 (for each chain):
            1. Try to load sigma from state._init_traj_file (last frame)
            2. Load GMM from same chain directory (if exists)
            3. Attach GMM-based prior (or fallback to parametric prior)
            4. If loading failed, sample from prior
        
        This ensures sigma continuity: each chain inherits sigma from its
        parent chain in the previous stage.
        """
        print(f"  Initializing sigma for stage {stage_idx+1}...")

        pair_types = list(get_default_sigma_ranges().keys())
        sigma_ranges = get_default_sigma_ranges()

        for chain_id, state in enumerate(states):
            state.sigma_range = sigma_ranges
            
            # Initialize tracking variables
            gmm_file = None
            loaded_from_parent = False
            source_description = self.prior_type

            # ===================================================================
            # STAGE N > 0: Try to inherit sigma and GMM from parent chain
            # ===================================================================
            if stage_idx > 0 and getattr(state, "_init_traj_file", None):
                parent_traj = state._init_traj_file
                parent_chain_dir = os.path.dirname(parent_traj)
                
                # Check for GMM from parent chain
                candidate_gmm = os.path.join(parent_chain_dir, "gmm_posterior.json")
                if os.path.exists(candidate_gmm):
                    gmm_file = candidate_gmm
                    source_description = "GMM"

                # Load sigma from parent chain's last frame
                parent_sigma, success = self._load_sigma_from_trajectory(parent_traj)
                
                if success:
                    # Validate and clip to bounds
                    cleaned_sigma = {}
                    for pt in pair_types:
                        if pt in parent_sigma and np.isfinite(parent_sigma[pt]):
                            low, high = sigma_ranges[pt]
                            cleaned_sigma[pt] = float(np.clip(parent_sigma[pt], low, high))
                    
                    # Only use if all pair types present
                    if len(cleaned_sigma) == len(pair_types):
                        state.sigma = cleaned_sigma
                        loaded_from_parent = True
                        source_description = "parent_last_frame"

            # ===================================================================
            # CREATE PRIOR (with GMM if available from parent)
            # ===================================================================
            state.sigma_prior = create_sigma_prior(
                pair_types=pair_types,
                sigma_ranges=sigma_ranges,
                gmm_file=gmm_file,
                prior_type=self.prior_type
            )

            # ===================================================================
            # FALLBACK: Sample from prior if not inherited
            # ===================================================================
            if not loaded_from_parent:
                rng = np.random.default_rng(self.base_seed + stage_idx * 1000 + chain_id)
                
                try:
                    state.sigma = state.sigma_prior.initialize_sigma(rng)
                except Exception:
                    # Ultimate fallback
                    state.sigma = initialize_sigma_dict(
                        pair_types=pair_types,
                        sigma_ranges=sigma_ranges,
                        rng=rng,
                        prior_type=self.prior_type
                    )
                
                source_description = "sampled_from_prior"

            # ===================================================================
            # LOGGING
            # ===================================================================
            if chain_id < 3:
                sigma_str = ", ".join(f"{k}={state.sigma[k]:.3f}" for k in pair_types)
                print(f"    Chain {chain_id+1}: {sigma_str} (source: {source_description})")

        if len(states) > 3:
            print(f"    ... and {len(states)-3} more chains")

    # =========================================================================
    # MCMC EXECUTION
    # =========================================================================
    
    def _run_single_chain(self, args):
        """Run a single MCMC chain (called by parallel executor)"""
        state, stage, chain_id, stage_output, n_steps, chain_seed = args
        
        # Ensure reproducibility
        self._seed_everything(chain_seed)

        # Setup output directory
        chain_output = os.path.join(stage_output, f"chain_{chain_id}")
        os.makedirs(chain_output, exist_ok=True)

        # Run sampler
        final_state, trajectory_file = stage['function'](
            state=state,
            n_steps=n_steps,
            output_dir=chain_output,
            **stage['kwargs']
        )

        return {
            'chain_id': chain_id,
            'final_state': final_state,
            'trajectory_file': trajectory_file,
            'final_sigma': {k: float(v) for k, v in final_state.sigma.items()}
        }

    def _run_parallel_chains(self, states, stage, stage_output, n_chains):
        """Run multiple MCMC chains in parallel"""
        print(f"  Running {n_chains} parallel chains...")
        
        # Prepare arguments for parallel execution
        args_list = [
            (states[i], stage, i+1, stage_output, stage['n_steps'], 
             self.base_seed + 1000 * len(self.stages) + i + 1)
            for i in range(n_chains)
        ]
        
        # Execute in parallel
        max_workers = min(n_chains, mp.cpu_count())
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self._run_single_chain, args_list))

    # =========================================================================
    # GMM FITTING & ANALYSIS
    # =========================================================================
    
    def _fit_robust_gmm(self, samples: np.ndarray) -> Dict:
        """
        Fit GMM with 1-3 components using BIC.
        
        Handles edge cases:
        - Low variance: fallback to single Gaussian
        - Small sample size: limit number of components
        """
        samples = samples.reshape(-1, 1)
        n_samples = len(samples)
        
        # Check for sufficient variation
        unique_samples = np.unique(samples)
        sample_std = np.std(samples)
        
        if len(unique_samples) < 3 or sample_std < 1e-6:
            # Degenerate case: single Gaussian
            mean_val = float(np.mean(samples))
            std_val = max(0.1, float(sample_std))
            return {
                'n_components': 1,
                'weights': [1.0],
                'means': [[mean_val]],
                'covariances': [[std_val**2]]
            }
        
        # Fit GMM with adaptive component count
        max_components = min(3, max(1, n_samples // 10))
        best_gmm = None
        best_bic = np.inf
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            
            for n_comp in range(1, max_components + 1):
                try:
                    gmm = GaussianMixture(
                        n_components=n_comp, 
                        random_state=42, 
                        max_iter=100, 
                        tol=1e-4
                    )
                    gmm.fit(samples)
                    
                    # Check if all components are used
                    if len(np.unique(gmm.predict(samples))) >= n_comp:
                        bic = gmm.bic(samples)
                        if bic < best_bic:
                            best_bic = bic
                            best_gmm = gmm
                            
                except Exception:
                    continue
        
        # Fallback if fitting failed
        if best_gmm is None:
            mean_val = float(np.mean(samples))
            std_val = max(0.1, float(np.std(samples)))
            return {
                'n_components': 1,
                'weights': [1.0],
                'means': [[mean_val]],
                'covariances': [[std_val**2]]
            }
        
        return {
            'n_components': best_gmm.n_components,
            'weights': best_gmm.weights_.tolist(),
            'means': best_gmm.means_.tolist(),
            'covariances': best_gmm.covariances_.tolist()
        }

    def _save_chain_gmms(self, stage_output: str, chain_dirs: List[str]):
        """
        Save GMM posterior for each chain.
        
        Purpose: These GMMs become the prior for the next stage
        """
        print(f"  Saving chain-specific GMMs...")
        
        for chain_dir in chain_dirs:
            h5_file = os.path.join(stage_output, chain_dir, "trajectory.h5")
            gmm_file = os.path.join(stage_output, chain_dir, "gmm_posterior.json")
            
            if not os.path.exists(h5_file):
                continue
            
            try:
                # Extract sigma samples (post burn-in)
                sigma_samples = self._extract_sigma_samples(h5_file, burn_in=0.5)
                gmm_params = {}
                
                # Fit GMM for each pair type
                for pair_type, samples in sigma_samples.items():
                    if len(samples) >= 5:
                        gmm_params[pair_type] = self._fit_robust_gmm(samples)
                
                # Ensure all pair types have entries (fallback to wide uniform)
                for pair_type in ['AA', 'AB', 'BC']:
                    if pair_type not in gmm_params:
                        low, high = DEFAULT_SIGMA_RANGES.get(pair_type, (0.5, 5.0))
                        mid = (low + high) / 2
                        spread = (high - low) / 6
                        gmm_params[pair_type] = {
                            'n_components': 1,
                            'weights': [1.0],
                            'means': [[mid]],
                            'covariances': [[spread**2]]
                        }
                
                # Save to disk
                with open(gmm_file, 'w') as f:
                    json.dump(gmm_params, f, indent=2)
                
                print(f"    Saved GMM for {chain_dir}")
                
            except Exception as e:
                print(f"    Warning: Failed to save GMM for {chain_dir}: {e}")

    def _fit_gmm_posterior(self, stage_output: str, stage_results: List[Dict]):
        """
        Fit GMM posterior - wrapper for compatibility.
        Handles both replica exchange and parallel chains.
        """
        # Find chain directories
        chain_dirs = sorted([
            d for d in os.listdir(stage_output) 
            if d.startswith('chain_') and 
            os.path.isdir(os.path.join(stage_output, d))
        ])
        
        if chain_dirs:
            self._save_chain_gmms(stage_output, chain_dirs)

    def _run_stage_diagnostics(self, stage_output: str, stage_results: List[Dict]):
        """Run convergence diagnostics - wrapper for compatibility"""
        self._run_analysis(stage_output, os.path.basename(stage_output))

    def _run_analysis(self, stage_output, stage_name):
        """
        Run post-stage analysis:
        1. Convert H5 to RMF3
        2. Fit GMMs for next stage
        3. Compute convergence diagnostics (R-hat)
        """
        print(f"  Running convergence analysis for {stage_name}...")
        
        try:
            # Find all chain directories
            chain_dirs = sorted([
                d for d in os.listdir(stage_output) 
                if d.startswith('chain_') and 
                os.path.isdir(os.path.join(stage_output, d))
            ])
            
            print(f"  Found {len(chain_dirs)} chains to analyze")
            
            # Convert H5 trajectories to RMF3 (for IMP visualization)
            rmf_files = []
            for chain_dir in chain_dirs:
                h5_file = os.path.join(stage_output, chain_dir, "trajectory.h5")
                rmf_file = os.path.join(stage_output, chain_dir, "trajectory.rmf3")
                
                if os.path.exists(h5_file):
                    try:
                        from analysis import h5_to_rmf3
                        h5_to_rmf3.convert_hdf5_to_rmf3(h5_file, rmf_file)
                        rmf_files.append(rmf_file)
                    except Exception as e:
                        print(f"      Warning: RMF3 conversion failed for {chain_dir}: {e}")
            
            # Run convergence diagnostics (R-hat, effective sample size, etc.)
            if len(chain_dirs) > 1:
                analysis_results = run_mcmc_diagnostics(
                    stage_output, 
                    stage_name, 
                    chain_dirs, 
                    rmf_files
                )
                
                # Save analysis results
                with open(os.path.join(stage_output, "analysis_results.json"), 'w') as f:
                    json.dump(analysis_results, f, indent=2)
                
                # Print R-hat summary
                if 'rhat' in analysis_results:
                    print(f"\n  R-hat values for {stage_name}:")
                    for param, rhat_value in analysis_results['rhat'].items():
                        status = "✓" if rhat_value < 1.1 else "✗"
                        print(f"    {status} {param}: {rhat_value:.3f}")
                
                return analysis_results
            
        except Exception as e:
            print(f"  Warning: Analysis failed: {e}")
            return {}

    # =========================================================================
    # MAIN PIPELINE EXECUTION
    # =========================================================================
    
    def run(
        self,
        output_base: str = "output/pipeline",
        n_chains: int = 4,
        use_replica_exchange: bool = False
    ) -> Dict[str, Any]:
        """
        Run multi-stage pipeline with parallel chains OR replica exchange.
        
        Args:
            output_base: Base directory for outputs
            n_chains: Number of parallel chains (or replicas if use_replica_exchange=True)
            use_replica_exchange: If True, use replica exchange instead of independent chains
        """
        os.makedirs(output_base, exist_ok=True)
        all_results = []
        prev_stage_output = None
        
        for i, stage in enumerate(self.stages):
            print(f"\n{'='*70}")
            print(f"=== Stage {i+1}/{len(self.stages)}: {stage['name']} ===")
            print(f"{'='*70}")
            
            stage_output = os.path.join(output_base, f"stage_{i+1}_{stage['name']}")
            os.makedirs(stage_output, exist_ok=True)
            
            # Create initial states for this stage
            starting_states = self._create_initial_states(i, n_chains, prev_stage_output)
            
            # Initialize sigma and sigma_prior for all states
            self._initialize_sigma_for_stage(starting_states, i, prev_stage_output)
            
            # ============================================================
            # KEY DECISION: Replica Exchange vs Parallel Chains
            # ============================================================
            if use_replica_exchange:
                print(f"  Running REPLICA EXCHANGE with {n_chains} temperature replicas")
                
                # Use first state as template (all start from same config)
                template_state = starting_states[0]
                
                # Import the replica exchange function
                from samplers.base import run_replica_exchange_mcmc
                
                # Extract temperature range from stage kwargs
                temp_min = stage['kwargs'].get('temp_end', 1.0)
                temp_max = stage['kwargs'].get('temp_start', 10.0)
                
                # Run replica exchange (returns best state + all trajectories)
                best_state, traj_files = run_replica_exchange_mcmc(
                    state=template_state,
                    score_fn=stage['score_fn'],
                    propose_fn_dict=stage['propose_fns'],
                    move_probs=stage['move_probs'],
                    n_steps=stage['n_steps'],
                    save_freq=stage['kwargs'].get('save_freq', 100),
                    output_dir=stage_output,
                    n_replicas=n_chains,
                    temp_min=temp_min,
                    temp_max=temp_max,
                    swap_freq=10,
                    equilibration_steps=stage['kwargs'].get('equilibration_steps', 500),
                    debug=False
                )
                
                # Create chain_0 directory with best replica for next stage
                chain_0_dir = os.path.join(stage_output, "chain_0")
                os.makedirs(chain_0_dir, exist_ok=True)
                
                # Copy lowest-T trajectory to chain_0
                import shutil
                if os.path.exists(traj_files[0]):
                    shutil.copy2(traj_files[0], os.path.join(chain_0_dir, "trajectory.h5"))
                
                # Convert to results format
                stage_results = [{
                    'chain_id': 0,
                    'final_state': best_state,
                    'trajectory_file': os.path.join(chain_0_dir, "trajectory.h5"),
                    'final_sigma': {k: float(v) for k, v in best_state.sigma.items()},
                    'all_trajectories': traj_files
                }]
                
            else:
                # Original parallel chain approach
                stage_results = self._run_parallel_chains(
                    starting_states, stage, stage_output, n_chains
                )
            
            # ============================================================
            # Post-stage analysis (same for both methods)
            # ============================================================
            all_results.append({
                'stage_name': stage['name'],
                'stage_output': stage_output,
                'results': stage_results
            })
            
            # Fit GMM to sigma posterior (for next stage prior)
            self._fit_gmm_posterior(stage_output, stage_results)
            
            # Run convergence diagnostics
            self._run_stage_diagnostics(stage_output, stage_results)
            
            prev_stage_output = stage_output
        
        return {'stages': all_results}