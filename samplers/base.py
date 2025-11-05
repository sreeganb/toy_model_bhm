# samplers/base.py
import numpy as np
import os
import h5py
from typing import Dict, Tuple, Any, Optional, Callable, List
from core.state import SystemState
from core.io_utils import save_state_to_disk
from pathlib import Path

def run_replica_exchange_mcmc(
    state: SystemState,
    score_fn: Callable,
    propose_fn_dict: Dict[str, Callable],
    move_probs: Dict[str, float],
    n_steps: int = 1000,
    save_freq: int = 100,
    output_dir: str = "output",
    n_replicas: int = 4,
    temp_min: float = 1.0,
    temp_max: float = 10.0,
    swap_freq: int = 10,
    equilibration_steps: int = 500,
    debug: bool = False
) -> Tuple[SystemState, List[str]]:
    """
    Replica Exchange MCMC - multiple temperatures, periodic swaps.
    
    Implements proper parallel tempering with:
    - Geometric temperature ladder
    - Nearest-neighbor swaps only
    - Alternating even/odd pair swaps
    - Correct detailed balance
    
    Returns:
        Best state (from lowest T) and list of all trajectory files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Temperature ladder (geometric spacing for optimal acceptance)
    temps = np.exp(np.linspace(np.log(temp_min), np.log(temp_max), n_replicas))
    
    print(f"\n{'='*70}")
    print(f"REPLICA EXCHANGE: {n_replicas} replicas, T ∈ [{temp_min:.1f}, {temp_max:.1f}]")
    print(f"Temps: {', '.join([f'{t:.2f}' for t in temps])}")
    
    # Calculate expected swap acceptance between adjacent temps
    if n_replicas > 1:
        beta_ratio = temps[0] / temps[1]  # Should be close to 1
        print(f"Temperature ratio (T_i/T_i+1): {beta_ratio:.3f}")
    print(f"{'='*70}\n")
    
    # Initialize replicas
    sigma_prior = getattr(state, 'sigma_prior', None)
    if sigma_prior is None:
        raise ValueError("state.sigma_prior not found")
    
    replicas = []
    for i, temp in enumerate(temps):
        rep_state = state.copy()
        rep_state.sigma_prior = sigma_prior
        score, *components = score_fn(rep_state, 0.0)
        traj_file = os.path.join(output_dir, f"replica_{i}_T{temp:.2f}.h5")
        if os.path.exists(traj_file):
            os.remove(traj_file)
        
        replicas.append({
            'state': rep_state,
            'temp': temp,
            'temp_idx': i,  # Track original temperature index
            'score': score,
            'components': components,
            'traj_file': traj_file,
            'accepts': {m: 0 for m in propose_fn_dict},
            'attempts': {m: 0 for m in propose_fn_dict},
            'swap_accepts': 0,
            'swap_attempts': 0
        })
    
    # Setup moves
    move_types = list(move_probs.keys())
    move_weights = np.array([move_probs[m] for m in move_types])
    move_weights /= move_weights.sum()
    
    best_state = replicas[0]['state'].copy()
    best_score = replicas[0]['score']
    
    # Main loop
    swap_direction = 0  # Alternates between 0 (even pairs) and 1 (odd pairs)
    
    for step in range(1, n_steps + 1):
        # Phase 1: Independent MCMC moves at each temperature
        for rep in replicas:
            move_type = np.random.choice(move_types, p=move_weights)
            rep['attempts'][move_type] += 1
            
            proposed = rep['state'].copy()
            propose_fn_dict[move_type](proposed)
            proposed.sigma_prior = sigma_prior
            
            prop_score, *prop_comp = score_fn(proposed, 0.0)
            delta = prop_score - rep['score']
            
            # Metropolis acceptance with temperature
            if delta < 0 or np.random.random() < np.exp(-delta / rep['temp']):
                rep['state'] = proposed
                rep['score'] = prop_score
                rep['components'] = prop_comp
                rep['accepts'][move_type] += 1
                
                # Track best from LOWEST temperature (index 0)
                if rep['temp_idx'] == 0 and prop_score < best_score:
                    best_score = prop_score
                    best_state = proposed.copy()
        
        # Phase 2: Replica swaps (every swap_freq steps, after equilibration)
        # Use alternating even/odd pairs for better mixing
        if step % swap_freq == 0 and step > equilibration_steps:
            # Attempt swaps for pairs: (0,1), (2,3), ... or (1,2), (3,4), ...
            for i in range(swap_direction, n_replicas - 1, 2):
                rep_i = replicas[i]
                rep_j = replicas[i + 1]
                
                rep_i['swap_attempts'] += 1
                rep_j['swap_attempts'] += 1
                
                # Detailed balance: P(swap) = min(1, exp((β_i - β_j)(E_j - E_i)))
                # This ensures detailed balance in configuration space
                beta_i = 1.0 / rep_i['temp']
                beta_j = 1.0 / rep_j['temp']
                
                # Energy difference (note: higher temp has higher index in our ladder)
                E_i = rep_i['score']
                E_j = rep_j['score']
                
                # Log acceptance probability
                # P_swap = exp[(β_i - β_j)(E_j - E_i)]
                # Since β_i > β_j (T_i < T_j), we want E_i < E_j for high acceptance
                log_accept = (beta_i - beta_j) * (E_j - E_i)
                
                if log_accept >= 0 or np.random.random() < np.exp(log_accept):
                    # Swap configurations (keep temperature indices and files!)
                    rep_i['state'], rep_j['state'] = rep_j['state'], rep_i['state']
                    rep_i['score'], rep_j['score'] = rep_j['score'], rep_i['score']
                    rep_i['components'], rep_j['components'] = rep_j['components'], rep_i['components']
                    
                    rep_i['swap_accepts'] += 1
                    rep_j['swap_accepts'] += 1
            
            # Alternate swap direction for next swap attempt
            swap_direction = 1 - swap_direction
        
        # Phase 3: Save trajectories
        if step % save_freq == 0 or step == n_steps:
            for rep in replicas:
                exvol = rep['components'][0] if len(rep['components']) > 0 else 0.0
                pair = rep['components'][1] if len(rep['components']) > 1 else 0.0
                tet = rep['components'][2] if len(rep['components']) > 2 else 0.0
                oct = rep['components'][3] if len(rep['components']) > 3 else 0.0
                prior = rep['components'][-1] if len(rep['components']) > 0 else 0.0
                
                save_state_to_disk(
                    step=step,
                    positions=rep['state'].positions,
                    sigmas=rep['state'].sigma,
                    score=rep['score'],
                    prior_score=prior,
                    pair_score=pair,
                    exvol_score=exvol,
                    tet_score=tet,
                    oct_score=oct,
                    types=getattr(rep['state'], 'types', None),
                    bead_numbers=getattr(rep['state'], 'bead_numbers', None),
                    traj_file=rep['traj_file']
                )
            
            # Progress report
            print(f"Step {step}/{n_steps}:")
            for i, rep in enumerate(replicas):
                acc = sum(rep['accepts'].values()) / max(1, sum(rep['attempts'].values()))
                swap = rep['swap_accepts'] / max(1, rep['swap_attempts'])
                print(f"  R{i} T={rep['temp']:5.2f}: Score={rep['score']:8.2f}, "
                      f"Acc={acc:.2%}, Swap={swap:.2%}")
    
    # Final stats
    print(f"\n{'='*70}")
    print("REPLICA EXCHANGE COMPLETE")
    print(f"Best score: {best_score:.2f} (from T={temp_min})")
    print(f"\nFinal states per temperature:")
    for i, rep in enumerate(replicas):
        swap_rate = rep['swap_accepts'] / max(1, rep['swap_attempts'])
        move_rates = ', '.join([f"{m[:3]}={rep['accepts'][m]/max(1,rep['attempts'][m]):.1%}" 
                               for m in move_types])
        print(f"  R{i} (T={rep['temp']:.2f}): Score={rep['score']:.2f}, "
              f"SwapAcc={swap_rate:.2%}, MoveAcc=[{move_rates}]")
    
    # Check for good swap rates (should be 20-40%)
    avg_swap_rate = np.mean([r['swap_accepts'] / max(1, r['swap_attempts']) for r in replicas])
    if avg_swap_rate < 0.15:
        print(f"\n  WARNING: Low swap rate ({avg_swap_rate:.1%}). Consider narrower temperature range.")
    elif avg_swap_rate > 0.5:
        print(f"\n  WARNING: High swap rate ({avg_swap_rate:.1%}). Consider wider temperature range.")
    else:
        print(f"\n Good average swap rate: {avg_swap_rate:.1%}")
    
    print(f"{'='*70}\n")
    
    traj_files = [rep['traj_file'] for rep in replicas]
    return best_state, traj_files

def run_mcmc_sampling(
    state: SystemState,
    score_fn: Callable,
    propose_fn_dict: Dict[str, Callable],
    move_probs: Dict[str, float],
    n_steps: int = 1000,
    save_freq: int = 100,
    output_dir: str = "output",
    temp_start: float = 10.0,
    temp_end: float = 1.0,
    equilibration_steps: int = 500,
    adapt_step_sizes: Optional[Callable] = None,
    debug: bool = False
) -> Tuple[SystemState, str]:
    """
    Generic MCMC sampling function that can be used by all samplers
    
    Args:
        state: Initial system state (with sigma_prior already attached)
        score_fn: Function to calculate score (neg_log_posterior)
                 Should compute prior internally from state.sigma_prior
        propose_fn_dict: Dict of proposal functions for each move type
                        Each function should take (state) and modify it in-place
        move_probs: Dict of probabilities for each move type
        n_steps: Number of MCMC steps to run
        save_freq: How often to save trajectory frames
        output_dir: Directory to save output
        temp_start: Initial temperature
        temp_end: Final temperature
        equilibration_steps: Number of steps at high temperature before annealing
        adapt_step_sizes: Function to adapt step sizes based on acceptance
        debug: Whether to print debug info
    
    Returns:
        Final state and path to trajectory file
    """
    # Setup output directory
    os.makedirs(output_dir, exist_ok=True)
    trajectory_file = os.path.join(output_dir, "trajectory.h5")
    
    if os.path.exists(trajectory_file):
        os.remove(trajectory_file)
    
    # Initialize tracking
    best_state = state.copy()
    best_score = float('inf')
    accepts = {move: 0 for move in propose_fn_dict}
    attempts = {move: 0 for move in propose_fn_dict}
    
    # Verify sigma prior is attached (should be done by pipeline)
    sigma_prior = getattr(state, 'sigma_prior', None)
    if sigma_prior is None:
        raise ValueError("state.sigma_prior not found - pipeline must attach it before sampling")

    # Calculate initial score (score_fn computes prior internally!)
    # The score_fn signature is: score_fn(state, prior_penalty=0.0)
    # But we pass prior_penalty as a dummy - the function computes it internally
    current_score, *score_components = score_fn(state, prior_penalty=0.0)
    
    # Unpack score components (order depends on sampler)
    # pair sampler: (total, exvol, pair, prior)
    # tetramer sampler: (total, exvol, pair, tetramer, prior)
    # octet sampler: (total, exvol, pair, tetramer, octet, prior)
    exvol_score = score_components[0] if len(score_components) > 0 else 0.0
    pair_score = score_components[1] if len(score_components) > 1 else 0.0
    tet_score = score_components[2] if len(score_components) > 2 else 0.0
    oct_score = score_components[3] if len(score_components) > 3 else 0.0
    prior_penalty = score_components[-1] if len(score_components) > 0 else 0.0  # Last component is always prior
    
    # Setup move selection
    move_types = list(move_probs.keys())
    move_weights = np.array([move_probs[m] for m in move_types])
    move_weights /= move_weights.sum()
    
    # Temperature schedule
    annealing_steps = n_steps - equilibration_steps
    temp_decay = (temp_end / temp_start) ** (1.0 / max(1, annealing_steps))
    temp = temp_start
    
    print(f"Starting MCMC sampling for {n_steps} steps...")
    print(f"  - Equilibration: {equilibration_steps} steps at T={temp_start:.2f}")
    print(f"  - Annealing: {annealing_steps} steps from T={temp_start:.2f} to T={temp_end:.2f}")
    print(f"  - Using {'GMM' if sigma_prior.use_gmm else sigma_prior.prior_type} prior for sigma")
    print(f"  - Initial score: {current_score:.2f} (prior penalty: {prior_penalty:.2f})")
    print(f"Output will be saved to: {trajectory_file}")
    
    # Main MCMC loop
    for step in range(1, n_steps + 1):
        # Select and apply move
        move_type = np.random.choice(move_types, p=move_weights)
        attempts[move_type] += 1
        
        # Create proposed state and apply move
        proposed_state = state.copy()
        propose_fn_dict[move_type](proposed_state)
        
        # CRITICAL: Copy sigma_prior reference to proposed state
        # (same prior object applies to both current and proposed)
        proposed_state.sigma_prior = sigma_prior
        
        # Calculate new score (score_fn computes prior internally from state.sigma_prior)
        # The function will evaluate: -log_prior(proposed_state.sigma)
        proposed_score, *prop_components = score_fn(proposed_state, prior_penalty=0.0)
        
        # Unpack proposed scores
        prop_exvol = prop_components[0] if len(prop_components) > 0 else 0.0
        prop_pair = prop_components[1] if len(prop_components) > 1 else 0.0
        prop_tet = prop_components[2] if len(prop_components) > 2 else 0.0
        prop_oct = prop_components[3] if len(prop_components) > 3 else 0.0
        new_prior = prop_components[-1] if len(prop_components) > 0 else 0.0
        
        # Metropolis-Hastings acceptance criterion
        # delta = -log(posterior_proposed) - (-log(posterior_current))
        # Accept if posterior_proposed > posterior_current (i.e., delta < 0)
        delta = proposed_score - current_score
        accept = delta < 0 or np.random.random() < np.exp(-delta / temp)
        
        if debug and step % 10 == 0:
            sigma_str = ', '.join([f'{k}={v:.2f}' for k, v in state.sigma.items()])
            print(f"Step {step}: Move={move_type}, Delta={delta:.2f}, T={temp:.2f}, "
                  f"Prior={prior_penalty:.2f}, Sigma=[{sigma_str}]")
            if move_type == 'sigma':
                prop_sigma_str = ', '.join([f'{k}={v:.2f}' for k, v in proposed_state.sigma.items()])
                print(f"  Proposed sigma: [{prop_sigma_str}], New prior: {new_prior:.2f}")
        
        if accept:
            state = proposed_state
            current_score = proposed_score
            exvol_score, pair_score = prop_exvol, prop_pair
            tet_score, oct_score = prop_tet, prop_oct
            prior_penalty = new_prior
            accepts[move_type] += 1
            
            if current_score < best_score:
                best_score = current_score
                best_state = state.copy()
        
        # Update temperature (simulated annealing)
        if step > equilibration_steps:
            temp = temp_start * (temp_decay ** (step - equilibration_steps))
        else:
            temp = temp_start
        
        # Adaptive step size tuning
        if adapt_step_sizes and step % 100 == 0:
            acceptance_rates = {k: accepts[k] / max(1, attempts[k]) for k in accepts}
            adapt_step_sizes(acceptance_rates)
        
        # Save trajectory to disk
        if step % save_freq == 0 or step == n_steps:
            save_state_to_disk(
                step=step,
                positions=state.positions,
                sigmas=state.sigma,
                score=current_score,
                prior_score=prior_penalty,
                pair_score=pair_score,
                exvol_score=exvol_score,
                tet_score=tet_score,
                oct_score=oct_score,
                types=getattr(state, 'types', None),
                bead_numbers=getattr(state, 'bead_numbers', None),
                traj_file=trajectory_file
            )
            
            accept_rate = sum(accepts.values()) / max(1, sum(attempts.values()))
            sigma_str = ', '.join([f'{k}={v:.2f}' for k, v in state.sigma.items()][:3])
            print(f"Step {step}/{n_steps}: Score={current_score:.2f}, T={temp:.2f}, "
                  f"Accept={accept_rate:.2%}, Sigma=[{sigma_str}]")
    
    # Print final statistics
    print("\n" + "="*60)
    print("Sampling complete!")
    print("="*60)
    print(f"Best score: {best_score:.2f}")
    print(f"Final score: {current_score:.2f}")
    print(f"\nAcceptance rates:")
    for move in move_types:
        rate = accepts[move] / max(1, attempts[move])
        print(f"  {move:>12s}: {rate:>6.2%} ({accepts[move]}/{attempts[move]})")
    
    print(f"\nFinal sigma values:")
    for k, v in state.sigma.items():
        print(f"  {k}: {v:.3f}")
    
    print(f"\nTrajectory saved to: {trajectory_file}")
    print("="*60 + "\n")
    
    return best_state, trajectory_file