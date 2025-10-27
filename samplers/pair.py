from typing import Optional, Tuple
from core.state import SystemState
from samplers.base import run_mcmc_sampling
from scoring.pair_score import PairNLL
from scoring.exvol_score import ExvolNLL
from core.movers import propose_particle_move, propose_sigma_move

def neg_log_posterior(
    state: SystemState, 
    prior_penalty: float = 0.0,  # Accept but ignore - we compute it internally
    excluded_pairs: Optional[set] = None
) -> Tuple[float, float, float, float]:
    """
    Calculate negative log posterior for pair sampler.
    
    Posterior = Likelihood × Prior
    Log-posterior = log(likelihood) + log(prior)
    Negative log-posterior = NLL + negative_log_prior
    
    Returns:
        Tuple of (total_score, exclusion_score, pair_score, prior_penalty)
    """
    # Calculate excluded volume contribution (likelihood term)
    exs = ExvolNLL(state.positions, kappa=100.0)
    exclusion_score = exs.compute_score()
    
    # Calculate pairwise score (likelihood term)
    ps = PairNLL(state.positions, state.sigma)
    pair_score = ps.compute_score()

    # Calculate prior penalty (ACTUALLY COMPUTE IT!)
    if hasattr(state, 'sigma_prior') and state.sigma_prior is not None:
        log_prior = state.sigma_prior.log_prior(state.sigma)
        prior_penalty = -log_prior  # Negative because we're minimizing negative log posterior
    else:
        prior_penalty = 0.0  # No prior (improper/flat)

    # Total negative log posterior = sum of all NLL terms + prior penalty
    total_score = exclusion_score + pair_score + prior_penalty
    
    return total_score, exclusion_score, pair_score, prior_penalty

def run_pair_sampling(
    state: SystemState,
    n_steps: int = 1000,
    output_dir: str = "output/pair_sampler",
    **kwargs
) -> Tuple[SystemState, str]:
    """Run pair-level MCMC sampling"""
    # Define proposal functions - direct from core/movers.py
    propose_fns = {
        'position': propose_particle_move,
        'sigma': propose_sigma_move
    }
    
    # Define move probabilities
    move_probs = {
        'position': 0.9,
        'sigma': 0.1
    }
    
    # Run MCMC - score_fn now computes prior internally
    # Note: We pass the function directly, not a lambda
    return run_mcmc_sampling(
        state=state,
        score_fn=neg_log_posterior,  # Pass function directly
        propose_fn_dict=propose_fns,
        move_probs=move_probs,
        n_steps=n_steps,
        output_dir=output_dir,
        **kwargs
    )