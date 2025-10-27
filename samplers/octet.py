from typing import Optional, Tuple
from core.state import SystemState
from samplers.base import run_mcmc_sampling
from scoring.pair_score import PairNLL
from scoring.tetramer_score import TetramerNLL
from scoring.octet_score import OctetNLL
from scoring.exvol_score import ExvolNLL
from core.movers import (
    propose_particle_move, 
    propose_sigma_move, 
    propose_tetramer_move,
    propose_octet_move
)

def neg_log_posterior(
    state: SystemState,
    prior_penalty: float = 0.0,   # Ignored input; prior is computed here
    excluded_pairs: Optional[set] = None
) -> Tuple[float, float, float, float, float, float]:
    """
    -log posterior = ExVol NLL + Pair NLL + Tetramer NLL + Octet NLL + (-log prior).

    The posterior includes:
    - Excluded volume constraints (soft repulsion)
    - Pairwise distance likelihoods (AA, AB, BC)
    - Tetramer formation scores (AB + 2×BC bonds)
    - Octet formation scores (inter-tetramer AA distances)
    - Prior on sigma parameters

    Returns:
        (total_score, exclusion_score, pair_score, tetramer_score, octet_score, prior_penalty)
    """
    # Excluded volume score
    exs = ExvolNLL(state.positions, kappa=100.0)
    exclusion_score = exs.compute_score()

    # Pair score (pairwise distance likelihood)
    ps = PairNLL(state.positions, state.sigma)
    pair_score = ps.compute_score()

    # Tetramer score (ABCC unit formation)
    ts = TetramerNLL(state)
    tetramer_score = ts.compute_score()

    # Octet score (tetramer pairing via A-A distances)
    os = OctetNLL(state)
    octet_score = os.compute_score()

    # Prior on sigma
    if hasattr(state, "sigma_prior") and state.sigma_prior is not None:
        prior_penalty = -state.sigma_prior.log_prior(state.sigma)
    else:
        prior_penalty = 0.0

    total_score = exclusion_score + pair_score + tetramer_score + octet_score + prior_penalty
    
    return total_score, exclusion_score, pair_score, tetramer_score, octet_score, prior_penalty

def run_octet_sampling(
    state: SystemState,
    n_steps: int = 1000,
    output_dir: str = "output/octet_sampler",
    **kwargs
) -> Tuple[SystemState, str]:
    """
    Run octet-level MCMC sampling with hierarchical move mix:
      - 40% octet rigid-body moves (translation/rotation of 8-particle units)
      - 30% tetramer rigid-body moves (translation/rotation of 4-particle units)
      - 20% single-particle moves (local adjustments)
      - 10% sigma parameter moves (likelihood width updates)

    This sampler is designed for systems where octets (pairs of adjacent tetramers)
    are the primary structural units. The move hierarchy allows:
    1. Large-scale rearrangements via octet moves
    2. Mid-scale adjustments via tetramer moves
    3. Fine-tuning via particle moves
    4. Uncertainty quantification via sigma moves

    Args:
        state: Initial SystemState with positions and sigma values
        n_steps: Number of MCMC steps to run
        output_dir: Directory for trajectory and diagnostics
        **kwargs: Additional arguments passed to run_mcmc_sampling

    Returns:
        (final_state, trajectory_file_path)
    """
    propose_fns = {
        "octet": propose_octet_move,
        "tetramer": propose_tetramer_move,
        "position": propose_particle_move,
        "sigma": propose_sigma_move,
    }
    
    move_probs = {
        "octet": 0.40,      # Largest structural units
        "tetramer": 0.30,   # Mid-level structural units
        "position": 0.20,   # Local fine-tuning
        "sigma": 0.10       # Parameter updates
    }

    return run_mcmc_sampling(
        state=state,
        score_fn=neg_log_posterior,
        propose_fn_dict=propose_fns,
        move_probs=move_probs,
        n_steps=n_steps,
        output_dir=output_dir,
        **kwargs,
    )

def get_octets(state) -> Tuple[list, list]:
    """
    Convenience passthrough to the octet finder in core.movers.
    
    Returns:
        (octets, tetramers) where:
        - octets: List of (tetramer1, tetramer2) pairs
        - tetramers: All tetramers found
    """
    from core.movers import get_octets as _get_octets
    return _get_octets(state)

def get_tetramers(state) -> list:
    """Convenience passthrough to the tetramer finder in core.movers."""
    from core.movers import get_tetramers as _get_tetramers
    return _get_tetramers(state)