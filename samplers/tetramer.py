from typing import Optional, Tuple
from core.state import SystemState
from samplers.base import run_mcmc_sampling
from scoring.pair_score import PairNLL
from scoring.tetramer_score import TetramerNLL
from scoring.exvol_score import ExvolNLL
from core.movers import propose_particle_move, propose_sigma_move, propose_tetramer_move

def neg_log_posterior(
    state: SystemState,
    prior_penalty: float = 0.0,   # Ignored input; prior is computed here
    excluded_pairs: Optional[set] = None
) -> Tuple[float, float, float, float, float]:
    """
    -log posterior = ExVol NLL + Pair NLL + Tetramer NLL + (-log prior).

    Returns:
        (total_score, exclusion_score, pair_score, tetramer_score, prior_penalty)
    """
    # Excluded volume score
    exs = ExvolNLL(state.positions, kappa=100.0)
    exclusion_score = exs.compute_score()

    # Pair score
    ps = PairNLL(state.positions, state.sigma)
    pair_score = ps.compute_score()

    # Tetramer score
    ts = TetramerNLL(state)
    tetramer_score = ts.compute_score()

    # Prior on sigma
    if hasattr(state, "sigma_prior") and state.sigma_prior is not None:
        prior_penalty = -state.sigma_prior.log_prior(state.sigma)
    else:
        prior_penalty = 0.0

    total_score = exclusion_score + pair_score + tetramer_score + prior_penalty
    return total_score, exclusion_score, pair_score, tetramer_score, prior_penalty

def run_tetramer_sampling(
    state: SystemState,
    n_steps: int = 1000,
    output_dir: str = "output/tetramer_sampler",
    **kwargs
) -> Tuple[SystemState, str]:
    """
    Run tetramer-level MCMC sampling with move mix:
      - 60% tetramer rigid-body moves
      - 30% single-particle moves
      - 10% sigma parameter moves
    """
    propose_fns = {
        "tetramer": propose_tetramer_move,
        "position": propose_particle_move,
        "sigma": propose_sigma_move,
    }
    move_probs = {"tetramer": 0.50, "position": 0.40, "sigma": 0.10}

    return run_mcmc_sampling(
        state=state,
        score_fn=neg_log_posterior,
        propose_fn_dict=propose_fns,
        move_probs=move_probs,
        n_steps=n_steps,
        output_dir=output_dir,
        **kwargs,
    )

def get_tetramers(state) -> list:
    """Convenience passthrough to the tetramer finder in core.movers."""
    from core.movers import get_tetramers as _get_tetramers
    return _get_tetramers(state)