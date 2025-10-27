from typing import Optional, Tuple
import os
import mrcfile
import numpy as np
from core.state import SystemState
from samplers.base import run_mcmc_sampling
from scoring.pair_score import PairNLL
from scoring.tetramer_score import TetramerNLL
from scoring.octet_score import OctetNLL
from scoring.exvol_score import ExvolNLL
from scoring.full_score import FullNLL
from core.movers import (
    propose_particle_move, 
    propose_sigma_move, 
    propose_tetramer_move,
    propose_octet_move
)

# Global EM map path (set this before running)
EM_MAP_FILE = None
EM_RESOLUTION = 50.0
EM_BACKEND = 'cpu'

def set_em_map(map_file: str, resolution: float = 50.0, backend: str = 'cpu'):
    """
    Set the EM map file to use for scoring.
    Call this before running full sampling.
    """
    global EM_MAP_FILE, EM_RESOLUTION, EM_BACKEND
    if not os.path.exists(map_file):
        raise FileNotFoundError(f"EM map file not found: {map_file}")
    EM_MAP_FILE = map_file
    EM_RESOLUTION = resolution
    EM_BACKEND = backend
    print(f"EM map set to: {map_file} (resolution={resolution}Å, backend={backend})")

def center_state_to_density(state: SystemState) -> SystemState:
    """
    Center the state positions to the density map COM.
    Should be called ONCE before starting sampling.
    
    Args:
        state: SystemState with positions to center
        
    Returns:
        SystemState with centered positions
    """
    if EM_MAP_FILE is None:
        raise ValueError("EM map not set. Call set_em_map() first.")
    
    # Load density map temporarily to get bounds
    density_map = mrcfile.open(EM_MAP_FILE, permissive=True)
    
    # Calculate bins
    nx, ny, nz = density_map.header.nx, density_map.header.ny, density_map.header.nz
    vx, vy, vz = density_map.voxel_size.x, density_map.voxel_size.y, density_map.voxel_size.z
    
    x_extent = nx * vx / 2
    y_extent = ny * vy / 2
    z_extent = nz * vz / 2
    
    box_min = np.array([-x_extent, -y_extent, -z_extent])
    box_max = np.array([x_extent, y_extent, z_extent])
    
    # Center positions
    centered_positions = FullNLL.center_particles_to_density_com(
        state.positions,
        density_map,
        box_min,
        box_max
    )
    
    density_map.close()
    
    # Update state
    state.positions = centered_positions
    return state

def neg_log_posterior(
    state: SystemState,
    prior_penalty: float = 0.0,   # Ignored input; prior is computed here
    excluded_pairs: Optional[set] = None
) -> Tuple[float, float, float, float, float, float, float]:
    """
    -log posterior = ExVol NLL + Pair NLL + Tetramer NLL + Octet NLL - CCC + (-log prior).

    The posterior includes:
    - Excluded volume constraints (soft repulsion)
    - Pairwise distance likelihoods (AA, AB, BC)
    - Tetramer formation scores (AB + 2×BC bonds)
    - Octet formation scores (inter-tetramer AA distances)
    - EM density fit (negative cross-correlation)
    - Prior on sigma parameters

    Returns:
        (total_score, exclusion_score, pair_score, tetramer_score, octet_score, em_score, prior_penalty)
    """
    if EM_MAP_FILE is None:
        raise ValueError("EM map not set. Call set_em_map() before running sampling.")
    
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

    # EM density score (negative CCC for minimization)
    em_scorer = FullNLL(state, em_map_file=EM_MAP_FILE, 
                     resolution=EM_RESOLUTION, backend=EM_BACKEND)
    em_score = em_scorer.compute_score()  # Returns -CCC

    # Prior on sigma
    if hasattr(state, "sigma_prior") and state.sigma_prior is not None:
        prior_penalty = -state.sigma_prior.log_prior(state.sigma)
    else:
        prior_penalty = 0.0

    # Total: minimize all components
    total_score = exclusion_score + pair_score + tetramer_score + octet_score + em_score + prior_penalty
    
    return total_score, exclusion_score, pair_score, tetramer_score, octet_score, em_score, prior_penalty

def run_full_sampling(
    state: SystemState,
    n_steps: int = 1000,
    output_dir: str = "output/full_sampler",
    em_map_file: Optional[str] = None,
    em_resolution: float = 50.0,
    em_backend: str = 'cpu',
    center_to_density: bool = True,
    **kwargs
) -> Tuple[SystemState, str]:
    """
    Run full-system MCMC sampling with EM density restraint.
    
    Move hierarchy:
      - 40% tetramer rigid-body moves (translation/rotation of 4-particle units)
      - 30% single-particle moves (local adjustments)
      - 20% octet rigid-body moves (translation/rotation of 8-particle units)
      - 10% sigma parameter moves (likelihood width updates)

    Args:
        state: Initial SystemState with positions and sigma values
        n_steps: Number of MCMC steps to run
        output_dir: Directory for trajectory and diagnostics
        em_map_file: Path to EM density map (if None, uses global EM_MAP_FILE)
        em_resolution: Resolution for Gaussian blurring (Angstroms)
        em_backend: 'cpu' or 'gpu' for computation
        center_to_density: If True, center particles to density COM before sampling
        **kwargs: Additional arguments passed to run_mcmc_sampling

    Returns:
        (final_state, trajectory_file_path)
    """
    # Set EM map parameters
    if em_map_file is not None:
        set_em_map(em_map_file, em_resolution, em_backend)
    elif EM_MAP_FILE is None:
        raise ValueError("EM map not specified. Provide em_map_file or call set_em_map().")
    
    # Center particles to density COM (done once at start)
    if center_to_density:
        print("Centering particles to density map COM...")
        state = center_state_to_density(state)
    
    # Define move proposals
    propose_fns = {
        "octet": propose_octet_move,
        "tetramer": propose_tetramer_move,
        "position": propose_particle_move,
        "sigma": propose_sigma_move,
    }
    
    move_probs = {
        "octet": 0.20,      # Large structural units
        "tetramer": 0.40,   # Mid-level structural units
        "position": 0.30,   # Local fine-tuning
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