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
    propose_octet_move,
    propose_full_move
)

# Global EM scorer (initialized ONCE before sampling)
_EM_SCORER = None
_EM_MAP_FILE = None
_EM_RESOLUTION = 50.0
_EM_BACKEND = 'cpu'

def set_em_map_config(map_file: str, resolution: float = 50.0, backend: str = 'cpu'):
    """
    Set EM map configuration globally (called before creating states).
    The actual scorer is initialized later when we have a state.
    
    Args:
        map_file: Path to EM density map
        resolution: Resolution for Gaussian blurring (Angstroms)
        backend: 'cpu' or 'gpu' for computation
    """
    global _EM_MAP_FILE, _EM_RESOLUTION, _EM_BACKEND
    
    if not os.path.exists(map_file):
        raise FileNotFoundError(f"EM map file not found: {map_file}")
    
    _EM_MAP_FILE = map_file
    _EM_RESOLUTION = resolution
    _EM_BACKEND = backend
    
    print(f"EM map configuration set:")
    print(f"  Map file: {map_file}")
    print(f"  Resolution: {resolution} Å")
    print(f"  Backend: {backend}")

def _initialize_em_scorer(state: SystemState):
    """
    Initialize the EM scorer ONCE when we have a state.
    Called automatically by run_full_sampling.
    """
    global _EM_SCORER, _EM_MAP_FILE, _EM_RESOLUTION, _EM_BACKEND
    
    if _EM_SCORER is not None:
        # Already initialized
        return
    
    if _EM_MAP_FILE is None:
        raise ValueError("EM map not configured. Call set_em_map_config() first.")
    
    print(f"\nInitializing EM scorer (processing experimental map)...")
    _EM_SCORER = FullNLL(
        state, 
        em_map_file=_EM_MAP_FILE,
        resolution=_EM_RESOLUTION, 
        backend=_EM_BACKEND
    )
    print(f"EM scorer initialized\n")

def center_state_to_density(state: SystemState) -> SystemState:
    """
    Center the state positions to the density map COM.
    Should be called ONCE before starting sampling.
    
    Args:
        state: SystemState with positions to center
        
    Returns:
        SystemState with centered positions
    """
    global _EM_MAP_FILE
    
    if _EM_MAP_FILE is None:
        raise ValueError("EM map not configured. Call set_em_map_config() first.")
    
    # Load density map temporarily to get bounds
    density_map = mrcfile.open(_EM_MAP_FILE, permissive=True)
    
    # Calculate bins
    nx, ny, nz = density_map.header.nx, density_map.header.ny, density_map.header.nz
    vx, vy, vz = density_map.voxel_size.x, density_map.voxel_size.y, density_map.voxel_size.z
    
    # Bins centered at origin
    binsx = (np.linspace(0, nx, nx + 1) - nx/2) * vx
    binsy = (np.linspace(0, ny, ny + 1) - ny/2) * vy
    binsz = (np.linspace(0, nz, nz + 1) - nz/2) * vz
    
    box_min = np.array([binsx[0], binsy[0], binsz[0]])
    box_max = np.array([binsx[-1], binsy[-1], binsz[-1]])
    
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
    -log posterior = ExVol NLL + Pair NLL + Tetramer NLL + Octet NLL + (1-CCC) + (-log prior).

    The posterior includes:
    - Excluded volume constraints (soft repulsion)
    - Pairwise distance likelihoods (AA, AB, BC)
    - Tetramer formation scores (AB + 2×BC bonds)
    - Octet formation scores (inter-tetramer AA distances)
    - EM density fit (1 - CCC, range 0-2)
    - Prior on sigma parameters

    Returns:
        (total_score, exclusion_score, pair_score, tetramer_score, octet_score, em_score, prior_penalty)
    """
    # **FIX: Use per-state EM scorer instead of global**
    # Each replica gets its own scorer instance to avoid race conditions
    if not hasattr(state, '_em_scorer') or state._em_scorer is None:
        if _EM_MAP_FILE is None:
            raise ValueError(
                "EM map not configured. Call set_em_map_config() before sampling, or "
                "pass em_map_file to run_full_sampling()."
            )
        # Initialize scorer for this state instance (happens once per replica)
        state._em_scorer = FullNLL(
            state, 
            em_map_file=_EM_MAP_FILE,
            resolution=_EM_RESOLUTION, 
            backend=_EM_BACKEND
        )
    
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

    # **FIX: Use this state's own EM scorer (not global)**
    # Each state has its own scorer instance, no race conditions
    state._em_scorer.coordinates = state.positions
    em_score = state._em_scorer.compute_score()  # Returns 1 - CCC

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
    """
    global _EM_MAP_FILE, _EM_RESOLUTION, _EM_BACKEND
    
    # Allow overriding global config with function arguments
    if em_map_file is not None:
        set_em_map_config(em_map_file, em_resolution, em_backend)
    elif _EM_MAP_FILE is None:
        raise ValueError(
            "EM map not configured. Either:\n"
            "1. Call set_em_map_config() before running, or\n"
            "2. Pass em_map_file argument to run_full_sampling()"
        )
    
    print("="*70)
    print("Full Sampling with EM Density Restraint")
    print("="*70)
    
    # Center particles to density COM (done once at start)
    if center_to_density:
        print("\nCentering particles to density map COM...")
        state = center_state_to_density(state)
    
    # **FIX: No need to initialize global scorer**
    # Each state (including replicas) will lazy-init its own scorer
    # The first call to neg_log_posterior() will create it
    
    print("Starting MCMC sampling...")
    print("="*70 + "\n")
    
    # Define move proposals
    propose_fns = {
        "octet": propose_octet_move,
        "tetramer": propose_tetramer_move,
        "position": propose_particle_move,
        "sigma": propose_sigma_move,
        "full": propose_full_move
    }
    
    move_probs = {
        "octet": 0.10,      # Large structural units
        "tetramer": 0.50,   # Mid-level structural units
        "position": 0.20,   # Local fine-tuning
        "sigma": 0.10,      # Parameter updates
        "full": 0.10        # Full system moves
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

# Backward compatibility
set_em_map = set_em_map_config  # Alias for old code