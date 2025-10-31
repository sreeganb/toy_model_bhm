# core/movers.py
import random
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Optional, Tuple
from core.parameters import SystemParameters

# --------------------------
# Helpers
# --------------------------
def _reflect_scalar(x: float, a: float, b: float) -> float:
    """Exact reflective boundary on [a, b] (preserves symmetry)."""
    if not np.isfinite(x):
        return float(np.clip(x, a, b))
    w = b - a
    if w <= 0.0:
        return float(np.clip(x, a, b))
    # Reflect repeatedly until within bounds
    while x < a or x > b:
        if x < a:
            x = a + (a - x)
        if x > b:
            x = b - (x - b)
    return float(x)

def _flat_index_choice(positions: Dict[str, np.ndarray]) -> Tuple[str, int]:
    """Choose a single particle uniformly over all particles across types."""
    types = list(positions.keys())
    counts = [positions[t].shape[0] for t in types]
    total = sum(counts)
    if total == 0:
        raise ValueError("No particles to move: positions are empty")
    # Draw a flat index then map to (type, local_idx)
    k = np.random.randint(total)
    acc = 0
    for t, c in zip(types, counts):
        if k < acc + c:
            return t, k - acc
        acc += c
    # Fallback (should not happen)
    return types[-1], counts[-1] - 1

# --------------------------
# Position proposal (additive Gaussian, reflective walls)
# --------------------------
def propose_particle_move(state, accept_rate: float = 0.5):
    """
    Single-particle Gaussian move with reflective walls on [0, box_size]^3.

    - Select one particle uniformly across all types.
    - Use type-dependent step scale based on radii (larger particles move less).
    - Reflect at the walls to preserve proposal symmetry.
    """
    params = SystemParameters()
    box_size = getattr(state, "box_size", params.box_size)

    # Select particle uniformly across all types
    ptype, local_idx = _flat_index_choice(state.positions)

    # Step size heuristic using radii - NO DEFAULTS, MUST BE DEFINED
    radius = params.radii[ptype]  # Will crash if not defined - that's correct!
    max_radius = max(params.radii.values())
    
    # Larger radius -> smaller step. Gaussian proposal.
    step_sigma = 2.0 * (max_radius / radius)

    current = state.positions[ptype][local_idx]
    proposal = current + np.random.normal(0.0, step_sigma, size=3)

    # Reflective walls on [0, box_size]
    for d in range(3):
        proposal[d] = _reflect_scalar(float(proposal[d]), 0.0, float(box_size))

    state.positions[ptype][local_idx] = proposal
    return True

# --------------------------
# Sigma proposal (additive Gaussian in linear sigma, reflective)
# --------------------------
def propose_sigma_move(state, accept_rate: Optional[float] = None):
    """
    Non-adaptive Metropolis proposal that preserves detailed balance.

    - Selects a single pair_type uniformly at random.
    - Uses an additive Gaussian step in linear sigma with constant scale
      per parameter (independent of the current value/state).
    - Applies exact reflective boundary conditions on [low, high].

    Modifies state.sigma in-place.
    """
    if not hasattr(state, "sigma") or not isinstance(state.sigma, dict) or len(state.sigma) == 0:
        return False

    # Choose parameter uniformly
    pair_type = random.choice(list(state.sigma.keys()))
    current_val = float(state.sigma[pair_type])

    # Bounds: from prior if present, else from state.sigma_range
    low, high = 1e-6, 20.0
    if hasattr(state, "sigma_prior") and getattr(state.sigma_prior, "sigma_ranges", None):
        low, high = state.sigma_prior.sigma_ranges.get(pair_type, (low, high))
    elif hasattr(state, "sigma_range") and isinstance(state.sigma_range, dict):
        low, high = state.sigma_range.get(pair_type, (low, high))

    if not np.isfinite(current_val) or current_val <= 0.0:
        # Snap invalid current value inside the bounds
        current_val = float(np.clip(0.5 * (low + high), low, high))

    # Constant, state-independent proposal width for symmetry
    width = max(high - low, 1e-9)
    step_sd = 0.15 * width  # tune globally if needed

    # Symmetric additive Gaussian proposal in sigma-space
    proposed = current_val + np.random.normal(0.0, step_sd)

    # Reflect to [low, high]
    proposed_val = _reflect_scalar(proposed, float(low), float(high))

    # Apply
    state.sigma[pair_type] = float(proposed_val)
    return True

# --------------------------
# Tetramer helpers and moves
# --------------------------
def get_tetramers(state) -> List[Tuple[int, ...]]:
    """Use Hungarian algorithm for optimal A-B matching, then greedy C selection."""
    try:
        from scipy.optimize import linear_sum_assignment

        positions = state.positions
        params = SystemParameters()

        if not all(k in positions and len(positions[k]) > 0 for k in ['A', 'B', 'C']) or len(positions['C']) < 2:
            return []

        a_pos, b_pos, c_pos = positions['A'], positions['B'], positions['C']

        ab_target = params.pair_distances['AB']
        bc_target = params.pair_distances['BC']

        dist_AB = cdist(a_pos, b_pos)
        cost_matrix = np.abs(dist_AB - ab_target)

        a_indices, b_indices = linear_sum_assignment(cost_matrix)

        dist_BC = cdist(b_pos, c_pos)

        c_used = set()
        tetramers = []

        pair_costs = cost_matrix[a_indices, b_indices]
        sorted_pairs = np.argsort(pair_costs)

        for pair_idx in sorted_pairs:
            a_idx = a_indices[pair_idx]
            b_idx = b_indices[pair_idx]

            available_c = [i for i in range(len(c_pos)) if i not in c_used]
            if len(available_c) < 2:
                break

            bc_dists = dist_BC[b_idx, available_c]
            c_scores = np.abs(bc_dists - bc_target)

            best_c_local = np.argsort(c_scores)[:2]
            best_c_indices = [available_c[i] for i in best_c_local]

            tetramers.append((a_idx, b_idx, best_c_indices[0], best_c_indices[1]))
            c_used.update(best_c_indices)

            if len(tetramers) >= min(len(a_pos), len(b_pos), len(c_pos) // 2):
                break

        return tetramers

    except Exception as e:
        print(f"Error in Hungarian tetramer generation: {e}")
        return []

def propose_tetramer_move(state, acceptance_rate: float = 0.5):
    """
    Optimized tetramer move proposal with decoupled translation/rotation.
    60% probability for translation, 30% for rotation, 10% for mixed moves.
    Clips coordinates to stay within box boundaries minus particle radius.

    Modifies state.positions in-place.
    """
    tetramers = get_tetramers(state)
    if not tetramers:
        return False

    params = SystemParameters()
    box_size = params.box_size
    half_box = box_size / 2.0

    # Choose random tetramer
    tetramer = tetramers[np.random.randint(len(tetramers))]
    a_idx, b_idx, c_idx1, c_idx2 = tetramer

    # Pre-extract
    particles = [('A', a_idx), ('B', b_idx), ('C', c_idx1), ('C', c_idx2)]
    coords = np.array([state.positions[part][idx] for part, idx in particles])
    centroid = np.mean(coords, axis=0)

    # Compute buffer based on particle radii - NO DEFAULTS
    tet_radii = [params.radii[p] for p, _ in particles]
    buffer_radius = max(tet_radii)

    max_coord = half_box - 2.0 * buffer_radius
    min_coord = -max_coord

    # Adaptive size factor (bounded)
    distances_from_center = np.linalg.norm(coords - centroid, axis=1)
    tetramer_radius = float(np.max(distances_from_center)) if distances_from_center.size else 1.0
    size_factor = max(0.5, min(2.0, tetramer_radius))

    # Fixed step sizes
    trans_step = 0.25
    rot_step = 0.2

    # Choose move type: 60% translation, 30% rotation, 10% mixed
    r = np.random.random()

    if r < 0.6:
        # Translation
        displacement = np.random.normal(0.0, trans_step * size_factor, 3)
        for i, (part, idx) in enumerate(particles):
            final_pos = coords[i] + displacement
            final_pos = np.clip(final_pos, min_coord, max_coord)
            state.positions[part][idx] = final_pos.astype(float)

    elif r < 0.9:
        # Rotation (Marsaglia)
        while True:
            x1, x2 = np.random.uniform(-1, 1, 2)
            if x1 * x1 + x2 * x2 < 1:
                break
        sqrt_term = np.sqrt(1 - x1 * x1 - x2 * x2)
        rotation_axis = np.array([2 * x1 * sqrt_term, 2 * x2 * sqrt_term, 1 - 2 * (x1 * x1 + x2 * x2)])
        rotation_axis /= max(np.linalg.norm(rotation_axis), 1e-12)

        rotation_angle = np.random.normal(0.0, rot_step)
        half_angle = rotation_angle / 2.0
        qw = np.cos(half_angle)
        sin_half = np.sin(half_angle)
        qx, qy, qz = rotation_axis * sin_half

        rot_matrix = np.array([
            [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]
        ])

        for i, (part, idx) in enumerate(particles):
            vec = coords[i] - centroid
            rotated = rot_matrix @ vec
            final_pos = centroid + rotated
            final_pos = np.clip(final_pos, min_coord, max_coord)
            state.positions[part][idx] = final_pos.astype(float)

    else:
        # Mixed (translation + rotation) with reduced steps
        trans_scale = 0.7
        rot_scale = 0.7

        displacement = np.random.normal(0.0, trans_step * size_factor * trans_scale, 3)

        while True:
            x1, x2 = np.random.uniform(-1, 1, 2)
            if x1 * x1 + x2 * x2 < 1:
                break
        sqrt_term = np.sqrt(1 - x1 * x1 - x2 * x2)
        rotation_axis = np.array([2 * x1 * sqrt_term, 2 * x2 * sqrt_term, 1 - 2 * (x1 * x1 + x2 * x2)])
        rotation_axis /= max(np.linalg.norm(rotation_axis), 1e-12)

        rotation_angle = np.random.normal(0.0, rot_step * rot_scale)
        half_angle = rotation_angle / 2.0
        qw = np.cos(half_angle)
        sin_half = np.sin(half_angle)
        qx, qy, qz = rotation_axis * sin_half

        rot_matrix = np.array([
            [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]
        ])

        for i, (part, idx) in enumerate(particles):
            vec = coords[i] - centroid
            rotated = rot_matrix @ vec
            final_pos = centroid + rotated + displacement
            final_pos = np.clip(final_pos, min_coord, max_coord)
            state.positions[part][idx] = final_pos.astype(float)

    return True

# --------------------------
# Octet helpers and moves
# --------------------------
def _compute_tetramer_centers_vectorized(tetramers: List[Tuple[int, ...]], 
                                        positions: Dict[str, np.ndarray]) -> np.ndarray:
    """Efficiently compute tetramer centers using vectorization."""
    if not tetramers:
        return np.array([]).reshape(0, 3)
    
    n_tetramers = len(tetramers)
    centers = np.zeros((n_tetramers, 3))
    
    # Vectorized computation
    for i, (a_idx, b_idx, c_idx1, c_idx2) in enumerate(tetramers):
        # Stack all coordinates and compute mean
        coords = np.array([
            positions['A'][a_idx],
            positions['B'][b_idx],
            positions['C'][c_idx1],
            positions['C'][c_idx2]
        ])
        centers[i] = np.mean(coords, axis=0)
    
    return centers

def get_octets(state) -> Tuple[List[Tuple[Tuple[int, ...], Tuple[int, ...]]], List[Tuple[int, ...]]]:
    """
    Hungarian algorithm for optimal tetramer pairing into octets - O(n³).
    
    Returns:
        (octets, tetramers) where:
        - octets: List of (tetramer1, tetramer2) pairs
        - tetramers: All tetramers found (for fallback)
    """
    tetramers = get_tetramers(state)
    
    if len(tetramers) < 2:
        return [], tetramers
    
    # For odd number of tetramers, we can't pair all of them
    n_pairs = len(tetramers) // 2
    if n_pairs == 0:
        return [], tetramers
    
    # Compute centers efficiently
    centers = _compute_tetramer_centers_vectorized(tetramers, state.positions)
    
    # Create cost matrix - only for the tetramers we can actually pair
    n_tetramers = 2 * n_pairs  # Use only even number
    cost_matrix = cdist(centers[:n_tetramers], centers[:n_tetramers])
    
    # Make diagonal infinite to prevent self-pairing
    np.fill_diagonal(cost_matrix, np.inf)
    
    # Solve assignment problem
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # Convert assignment to octets (avoid double counting)
    used = set()
    octets = []
    
    for i, j in zip(row_indices, col_indices):
        if i not in used and j not in used:
            octets.append((tetramers[i], tetramers[j]))
            used.add(i)
            used.add(j)
    
    return octets, tetramers

def propose_octet_move(state, acceptance_rate: float = 0.5):
    """
    Optimized octet move proposal with decoupled translation/rotation.
    60% probability for translation, 30% for rotation, 10% for mixed moves.
    Clips coordinates to stay within box boundaries minus particle radius.
    
    Modifies state.positions in-place.
    """
    octets, _ = get_octets(state)
    if not octets:
        return False
    
    params = SystemParameters()
    box_size = params.box_size
    half_box = box_size / 2.0
    
    # Select a random octet
    octet_idx = np.random.randint(len(octets))
    tetramer1, tetramer2 = octets[octet_idx]
    
    # Pre-extract octet particle information efficiently
    octet_particles = [
        ('A', tetramer1[0]), ('B', tetramer1[1]), ('C', tetramer1[2]), ('C', tetramer1[3]),
        ('A', tetramer2[0]), ('B', tetramer2[1]), ('C', tetramer2[2]), ('C', tetramer2[3])
    ]
    
    # Extract coordinates
    coords = np.array([state.positions[ptype][idx] for ptype, idx in octet_particles])
    centroid = np.mean(coords, axis=0)
    
    # Compute buffer based on particle radii - NO DEFAULTS
    oct_radii = [params.radii[p] for p, _ in octet_particles]
    buffer_radius = max(oct_radii)
    
    max_coord = half_box - 2.0 * buffer_radius
    min_coord = -max_coord
    
    # Calculate octet size for adaptive scaling
    distances_from_center = np.linalg.norm(coords - centroid, axis=1)
    octet_radius = float(np.max(distances_from_center)) if distances_from_center.size else 1.0
    size_factor = max(0.5, min(2.0, octet_radius))
    
    # Fixed step sizes
    trans_step = 0.25
    rot_step = 0.2
    
    # Choose move type: 60% translation, 30% rotation, 10% mixed
    rand_val = np.random.random()
    
    if rand_val < 0.6:
        # --- TRANSLATION MOVE ---
        displacement = np.random.normal(0, trans_step * size_factor, 3)
        
        for i, (ptype, idx) in enumerate(octet_particles):
            final_pos = coords[i] + displacement
            final_pos = np.clip(final_pos, min_coord, max_coord)
            state.positions[ptype][idx] = final_pos.astype(float)
    
    elif rand_val < 0.9:
        # --- ROTATION MOVE ---
        while True:
            x1, x2 = np.random.uniform(-1, 1, 2)
            if x1*x1 + x2*x2 < 1:
                break
        
        sqrt_term = np.sqrt(1 - x1*x1 - x2*x2)
        axis = np.array([2*x1*sqrt_term, 2*x2*sqrt_term, 1 - 2*(x1*x1 + x2*x2)])
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        
        angle = np.random.normal(0, rot_step)
        
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        one_minus_cos = 1 - cos_angle
        
        ux, uy, uz = axis
        rotation_matrix = np.array([
            [cos_angle + ux*ux*one_minus_cos, ux*uy*one_minus_cos - uz*sin_angle, ux*uz*one_minus_cos + uy*sin_angle],
            [uy*ux*one_minus_cos + uz*sin_angle, cos_angle + uy*uy*one_minus_cos, uy*uz*one_minus_cos - ux*sin_angle],
            [uz*ux*one_minus_cos - uy*sin_angle, uz*uy*one_minus_cos + ux*sin_angle, cos_angle + uz*uz*one_minus_cos]
        ])
        
        for i, (ptype, idx) in enumerate(octet_particles):
            centered = coords[i] - centroid
            rotated = rotation_matrix @ centered
            final_pos = centroid + rotated
            final_pos = np.clip(final_pos, min_coord, max_coord)
            state.positions[ptype][idx] = final_pos.astype(float)
    
    else:
        # --- MIXED MOVE ---
        trans_scale = 0.7
        rot_scale = 0.7
        
        displacement = np.random.normal(0, trans_step * size_factor * trans_scale, 3)
        
        while True:
            x1, x2 = np.random.uniform(-1, 1, 2)
            if x1*x1 + x2*x2 < 1:
                break
        
        sqrt_term = np.sqrt(1 - x1*x1 - x2*x2)
        axis = np.array([2*x1*sqrt_term, 2*x2*sqrt_term, 1 - 2*(x1*x1 + x2*x2)])
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        
        angle = np.random.normal(0, rot_step * rot_scale)
        
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        one_minus_cos = 1 - cos_angle
        
        ux, uy, uz = axis
        rotation_matrix = np.array([
            [cos_angle + ux*ux*one_minus_cos, ux*uy*one_minus_cos - uz*sin_angle, ux*uz*one_minus_cos + uy*sin_angle],
            [uy*ux*one_minus_cos + uz*sin_angle, cos_angle + uy*uy*one_minus_cos, uy*uz*one_minus_cos - ux*sin_angle],
            [uz*ux*one_minus_cos - uy*sin_angle, uz*uy*one_minus_cos + ux*sin_angle, cos_angle + uz*uz*one_minus_cos]
        ])
        
        for i, (ptype, idx) in enumerate(octet_particles):
            centered = coords[i] - centroid
            rotated = rotation_matrix @ centered
            final_pos = centroid + rotated + displacement
            final_pos = np.clip(final_pos, min_coord, max_coord)
            state.positions[ptype][idx] = final_pos.astype(float)
    
    return True

def propose_full_move(state, acceptance_rate: float = 0.5):
    """
    Propose a move that translates or rotates the entire system.
    80% translation, 20% rotation.
    """
    params = SystemParameters()
    box_size = params.box_size
    half_box = box_size / 2.0

    # Extract all particle information efficiently
    all_particles = []
    for ptype in ['A', 'B', 'C']:
        for idx in range(len(state.positions[ptype])):
            all_particles.append((ptype, idx))
    
    if not all_particles:
        return False
    
    # Extract coordinates
    coords = np.array([state.positions[ptype][idx] for ptype, idx in all_particles])
    centroid = np.mean(coords, axis=0)
    
    # Compute buffer based on particle radii - NO DEFAULTS
    all_radii = [params.radii[p] for p, _ in all_particles]
    buffer_radius = max(all_radii)
    
    max_coord = half_box - 2.0 * buffer_radius
    min_coord = -max_coord
    
    # Calculate system size for adaptive scaling
    distances_from_center = np.linalg.norm(coords - centroid, axis=1)
    system_radius = float(np.max(distances_from_center)) if distances_from_center.size else 1.0
    size_factor = max(0.5, min(2.0, system_radius))
    
    # Fixed step sizes
    trans_step = 0.1 * size_factor
    rot_step = 0.1 * size_factor
    
    # Choose move type: 80% translation, 20% rotation
    rand_val = np.random.random()
    
    if rand_val < 0.8:
        # --- TRANSLATION MOVE ---
        displacement = np.random.normal(0, trans_step, 3)
        
        for i, (ptype, idx) in enumerate(all_particles):
            final_pos = coords[i] + displacement
            final_pos = np.clip(final_pos, min_coord, max_coord)
            state.positions[ptype][idx] = final_pos.astype(float)
    else:
        # --- ROTATION MOVE ---
        while True:
            x1, x2 = np.random.uniform(-1, 1, 2)
            if x1*x1 + x2*x2 < 1:
                break
        
        sqrt_term = np.sqrt(1 - x1*x1 - x2*x2)
        axis = np.array([2*x1*sqrt_term, 2*x2*sqrt_term, 1 - 2*(x1*x1 + x2*x2)])
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        
        angle = np.random.normal(0, rot_step)
        
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        one_minus_cos = 1 - cos_angle
        
        ux, uy, uz = axis
        rotation_matrix = np.array([
            [cos_angle + ux*ux*one_minus_cos, ux*uy*one_minus_cos - uz*sin_angle, ux*uz*one_minus_cos + uy*sin_angle],
            [uy*ux*one_minus_cos + uz*sin_angle, cos_angle + uy*uy*one_minus_cos, uy*uz*one_minus_cos - ux*sin_angle],
            [uz*ux*one_minus_cos - uy*sin_angle, uz*uy*one_minus_cos + ux*sin_angle, cos_angle + uz*uz*one_minus_cos]
        ])
        
        for i, (ptype, idx) in enumerate(all_particles):
            centered = coords[i] - centroid
            rotated = rotation_matrix @ centered
            final_pos = centroid + rotated
            final_pos = np.clip(final_pos, min_coord, max_coord)
            state.positions[ptype][idx] = final_pos.astype(float)
    
    return True