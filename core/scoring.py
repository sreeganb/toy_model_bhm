#--------------------------------------------------------------------------------
# Scoring functions for pair, tetramer, and octamer interaction models
# These are optimized using vectorized operations and designed to be modular
#--------------------------------------------------------------------------------
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any, Union
import numba as nb
from functools import lru_cache

#************************************
# Imports from files here
#************************************
from core.parameters import SystemParameters

#--------------------------------------------------------------------------------
# Utility functions
#--------------------------------------------------------------------------------
def get_system_parameters():
    """Get system parameters from cached or default values"""
    # This could be replaced with a singleton or cached version
    # for now, just returning default values
    return {
        'pair_distances': {
            'AA': 4.0,
            'AB': 3.0,
            'BC': 2.0,
            'CC': 2.5
        },
        'radii': {
            'A': 0.5,
            'B': 0.5, 
            'C': 0.5
        },
        'box_size': 10.0,
        'excluded_volume_scale': 1000.0
    }

#--------------------------------------------------------------------------------
# Base scoring components
#--------------------------------------------------------------------------------
def calculate_excluded_volume(
    positions: Dict[str, np.ndarray],
    params: Optional[Dict[str, Any]] = None
) -> float:
    """
    Calculate excluded volume score efficiently using vectorized operations.
    
    Args:
        positions: Dictionary mapping particle types to position arrays (shape: Nx3)
        params: Optional parameters dictionary
        
    Returns:
        Excluded volume score (higher for overlapping particles)
    """
    if params is None:
        params = get_system_parameters()
    
    ex_score = 0.0
    radii = params['radii']
    box_size = params.get('box_size', 10.0)
    scale = params.get('excluded_volume_scale', 1000.0)
    
    # Calculate score for each pair of particle types
    particle_types = list(positions.keys())
    
    for i, type1 in enumerate(particle_types):
        pos1 = positions[type1]
        if len(pos1) == 0:
            continue
            
        r1 = radii[type1]
        
        # Calculate intra-type interactions (particles of same type)
        if len(pos1) > 1:
            # Compute all pairwise distances using broadcasting
            delta = pos1[:, np.newaxis, :] - pos1[np.newaxis, :, :]
            
            # Apply periodic boundary conditions (assuming cubic box)
            delta = np.where(delta > box_size/2, delta - box_size, delta)
            delta = np.where(delta < -box_size/2, delta + box_size, delta)
            
            # Calculate squared distances (avoid sqrt for performance)
            sq_dist = np.sum(delta**2, axis=2)
            
            # Set diagonal to infinity to exclude self-interactions
            np.fill_diagonal(sq_dist, np.inf)
            
            # Calculate overlap penalties where distance < 2*radius
            min_dist = 2 * r1
            overlap = np.maximum(0, min_dist**2 - sq_dist)
            ex_score += scale * np.sum(overlap) / 2  # Divide by 2 to avoid double counting
        
        # Calculate inter-type interactions (particles of different types)
        for j, type2 in enumerate(particle_types[i+1:], i+1):
            pos2 = positions[type2]
            if len(pos2) == 0:
                continue
                
            r2 = radii[type2]
            min_dist = r1 + r2
            
            # Calculate all pairwise distances between different types
            delta = pos1[:, np.newaxis, :] - pos2[np.newaxis, :, :]
            
            # Apply periodic boundary conditions
            delta = np.where(delta > box_size/2, delta - box_size, delta)
            delta = np.where(delta < -box_size/2, delta + box_size, delta)
            
            sq_dist = np.sum(delta**2, axis=2)
            
            # Calculate overlap penalties
            overlap = np.maximum(0, min_dist**2 - sq_dist)
            ex_score += scale * np.sum(overlap)
    
    return ex_score

def calculate_pairwise_distances(
    positions: Dict[str, np.ndarray], 
    pair_types: List[Tuple[str, str]],
    box_size: float = 10.0
) -> Dict[Tuple[str, str], np.ndarray]:
    """
    Calculate pairwise distance matrices for specified particle type pairs.
    
    Args:
        positions: Dictionary mapping particle types to position arrays
        pair_types: List of (type1, type2) pairs to calculate
        box_size: Size of periodic box
        
    Returns:
        Dictionary mapping (type1, type2) to distance matrix
    """
    distances = {}
    
    for type1, type2 in pair_types:
        pos1 = positions[type1]
        pos2 = positions[type2]
        
        if len(pos1) == 0 or len(pos2) == 0:
            # Create empty matrix if no particles of this type
            distances[(type1, type2)] = np.zeros((0, 0))
            continue
        
        # Calculate all pairwise differences
        delta = pos1[:, np.newaxis, :] - pos2[np.newaxis, :, :]
        
        # Apply periodic boundary conditions
        delta = np.where(delta > box_size/2, delta - box_size, delta)
        delta = np.where(delta < -box_size/2, delta + box_size, delta)
        
        # Calculate Euclidean distances
        dist_matrix = np.sqrt(np.sum(delta**2, axis=2))
        
        # Store in dictionary
        distances[(type1, type2)] = dist_matrix
        
    return distances

#--------------------------------------------------------------------------------
# Pair scoring
#--------------------------------------------------------------------------------
def calculate_pair_scores(
    positions: Dict[str, np.ndarray],
    sigma: Dict[str, float],
    excluded_pairs: Optional[Set[Tuple]] = None,
    params: Optional[Dict[str, Any]] = None,
    debug: bool = False
) -> float:
    """
    Calculate pairwise interaction scores with optional excluded pairs.
    
    Args:
        positions: Dictionary mapping particle types to position arrays
        sigma: Dictionary mapping pair types to sigma values
        excluded_pairs: Set of (type1, idx1, type2, idx2) tuples to exclude
        params: Optional parameters dictionary
        debug: Whether to print debug information
        
    Returns:
        Total pairwise score
    """
    if params is None:
        params = get_system_parameters()
    
    pair_distances = params['pair_distances']
    box_size = params.get('box_size', 10.0)
    
    # Initialize score and tracked pairs
    total_score = 0.0
    
    # Convert excluded_pairs to a more efficient lookup structure
    excluded_lookup = {}
    if excluded_pairs:
        for type1, idx1, type2, idx2 in excluded_pairs:
            key = (type1, type2)
            if key not in excluded_lookup:
                excluded_lookup[key] = set()
            excluded_lookup[key].add((idx1, idx2))
    
    # Calculate scores for each pair type
    for pair_key, target_dist in pair_distances.items():
        # Skip if this pair type doesn't have a sigma
        if pair_key not in sigma:
            if debug:
                print(f"Warning: No sigma for pair type {pair_key}, skipping")
            continue
        
        # Parse pair type
        if len(pair_key) == 2:
            type1, type2 = pair_key
        else:
            # Handle case like 'AA', 'AB', etc.
            type1, type2 = pair_key[0], pair_key[1]
        
        # Skip if either particle type doesn't exist
        if type1 not in positions or type2 not in positions:
            if debug:
                print(f"Warning: Missing particle type(s) for {pair_key}, skipping")
            continue
        
        pos1 = positions[type1]
        pos2 = positions[type2]
        
        # Skip if no particles
        if len(pos1) == 0 or len(pos2) == 0:
            continue
            
        # Handle self-interaction differently
        if type1 == type2:
            # Calculate all pairwise differences
            delta = pos1[:, np.newaxis, :] - pos1[np.newaxis, :, :]
            
            # Apply periodic boundary conditions
            delta = np.where(delta > box_size/2, delta - box_size, delta)
            delta = np.where(delta < -box_size/2, delta + box_size, delta)
            
            # Calculate distances
            distances = np.sqrt(np.sum(delta**2, axis=2))
            np.fill_diagonal(distances, np.inf)  # Avoid self-interactions
            
            # Create mask for excluded pairs
            mask = np.ones_like(distances, dtype=bool)
            if pair_key in excluded_lookup:
                for i, j in excluded_lookup[pair_key]:
                    if i < len(pos1) and j < len(pos1):
                        mask[i, j] = False
                        mask[j, i] = False  # Also exclude the reverse pair
            
            # Apply mask
            masked_distances = np.where(mask, distances, np.inf)
            
            # Find minimum distances in each row/column
            row_min_idx = np.argmin(masked_distances, axis=1)
            col_min_idx = np.argmin(masked_distances, axis=0)
            
            # Collect unique pairs
            pairs = set()
            for i, j in enumerate(row_min_idx):
                if masked_distances[i, j] < np.inf:
                    pairs.add((min(i, j), max(i, j)))  # Ordered pair to avoid duplicates
            
            for j, i in enumerate(col_min_idx):
                if masked_distances[i, j] < np.inf:
                    pairs.add((min(i, j), max(i, j)))
                    
            # Calculate score for each pair
            sigma_val = sigma[pair_key]
            for i, j in pairs:
                dist = distances[i, j]
                # Gaussian score
                pair_score = ((dist - target_dist)**2) / (2 * sigma_val**2) + np.log(2 * np.pi * sigma_val)
                total_score += pair_score
                
                if debug and pair_score > 10:
                    print(f"High score: {type1}({i})-{type2}({j}): {pair_score:.2f} (dist={dist:.2f}, target={target_dist:.2f})")
        
        else:
            # Inter-type interactions
            # Calculate all pairwise differences
            delta = pos1[:, np.newaxis, :] - pos2[np.newaxis, :, :]
            
            # Apply periodic boundary conditions
            delta = np.where(delta > box_size/2, delta - box_size, delta)
            delta = np.where(delta < -box_size/2, delta + box_size, delta)
            
            # Calculate distances
            distances = np.sqrt(np.sum(delta**2, axis=2))
            
            # Create mask for excluded pairs
            mask = np.ones_like(distances, dtype=bool)
            if (type1, type2) in excluded_lookup:
                for i, j in excluded_lookup[(type1, type2)]:
                    if i < len(pos1) and j < len(pos2):
                        mask[i, j] = False
                        
            # Also check reverse direction
            if (type2, type1) in excluded_lookup:
                for j, i in excluded_lookup[(type2, type1)]:
                    if i < len(pos1) and j < len(pos2):
                        mask[i, j] = False
            
            # Apply mask
            masked_distances = np.where(mask, distances, np.inf)
            
            # Find minimum distances in each row/column
            row_min_idx = np.argmin(masked_distances, axis=1)
            col_min_idx = np.argmin(masked_distances, axis=0)
            
            # Collect unique pairs
            pairs = set()
            for i, j in enumerate(row_min_idx):
                if masked_distances[i, j] < np.inf:
                    pairs.add((i, j))
                    
            for j, i in enumerate(col_min_idx):
                if masked_distances[j, i] < np.inf:
                    pairs.add((i, j))
            
            # Calculate score for each pair
            sigma_val = sigma[pair_key]
            for i, j in pairs:
                dist = distances[i, j]
                # Gaussian score
                pair_score = ((dist - target_dist)**2) / (2 * sigma_val**2) + np.log(2 * np.pi * sigma_val)
                total_score += pair_score
                
                if debug and pair_score > 10:
                    print(f"High score: {type1}({i})-{type2}({j}): {pair_score:.2f} (dist={dist:.2f}, target={target_dist:.2f})")
    
    return total_score

#--------------------------------------------------------------------------------
# Tetramer scoring functions
#--------------------------------------------------------------------------------
def get_tetramers(
    positions: Dict[str, np.ndarray],
    params: Optional[Dict[str, Any]] = None,
    cutoff_factor: float = 1.5
) -> List[Tuple[int, int, int, int]]:
    """
    Identify tetramers in the system based on spatial proximity.
    Tetramers consist of A-B-C-C units where B connects to two C particles.
    
    Args:
        positions: Dictionary mapping particle types to position arrays
        params: Optional parameters dictionary
        cutoff_factor: Distance cutoff factor relative to target distances
        
    Returns:
        List of tetramers as (a_idx, b_idx, c1_idx, c2_idx) tuples
    """
    if params is None:
        params = get_system_parameters()
    
    pair_distances = params['pair_distances']
    box_size = params.get('box_size', 10.0)
    
    # Verify required particle types exist
    for particle_type in ['A', 'B', 'C']:
        if particle_type not in positions or len(positions[particle_type]) == 0:
            return []  # Cannot form tetramers if missing any component
    
    # Calculate cutoff distances
    ab_cutoff = cutoff_factor * pair_distances['AB']
    bc_cutoff = cutoff_factor * pair_distances['BC']
    cc_cutoff = cutoff_factor * pair_distances['CC']
    
    # Calculate distance matrices
    ab_distances = calculate_pairwise_distances(
        positions, [('A', 'B')], box_size
    )[('A', 'B')]
    
    bc_distances = calculate_pairwise_distances(
        positions, [('B', 'C')], box_size
    )[('B', 'C')]
    
    cc_distances = calculate_pairwise_distances(
        positions, [('C', 'C')], box_size
    )[('C', 'C')]
    
    # Identify potential AB connections
    ab_connections = []
    for a_idx in range(len(positions['A'])):
        # Get indices of B particles within cutoff
        b_indices = np.where(ab_distances[a_idx] < ab_cutoff)[0]
        for b_idx in b_indices:
            ab_connections.append((a_idx, b_idx))
    
    # Find tetramers
    tetramers = []
    
    # For each AB connection, find C pairs connected to B
    for a_idx, b_idx in ab_connections:
        # Get C particles connected to this B
        c_indices = np.where(bc_distances[b_idx] < bc_cutoff)[0]
        
        # Check all pairs of C particles
        for i, c1_idx in enumerate(c_indices):
            for c2_idx in c_indices[i+1:]:
                # Verify C-C distance
                c_dist = cc_distances[c1_idx, c2_idx]
                if c_dist < cc_cutoff:
                    tetramers.append((a_idx, b_idx, c1_idx, c2_idx))
    
    return tetramers

def calculate_tetramer_scores(
    positions: Dict[str, np.ndarray],
    tetramers: List[Tuple[int, int, int, int]],
    sigma: Dict[str, float],
    params: Optional[Dict[str, Any]] = None,
    debug: bool = False
) -> np.ndarray:
    """
    Calculate scores for all tetramers with vectorized operations.
    
    Args:
        positions: Dictionary mapping particle types to position arrays
        tetramers: List of (a_idx, b_idx, c1_idx, c2_idx) tuples
        sigma: Dictionary mapping pair types to sigma values
        params: Optional parameters dictionary
        debug: Whether to print debug information
        
    Returns:
        Array of scores, one per tetramer
    """
    if not tetramers:
        return np.array([], dtype=np.float32)
    
    if params is None:
        params = get_system_parameters()
        
    pair_distances = params['pair_distances']
    box_size = params.get('box_size', 10.0)
    
    n_tetramers = len(tetramers)
    
    # Extract all tetramer indices efficiently
    a_indices = np.array([t[0] for t in tetramers], dtype=np.int32)
    b_indices = np.array([t[1] for t in tetramers], dtype=np.int32)
    c1_indices = np.array([t[2] for t in tetramers], dtype=np.int32)
    c2_indices = np.array([t[3] for t in tetramers], dtype=np.int32)
    
    # Get all positions in single vectorized operations
    pos_a = positions['A'][a_indices]
    pos_b = positions['B'][b_indices]
    pos_c1 = positions['C'][c1_indices]
    pos_c2 = positions['C'][c2_indices]
    
    # Calculate all distances, handling periodic boundary conditions
    ab_delta = pos_a - pos_b
    bc1_delta = pos_b - pos_c1
    bc2_delta = pos_b - pos_c2
    cc_delta = pos_c1 - pos_c2
    
    # Apply periodic boundary conditions
    for delta in [ab_delta, bc1_delta, bc2_delta, cc_delta]:
        delta[delta > box_size/2] -= box_size
        delta[delta < -box_size/2] += box_size
    
    # Calculate all distances at once
    ab_dists = np.sqrt(np.sum(ab_delta**2, axis=1))
    bc1_dists = np.sqrt(np.sum(bc1_delta**2, axis=1))
    bc2_dists = np.sqrt(np.sum(bc2_delta**2, axis=1))
    cc_dists = np.sqrt(np.sum(cc_delta**2, axis=1))
    
    # Cache target distances for performance
    ab_target = pair_distances['AB']
    bc_target = pair_distances['BC']
    cc_target = pair_distances['CC']
    
    # Calculate scores individually for detailed logging
    ab_scores = ((ab_dists - ab_target)**2) / (2 * sigma['AB']**2) + np.log(2 * np.pi * sigma['AB'])
    bc1_scores = ((bc1_dists - bc_target)**2) / (2 * sigma['BC']**2) + np.log(2 * np.pi * sigma['BC'])
    bc2_scores = ((bc2_dists - bc_target)**2) / (2 * sigma['BC']**2) + np.log(2 * np.pi * sigma['BC'])
    cc_scores = ((cc_dists - cc_target)**2) / (2 * sigma['CC']**2) + np.log(2 * np.pi * sigma['CC'])
    
    # Calculate total scores
    scores = ab_scores + bc1_scores + bc2_scores + cc_scores
    
    # Debug output if requested
    if debug:
        for i in range(min(5, n_tetramers)):  # Show first 5 for debugging
            print(f"Tetramer {i}: Score={scores[i]:.2f}")
            print(f"  A-B: {ab_dists[i]:.2f} vs {ab_target:.2f}, Score: {ab_scores[i]:.2f}")
            print(f"  B-C1: {bc1_dists[i]:.2f} vs {bc_target:.2f}, Score: {bc1_scores[i]:.2f}")
            print(f"  B-C2: {bc2_dists[i]:.2f} vs {bc_target:.2f}, Score: {bc2_scores[i]:.2f}")
            print(f"  C-C: {cc_dists[i]:.2f} vs {cc_target:.2f}, Score: {cc_scores[i]:.2f}")
    
    return scores

def get_tetramer_pairs(
    tetramers: List[Tuple[int, int, int, int]]
) -> Set[Tuple[str, int, str, int]]:
    """
    Get the set of pairs that are part of tetramers, for exclusion from pair scoring.
    
    Args:
        tetramers: List of (a_idx, b_idx, c1_idx, c2_idx) tuples
        
    Returns:
        Set of (type1, idx1, type2, idx2) tuples representing pairs in tetramers
    """
    if not tetramers:
        return set()
    
    tetramer_pairs = set()
    
    # Process all tetramers
    for a_idx, b_idx, c1_idx, c2_idx in tetramers:
        # Add pairs to excluded set (these are scored separately as tetramer pairs)
        tetramer_pairs.add(('A', a_idx, 'B', b_idx))
        tetramer_pairs.add(('B', b_idx, 'A', a_idx))  # Add reverse pair too
        
        tetramer_pairs.add(('B', b_idx, 'C', c1_idx))
        tetramer_pairs.add(('C', c1_idx, 'B', b_idx))
        
        tetramer_pairs.add(('B', b_idx, 'C', c2_idx))
        tetramer_pairs.add(('C', c2_idx, 'B', b_idx))
        
        tetramer_pairs.add(('C', c1_idx, 'C', c2_idx))
        tetramer_pairs.add(('C', c2_idx, 'C', c1_idx))
    
    return tetramer_pairs

#--------------------------------------------------------------------------------
# Octamer scoring functions 
#--------------------------------------------------------------------------------
def get_octets(
    positions: Dict[str, np.ndarray],
    tetramers: Optional[List[Tuple[int, int, int, int]]] = None,
    params: Optional[Dict[str, Any]] = None,
    temperature: float = 0.9,
    distance_cutoff_factor: float = 2.0
) -> List[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]]:
    """
    Group tetramers into octets (pairs of tetramers) with temperature-based selection.
    
    Args:
        positions: Dictionary mapping particle types to position arrays
        tetramers: List of tetramers, if already calculated
        params: Optional parameters dictionary
        temperature: Temperature parameter for probabilistic selection (0-1)
        distance_cutoff_factor: How far tetramers can be to form an octet
        
    Returns:
        List of (tetramer1, tetramer2) pairs forming octets
    """
    if params is None:
        params = get_system_parameters()
        
    box_size = params.get('box_size', 10.0)
    
    # Calculate tetramers if not provided
    if tetramers is None:
        tetramers = get_tetramers(positions, params)
    
    if len(tetramers) < 2:
        return []
    
    # Calculate tetramer centers using vectorized operations
    centers = np.zeros((len(tetramers), 3))
    for i, (a_idx, b_idx, c_idx1, c_idx2) in enumerate(tetramers):
        coords = np.vstack([
            positions['A'][a_idx],
            positions['B'][b_idx],
            positions['C'][c_idx1],
            positions['C'][c_idx2]
        ])
        centers[i] = np.mean(coords, axis=0)
    
    # Calculate typical distance scale from parameters
    distance_scale = params['pair_distances']['AB']
    max_distance = distance_cutoff_factor * distance_scale
    
    # Form octets by pairing tetramers
    octets = []
    available = list(range(len(tetramers)))
    
    while len(available) >= 2:
        # Pick first tetramer randomly
        idx1 = np.random.choice(available)
        available.remove(idx1)
        
        # Calculate distances to all other tetramers
        diffs = centers[available] - centers[idx1]
        
        # Apply periodic boundary conditions
        diffs = np.where(diffs > box_size/2, diffs - box_size, diffs)
        diffs = np.where(diffs < -box_size/2, diffs + box_size, diffs)
        
        # Calculate distances and sort
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        sorted_indices = np.argsort(distances)
        
        # Filter by maximum distance
        valid_indices = [i for i, d in enumerate(distances) if d <= max_distance]
        
        if not valid_indices:
            # No valid partners for this tetramer
            continue
            
        # Apply temperature-based selection
        if temperature > 0:
            # Calculate selection probabilities (closer tetramers more likely)
            valid_distances = distances[valid_indices]
            probs = np.exp(-valid_distances / (temperature * distance_scale))
            probs /= probs.sum()
            
            # Select partner based on probabilities
            selected_idx = np.random.choice(valid_indices, p=probs)
        else:
            # At zero temperature, always pick closest
            selected_idx = sorted_indices[0] if distances[sorted_indices[0]] <= max_distance else None
            
        if selected_idx is not None:
            # Get the actual index in available list
            idx2 = available[selected_idx]
            available.remove(idx2)
            
            # Form the octet
            octets.append((tetramers[idx1], tetramers[idx2]))
    
    return octets

def calculate_octet_scores(
    positions: Dict[str, np.ndarray],
    octets: List[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]],
    sigma: Dict[str, float],
    params: Optional[Dict[str, Any]] = None,
    debug: bool = False
) -> np.ndarray:
    """
    Calculate scores for octets, treating each component tetramer individually.
    
    Args:
        positions: Dictionary mapping particle types to position arrays
        octets: List of (tetramer1, tetramer2) pairs
        sigma: Dictionary mapping pair types to sigma values
        params: Optional parameters dictionary
        debug: Whether to print debug information
        
    Returns:
        Array of scores, one per octet
    """
    if not octets:
        return np.array([], dtype=np.float32)
    
    # Extract tetramers from octets
    all_tetramers = []
    for tet1, tet2 in octets:
        all_tetramers.extend([tet1, tet2])
    
    # Calculate scores for all tetramers
    tetramer_scores = calculate_tetramer_scores(positions, all_tetramers, sigma, params, debug=False)
    
    # Group scores by octet
    octet_scores = np.zeros(len(octets), dtype=np.float32)
    for i in range(len(octets)):
        # Sum scores of both tetramers in the octet
        octet_scores[i] = tetramer_scores[2*i] + tetramer_scores[2*i + 1]
        
        # Special octet interactions - placeholder for inter-tetramer interactions
        # This could be expanded with specific octet interaction terms
    
    if debug and len(octets) > 0:
        print(f"Calculated scores for {len(octets)} octets:")
        for i in range(min(5, len(octets))):
            print(f"  Octet {i}: Score={octet_scores[i]:.2f}")
    
    return octet_scores

def get_octet_pairs(
    octets: List[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]]
) -> Set[Tuple[str, int, str, int]]:
    """
    Get all pairs that are part of octets, for exclusion from pair scoring.
    This combines pairs from all tetramers that are part of octets.
    
    Args:
        octets: List of (tetramer1, tetramer2) pairs
        
    Returns:
        Set of (type1, idx1, type2, idx2) tuples representing pairs in octets
    """
    # Extract individual tetramers
    tetramers = []
    for tet1, tet2 in octets:
        tetramers.extend([tet1, tet2])
    
    # Get pairs using the tetramer_pairs function
    return get_tetramer_pairs(tetramers)

#--------------------------------------------------------------------------------
# Combined scoring function
#--------------------------------------------------------------------------------
def calculate_total_score(
    positions: Dict[str, np.ndarray],
    sigma: Dict[str, float],
    tetramers: Optional[List[Tuple]] = None,
    octets: Optional[List[Tuple]] = None,
    sigma_prior_penalty: float = 0.0,
    exclusion_weight: float = 1.0,
    pair_weight: float = 1.0,
    tetramer_weight: float = 1.0,
    octet_weight: float = 1.0,
    params: Optional[Dict[str, Any]] = None,
    debug: bool = False
) -> Tuple[float, float, float, float, float]:
    """
    Calculate combined score for the system with all components.
    
    Args:
        positions: Dictionary mapping particle types to position arrays
        sigma: Dictionary mapping pair types to sigma values
        tetramers: List of tetramers, if already calculated
        octets: List of octets, if already calculated
        sigma_prior_penalty: Prior penalty for sigma values
        exclusion_weight: Weight for excluded volume score
        pair_weight: Weight for pairwise score
        tetramer_weight: Weight for tetramer score
        octet_weight: Weight for octet score
        params: Optional parameters dictionary
        debug: Whether to print debug information
        
    Returns:
        Tuple of (total_score, exclusion_score, pair_score, tetramer_score, octet_score)
    """
    if params is None:
        params = get_system_parameters()
    
    # Calculate excluded volume score
    exclusion_score = exclusion_weight * calculate_excluded_volume(positions, params)
    
    # Identify tetramers if not provided
    if tetramers is None:
        tetramers = get_tetramers(positions, params)
        
    # Identify octets if not provided
    if octets is None and tetramers:
        octets = get_octets(positions, tetramers, params)
    elif octets is None:
        octets = []
    
    # Get pairs to exclude from pair scoring (pairs in tetramers and octets)
    excluded_pairs = set()
    if tetramers:
        excluded_pairs.update(get_tetramer_pairs(tetramers))
    
    # Calculate pairwise score excluding tetramer/octet pairs
    pair_score = pair_weight * calculate_pair_scores(
        positions, sigma, excluded_pairs, params, debug=debug
    )
    
    # Calculate tetramer score
    tetramer_score = 0.0
    if tetramers:
        scores = calculate_tetramer_scores(positions, tetramers, sigma, params, debug=debug)
        tetramer_score = tetramer_weight * np.sum(scores)
    
    # Calculate octet score
    octet_score = 0.0
    if octets:
        scores = calculate_octet_scores(positions, octets, sigma, params, debug=debug)
        octet_score = octet_weight * np.sum(scores)
    
    # Calculate total score
    total_score = exclusion_score + pair_score + tetramer_score + octet_score + sigma_prior_penalty
    
    # Debug output if requested
    if debug:
        print(f"\n===== SCORE SUMMARY =====")
        print(f"Exclusion Score: {exclusion_score:.2f}")
        print(f"Pair Score: {pair_score:.2f}")
        print(f"Tetramer Score: {tetramer_score:.2f}")
        print(f"Octet Score: {octet_score:.2f}")
        print(f"Prior Penalty: {sigma_prior_penalty:.2f}")
        print(f"Total Score: {total_score:.2f}")
    
    return total_score, exclusion_score, pair_score, tetramer_score, octet_score