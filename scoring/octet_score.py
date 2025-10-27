#============================================================================
# Octet scoring function (pairs of adjacent tetramers)
#============================================================================
import numpy as np
from core.parameters import SystemParameters
from core.movers import get_octets
from typing import Dict, List, Tuple

class OctetNLL:
    def __init__(self, state):
        """
        Initialize octet scoring with system state.
        
        Args:
            state: SystemState object containing positions and sigma values
        """
        self.state = state
        self.coordinates = state.positions
        self.nuisance_parameters = state.sigma
        self.params = SystemParameters()

    def calculate_octet_scores_batch(self, positions, octets, sig, debug_logging=False):
        """
        Calculate scores for all octets (pairs of adjacent tetramers).
        
        Each octet consists of two tetramers that should be adjacent, measured
        by the A-A distance between the two A particles in each tetramer.
        
        Uses Gaussian likelihood with proper normalization:
        NLL = (d - target)²/(2σ²) + 0.5*log(2πσ²)
        
        Args:
            positions: Dict mapping particle types to position arrays
            octets: List of (tetramer1, tetramer2) tuples
            sig: Dict of sigma values per pair type
            debug_logging: If True, print diagnostic information
            
        Returns:
            np.ndarray of scores (one per octet)
        """
        if not octets:
            return np.array([], dtype=np.float32)
        
        # Extract indices for A particles in each tetramer pair
        a1_indices = np.array([tet1[0] for tet1, tet2 in octets], dtype=np.int32)
        a2_indices = np.array([tet2[0] for tet1, tet2 in octets], dtype=np.int32)
        
        # Get A particle positions
        pos_a1 = positions['A'][a1_indices]
        pos_a2 = positions['A'][a2_indices]
        
        # Calculate distances between A1 and A2 in each octet (explicit sqrt)
        aa_dists = np.sqrt(np.sum((pos_a1 - pos_a2)**2, axis=1))
        
        # Define target distance for A-A between adjacent tetramers
        aa_inter_tetramer_target = self.params.pair_distances['AA']
        
        # Calculate NLL with proper Gaussian normalization
        # NLL = (d - μ)²/(2σ²) + 0.5*log(2πσ²)
        aa_scores = (
            ((aa_dists - aa_inter_tetramer_target)**2) / (2 * sig['AA']**2) + 
            0.5 * np.log(2 * np.pi * sig['AA']**2)
        )
        
        if debug_logging:
            print(f"Octet batch scores: mean={np.mean(aa_scores):.3f}, std={np.std(aa_scores):.3f}")
            print(f"  AA distances: mean={np.mean(aa_dists):.2f}, target={aa_inter_tetramer_target:.2f}")
            print(f"  Sigma AA: {sig['AA']:.3f}")
        
        return aa_scores

    def compute_score(self) -> float:
        """
        Sum of octet scores over current tetramer pairings.
        
        Uses Hungarian algorithm to optimally pair tetramers into octets,
        then scores each octet based on A-A inter-tetramer distance.
        
        Returns:
            Total negative log-likelihood for all octets
        """
        octets, _ = get_octets(self.state)
        
        if not octets:
            return 0.0
        
        octet_scores = self.calculate_octet_scores_batch(
            positions=self.coordinates,
            octets=octets,
            sig=self.nuisance_parameters,
            debug_logging=False
        )
        
        return float(np.sum(octet_scores))

    def compute_score_with_breakdown(self) -> Tuple[float, Dict[str, float]]:
        """
        Compute octet score with detailed breakdown.
        
        Returns:
            (total_score, breakdown_dict)
            
        breakdown_dict contains:
            - 'n_octets': Number of octets found
            - 'mean_score': Average score per octet
            - 'mean_distance': Average A-A inter-tetramer distance
            - 'target_distance': Target A-A distance
            - 'sigma': Sigma value used for scoring
        """
        octets, _ = get_octets(self.state)
        
        if not octets:
            return 0.0, {
                'n_octets': 0,
                'mean_score': 0.0,
                'mean_distance': 0.0,
                'target_distance': self.params.pair_distances['AA'],
                'sigma': self.nuisance_parameters.get('AA', 1.0)
            }
        
        # Calculate scores
        octet_scores = self.calculate_octet_scores_batch(
            positions=self.coordinates,
            octets=octets,
            sig=self.nuisance_parameters,
            debug_logging=False
        )
        
        # Calculate distances for breakdown
        a1_indices = np.array([tet1[0] for tet1, tet2 in octets], dtype=np.int32)
        a2_indices = np.array([tet2[0] for tet1, tet2 in octets], dtype=np.int32)
        pos_a1 = self.coordinates['A'][a1_indices]
        pos_a2 = self.coordinates['A'][a2_indices]
        aa_dists = np.sqrt(np.sum((pos_a1 - pos_a2)**2, axis=1))
        
        total_score = float(np.sum(octet_scores))
        
        breakdown = {
            'n_octets': len(octets),
            'mean_score': float(np.mean(octet_scores)),
            'mean_distance': float(np.mean(aa_dists)),
            'target_distance': self.params.pair_distances['AA'],
            'sigma': self.nuisance_parameters.get('AA', 1.0),
            'total_score': total_score
        }
        
        return total_score, breakdown
#============================================================================