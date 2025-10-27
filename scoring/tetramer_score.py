#============================================================================
# Tetrameric scoring function (matches previous batch scorer)
#============================================================================
import numpy as np
from core.parameters import SystemParameters
from core.movers import get_tetramers
from typing import Dict, List, Tuple

class TetramerNLL:
    def __init__(self, state):
        """
        Initialize tetramer scoring with system state.
        
        Args:
            state: SystemState object containing positions and sigma values
        """
        self.state = state
        self.coordinates = state.positions
        self.nuisance_parameters = state.sigma
        self.params = SystemParameters()

    def calculate_tetramer_scores_batch(self, positions, tetramers, sig, debug_logging=False):
        """Calculate scores for all tetramers (AB + BC1 + BC2) with 0.5*log(2πσ²)."""
        if not tetramers:
            return np.array([], dtype=np.float32)
        
        # Extract indices
        a_indices = np.array([t[0] for t in tetramers], dtype=np.int32)
        b_indices = np.array([t[1] for t in tetramers], dtype=np.int32)
        c1_indices = np.array([t[2] for t in tetramers], dtype=np.int32)
        c2_indices = np.array([t[3] for t in tetramers], dtype=np.int32)
        
        # Get positions
        pos_a = positions['A'][a_indices]
        pos_b = positions['B'][b_indices]
        pos_c1 = positions['C'][c1_indices]
        pos_c2 = positions['C'][c2_indices]
        
        # Distances (explicit sqrt of squared sums)
        ab_dists = np.sqrt(np.sum((pos_a - pos_b)**2, axis=1))
        bc1_dists = np.sqrt(np.sum((pos_b - pos_c1)**2, axis=1))
        bc2_dists = np.sqrt(np.sum((pos_b - pos_c2)**2, axis=1))
        
        # Targets
        ab_target = self.params.pair_distances['AB']
        bc_target = self.params.pair_distances['BC']
        
        # NLL components with normalization
        ab_scores = ((ab_dists - ab_target)**2) / (2 * sig['AB']**2) + 0.5 * np.log(2 * np.pi * sig['AB']**2)
        bc1_scores = ((bc1_dists - bc_target)**2) / (2 * sig['BC']**2) + 0.5 * np.log(2 * np.pi * sig['BC']**2)
        bc2_scores = ((bc2_dists - bc_target)**2) / (2 * sig['BC']**2) + 0.5 * np.log(2 * np.pi * sig['BC']**2)
        
        scores = ab_scores + bc1_scores + bc2_scores
        
        if debug_logging:
            print(f"Tetramer batch scores: mean={np.mean(scores):.3f}, std={np.std(scores):.3f}")
        
        return scores

    def compute_score(self) -> float:
        """
        Sum of tetramer scores over current ABCC assignments.
        """
        tetramers = get_tetramers(self.state)
        if not tetramers:
            return 0.0
        
        tetramer_scores = self.calculate_tetramer_scores_batch(
            positions=self.coordinates,
            tetramers=tetramers,
            sig=self.nuisance_parameters,
            debug_logging=False
        )
        return float(np.sum(tetramer_scores))
#============================================================================