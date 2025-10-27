import numpy as np
from core.parameters import SystemParameters

#==================================================================================
# Pair scoring function, agnostic to sampler, takes in coordinates, nuisance parameter
# as input and computes the pair score
#==================================================================================

class PairNLL:
    def __init__(self, coordinates, nuisance_parameters):
        """
        coordinates: dictionary {A: np.array, B: np.array, C: np.array}
        nuisance_parameters: dictionary {AA: value, AB: value, BC: value}
        pair distances are computed for AA particles, these are same type particles so
        diagonals are self interactions
        pair distance score is 1/sqrt(2*pi*sigma**2) * exp(-(d-d0)**2/(2*sigma**2))
        """
        self.coordinates = coordinates
        self.nuisance_parameters = nuisance_parameters
        self.pair_types = list(nuisance_parameters.keys())
        params = SystemParameters()
        self.pair_distances = params.pair_distances  # Expected distances for each pair type
    
    def compute_score(self):
        """
        Compute negative log-likelihood scores for particle pairs and select
        the most likely pairings using a union-of-argmin strategy.
        
        For each particle type pair, we:
        1. Compute -log(likelihood) for all possible pairs based on distance
        2. For same-type particles, only consider upper triangle (j > i) to avoid
        self-interactions and double-counting
        3. Select pairs where each particle finds its best partner (lowest NLL)
        using both row-wise and column-wise minimum searches
        
        Returns:
            float: Total negative log-likelihood score across all selected pairs
        """
        pair_distance_scores = {}
        total_nll = 0.0
        
        for pair_type in self.pair_types:
            type1, type2 = pair_type[0], pair_type[1]
            coords1, coords2 = self.coordinates[type1], self.coordinates[type2]
            n1, n2 = coords1.shape[0], coords2.shape[0]
            
            d0 = self.pair_distances[pair_type]  # Expected distance for this pair type
            sigma = self.nuisance_parameters[pair_type]  # Distance uncertainty
            
            # Vectorized distance computation
            # coords1: (n1, 3), coords2: (n2, 3)
            diff = coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :]  # (n1, n2, 3)
            distances = np.linalg.norm(diff, axis=2)  # (n1, n2)
            
            # Compute negative log-likelihood matrix (vectorized)
            nll_matrix = (distances - d0)**2 / (2 * sigma**2) + np.log(sigma * np.sqrt(2 * np.pi))
            
            selected_pairs = set()
            
            if type1 == type2:
                # Same particle type: avoid self-interaction (i == j) and 
                # double-counting by only considering upper triangle (j > i)
                
                # Row-wise minima: for each particle i, find its best partner j where j > i
                for i in range(n1):
                    valid_cols = list(range(i + 1, n2))  # Only consider j > i
                    if valid_cols:
                        # Extract NLL scores for valid columns only
                        valid_nll = nll_matrix[i, valid_cols]
                        # Find the column with minimum NLL
                        best_local_idx = np.argmin(valid_nll)
                        j_idx = valid_cols[best_local_idx]  # Map back to original column index
                        selected_pairs.add((type1, i, type2, j_idx))
                
                # Column-wise minima: for each particle j, find its best partner i where i < j
                for j in range(n2):
                    valid_rows = list(range(0, j))  # Only consider i < j
                    if valid_rows:
                        # Extract NLL scores for valid rows only
                        valid_nll = nll_matrix[valid_rows, j]
                        # Find the row with minimum NLL
                        best_local_idx = np.argmin(valid_nll)
                        i_idx = valid_rows[best_local_idx]  # Map back to original row index
                        selected_pairs.add((type1, i_idx, type2, j))
                
                # Note: Using a set automatically handles duplicates. If particle i picks j
                # as its best partner AND particle j picks i as its best partner, the pair
                # (i, j) will only appear once in the set.
            
            else:
                # Different particle types: all pairings are valid, no symmetry issues
                
                # Row-wise minima: for each type1 particle, find its best type2 partner
                for i in range(n1):
                    j_idx = np.argmin(nll_matrix[i, :])
                    selected_pairs.add((type1, i, type2, j_idx))
                
                # Column-wise minima: for each type2 particle, find its best type1 partner
                for j in range(n2):
                    i_idx = np.argmin(nll_matrix[:, j])
                    selected_pairs.add((type1, i_idx, type2, j))
            
            # Store selected pairs
            pair_distance_scores[pair_type] = selected_pairs
            
            # Sum the NLL values for all selected pairs
            pair_nll_sum = 0.0
            for t1, i, t2, j in selected_pairs:
                pair_nll_sum += nll_matrix[i, j]
            
            total_nll += pair_nll_sum
        
        return total_nll