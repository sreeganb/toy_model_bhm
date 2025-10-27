import numpy as np
from core.parameters import SystemParameters

#==============================================================================
# Excluded volume negative log-likelihood score
#==============================================================================

class ExvolNLL():
    def __init__(self, coordinates_dict, kappa):
        """
        Excluded volume penalty is computed for all given particles with coordinates 
        and radii. These are spherical particles. First compute the overlap between 
        particle pairs as overlap_ij = r_i + r_j - d_ij, where d_ij is the distance
        between particles i and j. If overlap_ij > 0, then there is an overlap, 
        otherwise no overlap. The excluded volume gets a squared penalty with a 
        parameter kappa: (overlap_ij)^2 * kappa is the penalty for the pair. 
        The total excluded volume penalty is the sum of all pair penalties.
        
        Args:
            coordinates_dict: Dict with particle types as keys and coordinate arrays as values
            radii_dict: Dict with particle types as keys and radius values
            kappa: Penalty parameter for overlaps
        """
        self.coordinates = coordinates_dict
        self.params = SystemParameters()
        self.radii_dict = self.params.radii
        self.kappa = kappa
        
        # Generate all unique pair types
        particle_types = list(coordinates_dict.keys())
        self.pair_types = []
        for i, type1 in enumerate(particle_types):
            for j, type2 in enumerate(particle_types):
                if i <= j:  # Only consider unique pairs and same-type pairs
                    self.pair_types.append((type1, type2))
        
    def compute_score(self):
        """
        Vectorized computation of excluded volume penalty with proper handling 
        of same vs different particle types.
        """
        total_penalty = 0.0
        
        for pair_type in self.pair_types:
            type1, type2 = pair_type[0], pair_type[1]
            coords1, coords2 = self.coordinates[type1], self.coordinates[type2]
            n1, n2 = coords1.shape[0], coords2.shape[0]
            
            # Get radii for this pair type
            radius1, radius2 = self.radii_dict[type1], self.radii_dict[type2]
            
            # Vectorized distance computation
            # coords1: (n1, 3), coords2: (n2, 3)
            # Broadcast to (n1, n2, 3) then compute distances
            diff = coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :]  # (n1, n2, 3)
            distances = np.linalg.norm(diff, axis=2)  # (n1, n2)
            
            # Compute overlaps for all pairs
            sum_radii = radius1 + radius2
            overlaps = sum_radii - distances  # (n1, n2)
            
            # Only consider positive overlaps (actual collisions)
            collision_mask = overlaps > 0
            
            if type1 == type2:
                # Same particle type: avoid self-interaction (i == j) and 
                # double-counting by only considering upper triangle (j > i)
                upper_triangle_mask = np.triu(np.ones((n1, n2), dtype=bool), k=1)
                valid_mask = collision_mask & upper_triangle_mask
            else:
                # Different particle types: all pairings are valid
                valid_mask = collision_mask
            
            # Extract valid overlaps and compute penalties
            valid_overlaps = overlaps[valid_mask]
            if len(valid_overlaps) > 0:
                penalties = self.kappa * (valid_overlaps ** 2)
                total_penalty += np.sum(penalties)
        
        return total_penalty

def main():
    """Test function to verify correctness of the ExvolNLL implementation."""
    
    print("Testing ExvolNLL implementation...")
    
    # Test Case 1: No overlaps - should give zero penalty
    print("\n1. Test Case: No overlaps")
    coordinates_no_overlap = {
        'A': np.array([[0, 0, 0], [10, 0, 0]]),
        'B': np.array([[0, 10, 0], [10, 10, 0]])
    }
    radii_no_overlap = {'A': 1.0, 'B': 1.0}
    
    evol_no_overlap = ExvolNLL(coordinates_no_overlap, radii_no_overlap, kappa=100.0)
    penalty_no_overlap = evol_no_overlap.compute_score()
    print(f"Penalty (should be 0): {penalty_no_overlap}")
    
    # Test Case 2: Same type overlaps
    print("\n2. Test Case: Same type overlaps")
    coordinates_same_overlap = {
        'A': np.array([[0, 0, 0], [1.5, 0, 0], [3, 0, 0]]),  # A particles with overlaps
        'B': np.array([[0, 10, 0]])  # B particle far away
    }
    radii_same_overlap = {'A': 1.0, 'B': 1.0}
    
    evol_same_overlap = ExvolNLL(coordinates_same_overlap, radii_same_overlap, kappa=100.0)
    penalty_same_overlap = evol_same_overlap.compute_score()
    print(f"Penalty from same-type overlaps: {penalty_same_overlap}")
    
    # Manual calculation for verification
    # Distance between A[0] and A[1]: 1.5, sum of radii: 2.0, overlap: 0.5
    # Distance between A[1] and A[2]: 1.5, sum of radii: 2.0, overlap: 0.5
    # Expected penalty: 100 * (0.5^2 + 0.5^2) = 100 * 0.5 = 50.0
    expected_penalty_same = 100.0 * (0.5**2 + 0.5**2)
    print(f"Expected penalty: {expected_penalty_same}")
    
    # Test Case 3: Different type overlaps
    print("\n3. Test Case: Different type overlaps")
    coordinates_diff_overlap = {
        'A': np.array([[0, 0, 0]]),
        'B': np.array([[1.0, 0, 0]])  # B particle overlapping with A
    }
    radii_diff_overlap = {'A': 1.0, 'B': 1.0}
    
    evol_diff_overlap = ExvolNLL(coordinates_diff_overlap, radii_diff_overlap, kappa=100.0)
    penalty_diff_overlap = evol_diff_overlap.compute_score()
    print(f"Penalty from different-type overlap: {penalty_diff_overlap}")
    
    # Manual calculation
    # Distance: 1.0, sum of radii: 2.0, overlap: 1.0
    # Expected penalty: 100 * 1.0^2 = 100.0
    expected_penalty_diff = 100.0 * (1.0**2)
    print(f"Expected penalty: {expected_penalty_diff}")
    
    # Test Case 4: Mixed scenario
    print("\n4. Test Case: Mixed overlaps")
    coordinates_mixed = {
        'A': np.array([[0, 0, 0], [1.8, 0, 0]]),  # Two A particles with slight overlap
        'B': np.array([[0.5, 0, 0]]),  # B particle overlapping with first A
        'C': np.array([[20, 0, 0]])   # C particle far away (no overlaps)
    }
    radii_mixed = {'A': 1.0, 'B': 0.8, 'C': 1.0}
    
    evol_mixed = ExvolNLL(coordinates_mixed, radii_mixed, kappa=50.0)
    penalty_mixed = evol_mixed.compute_score()
    print(f"Penalty from mixed scenario: {penalty_mixed}")
    
    # Manual calculation
    # A-A overlap: distance=1.8, sum_radii=2.0, overlap=0.2, penalty=50*(0.2^2)=2.0
    # A-B overlap: distance=0.5, sum_radii=1.8, overlap=1.3, penalty=50*(1.3^2)=84.5
    # B-A overlap: same as A-B (but counted only once due to different types)
    expected_penalty_mixed = 50.0 * (0.2**2 + 1.3**2)
    print(f"Expected penalty: {expected_penalty_mixed}")
    
    # Test Case 5: Compare with original implementation
    print("\n5. Test Case: Comparison with original implementation")

if __name__ == "__main__":
    main()