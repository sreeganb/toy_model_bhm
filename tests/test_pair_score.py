import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoring.pair_score import PairNLL
from core.state import SystemState
from core.parameters import SystemParameters

class TestPairNLL(unittest.TestCase):
    
    def setUp(self):
        """Set up test data with known coordinates and parameters"""
        # Create simple 2D coordinates for easier manual calculation
        # A particles: 2 particles
        self.coords_A = np.array([
            [0.0, 0.0, 0.0],      # A0 at origin
            [120.0, 90.0, 0.0]    # A1 at (120,90,0) -> distance from A0 = 150.0
        ])
        
        # B particles: 2 particles  
        self.coords_B = np.array([
            [70.0, 150.0, 0.0],   # B0 at (70,150,0)
            [-60.0, 15.0, 0.0]    # B1 at (-60,15,0)
        ])
        
        self.coordinates = {
            'A': self.coords_A,
            'B': self.coords_B
        }
        
        # Nuisance parameters (sigma values)
        self.nuisance_params = {
            'AA': 2.0,  # sigma for A-A pairs
            'AB': 1.0   # sigma for A-B pairs
        }
        
        # Expected distances (from SystemParameters)
        self.expected_distances = {
            'AA': 48.22,  # Expected A-A distance
            'AB': 38.5    # Expected A-B distance
        }
        
    def test_simple_pair_calculation(self):
        """Test pair NLL calculation with manually computed expected values"""
        
        # Create PairNLL instance
        pair_nll = PairNLL(self.coordinates, self.nuisance_params)
        # Override expected distances for easier manual calculation
        pair_nll.pair_distances = self.expected_distances
        
        # Manual calculations:
        # 1. A-A pairs (only upper triangle: A0-A1)
        d_A0_A1 = 150.0  # Distance between A0 and A1
        d0_AA = 48.22
        sigma_AA = 2.0
        nll_A0_A1 = (d_A0_A1 - d0_AA)**2 / (2 * sigma_AA**2) + np.log(sigma_AA * np.sqrt(2 * np.pi))
        expected_AA_nll = nll_A0_A1
        
        # 2. A-B pairs (all combinations, argmin selection)
        # Compute exact distances
        d_A0_B0 = np.sqrt((0-70)**2 + (0-150)**2 + 0**2)      # sqrt(70^2 + 150^2) = 165.53
        d_A0_B1 = np.sqrt((0-(-60))**2 + (0-15)**2 + 0**2)    # sqrt(60^2 + 15^2) = 61.85
        d_A1_B0 = np.sqrt((120-70)**2 + (90-150)**2 + 0**2)   # sqrt(50^2 + 60^2) = 78.10
        d_A1_B1 = np.sqrt((120-(-60))**2 + (90-15)**2 + 0**2) # sqrt(180^2 + 75^2) = 195.00

        d0_AB = 38.5
        sigma_AB = 1.0  # FIXED: was 2.0, should match setUp

        # NLL matrix for A-B pairs:
        nll_A0_B0 = (d_A0_B0 - d0_AB)**2 / (2 * sigma_AB**2) + np.log(sigma_AB * np.sqrt(2 * np.pi))
        nll_A0_B1 = (d_A0_B1 - d0_AB)**2 / (2 * sigma_AB**2) + np.log(sigma_AB * np.sqrt(2 * np.pi))
        nll_A1_B0 = (d_A1_B0 - d0_AB)**2 / (2 * sigma_AB**2) + np.log(sigma_AB * np.sqrt(2 * np.pi))
        nll_A1_B1 = (d_A1_B1 - d0_AB)**2 / (2 * sigma_AB**2) + np.log(sigma_AB * np.sqrt(2 * np.pi))
        
        # Print for debugging
        print(f"\n{'='*60}")
        print(f"Manual calculations:")
        print(f"{'='*60}")
        print(f"AA pair:")
        print(f"  d_A0_A1 = {d_A0_A1:.2f}, nll_A0_A1 = {nll_A0_A1:.3f}")
        print(f"\nAB distances:")
        print(f"  d_A0_B0 = {d_A0_B0:.2f}, d_A0_B1 = {d_A0_B1:.2f}")
        print(f"  d_A1_B0 = {d_A1_B0:.2f}, d_A1_B1 = {d_A1_B1:.2f}")
        print(f"\nAB NLL values:")
        print(f"  nll_A0_B0 = {nll_A0_B0:.3f}, nll_A0_B1 = {nll_A0_B1:.3f}")
        print(f"  nll_A1_B0 = {nll_A1_B0:.3f}, nll_A1_B1 = {nll_A1_B1:.3f}")
        
        # Determine selected pairs by argmin logic:
        # Row argmin: each A selects its best B
        a0_best_b = 0 if nll_A0_B0 < nll_A0_B1 else 1
        a1_best_b = 0 if nll_A1_B0 < nll_A1_B1 else 1
        
        # Column argmin: each B selects its best A
        b0_best_a = 0 if nll_A0_B0 < nll_A1_B0 else 1
        b1_best_a = 0 if nll_A0_B1 < nll_A1_B1 else 1
        
        print(f"\nArgmin selections:")
        print(f"  A0 picks B{a0_best_b} (row 0 min)")
        print(f"  A1 picks B{a1_best_b} (row 1 min)")
        print(f"  B0 picks A{b0_best_a} (col 0 min)")
        print(f"  B1 picks A{b1_best_a} (col 1 min)")
        
        # Union of selected pairs (removing duplicates)
        selected_ab_pairs = set()
        selected_ab_pairs.add(('A', 0, 'B', a0_best_b))
        selected_ab_pairs.add(('A', 1, 'B', a1_best_b))
        selected_ab_pairs.add(('A', b0_best_a, 'B', 0))
        selected_ab_pairs.add(('A', b1_best_a, 'B', 1))
        
        # Compute expected AB NLL sum
        expected_AB_nll = 0.0
        ab_nll_values = {
            (0, 0): nll_A0_B0, (0, 1): nll_A0_B1,
            (1, 0): nll_A1_B0, (1, 1): nll_A1_B1
        }
        
        print(f"\nSelected AB pairs (union): {selected_ab_pairs}")
        for t1, i, t2, j in selected_ab_pairs:
            nll_val = ab_nll_values[(i, j)]
            expected_AB_nll += nll_val
            print(f"  {t1}{i}-{t2}{j}: nll = {nll_val:.3f}")
        
        expected_total_nll = expected_AA_nll + expected_AB_nll
        
        print(f"\nExpected scores:")
        print(f"  AA NLL: {expected_AA_nll:.3f}")
        print(f"  AB NLL: {expected_AB_nll:.3f}")
        print(f"  Total NLL: {expected_total_nll:.3f}")
        
        # Compute using the class
        computed_nll = pair_nll.compute_score()
        
        print(f"\nComputed NLL: {computed_nll:.3f}")
        print(f"Difference: {abs(computed_nll - expected_total_nll):.6f}")
        print(f"{'='*60}\n")
        
        # Test with reasonable tolerance (floating point precision)
        self.assertAlmostEqual(computed_nll, expected_total_nll, places=5,
                              msg=f"Expected {expected_total_nll:.6f}, got {computed_nll:.6f}")
    
    def test_empty_coordinates(self):
        """Test behavior with empty coordinate arrays"""
        empty_coords = {'A': np.array([]).reshape(0, 3), 'B': np.array([]).reshape(0, 3)}
        pair_nll = PairNLL(empty_coords, self.nuisance_params)
        score = pair_nll.compute_score()
        self.assertEqual(score, 0.0, "Empty coordinates should give zero score")
    
    def test_single_particle_per_type(self):
        """Test with only one particle per type"""
        single_coords = {
            'A': np.array([[0.0, 0.0, 0.0]]),
            'B': np.array([[1.0, 0.0, 0.0]])
        }
        
        pair_nll = PairNLL(single_coords, self.nuisance_params)
        pair_nll.pair_distances = self.expected_distances
        
        # Should only compute AB pair (no AA pairs possible with single A)
        score = pair_nll.compute_score()
        
        # Manual calculation for single AB pair
        d_AB = 1.0
        d0_AB = 38.5
        sigma_AB = 1.0  # FIXED: was 2.0
        expected_score = (d_AB - d0_AB)**2 / (2 * sigma_AB**2) + np.log(sigma_AB * np.sqrt(2 * np.pi))
        
        print(f"\nSingle particle test: expected={expected_score:.3f}, computed={score:.3f}")
        self.assertAlmostEqual(score, expected_score, places=5)
    
    def test_same_type_pairs_only(self):
        """Test scoring with only same-type particles (AA pairs only)"""
        aa_coords = {'A': self.coords_A}
        aa_params = {'AA': 2.0}
        
        pair_nll = PairNLL(aa_coords, aa_params)
        pair_nll.pair_distances = {'AA': 48.22}
        
        score = pair_nll.compute_score()
        
        # Should only have one AA pair: A0-A1
        d_A0_A1 = 150.0
        d0_AA = 48.22
        sigma_AA = 2.0
        expected_score = (d_A0_A1 - d0_AA)**2 / (2 * sigma_AA**2) + np.log(sigma_AA * np.sqrt(2 * np.pi))
        
        print(f"\nSame-type test: expected={expected_score:.3f}, computed={score:.3f}")
        self.assertAlmostEqual(score, expected_score, places=5)
    
    def test_symmetric_configuration(self):
        """Test with symmetric particle placement to verify pair selection logic"""
        # Place particles in a square configuration
        symmetric_coords = {
            'A': np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]]),
            'B': np.array([[0.0, 100.0, 0.0], [100.0, 100.0, 0.0]])
        }
        
        pair_nll = PairNLL(symmetric_coords, self.nuisance_params)
        pair_nll.pair_distances = self.expected_distances
        
        # In this symmetric case:
        # A0-A1 distance = 100
        # A0-B0 = 100, A0-B1 = 141.42
        # A1-B0 = 141.42, A1-B1 = 100
        # Each A should pick the closest B (diagonal pairs)
        
        score = pair_nll.compute_score()
        
        # Manual calculation
        d_A0_A1 = 100.0
        sigma_AA = 2.0
        d0_AA = 48.22
        nll_AA = (d_A0_A1 - d0_AA)**2 / (2 * sigma_AA**2) + np.log(sigma_AA * np.sqrt(2 * np.pi))
        
        d_close = 100.0  # A0-B0 and A1-B1
        sigma_AB = 1.0
        d0_AB = 38.5
        nll_close = (d_close - d0_AB)**2 / (2 * sigma_AB**2) + np.log(sigma_AB * np.sqrt(2 * np.pi))
        
        # Should select A0-B0 and A1-B1 (2 pairs, but counted once each direction)
        expected_symmetric = nll_AA + 2 * nll_close
        
        print(f"\nSymmetric test: expected={expected_symmetric:.3f}, computed={score:.3f}")
        self.assertAlmostEqual(score, expected_symmetric, places=5)

if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)