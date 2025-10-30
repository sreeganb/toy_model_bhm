"""
Test to generate RMSD vs Score graphs for different samplers.

This test:
1. Takes the ideal structure from SystemParameters
2. Scores it using different sampler scoring functions
3. Creates perturbations of varying magnitudes
4. Calculates RMSD for each perturbation
5. Scores each perturbation
6. Generates RMSD vs Score graphs for each sampler type
"""

import unittest
import numpy as np
import sys
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.parameters import SystemParameters
from core.state import SystemState
from core.sigma import initialize_sigma_dict, get_default_sigma_ranges, create_sigma_prior

# Import scoring functions from different samplers
from samplers.pair import neg_log_posterior as pair_score
from samplers.tetramer import neg_log_posterior as tetramer_score
from samplers.octet import neg_log_posterior as octet_score


def calculate_rmsd(coords1, coords2):
    """
    Calculate RMSD between two coordinate dictionaries.
    
    Args:
        coords1: Dictionary of {type: np.array of shape (n, 3)}
        coords2: Dictionary of {type: np.array of shape (n, 3)}
    
    Returns:
        float: RMSD value
    """
    squared_diffs = []
    
    for particle_type in coords1.keys():
        if particle_type in coords2:
            diff = coords1[particle_type] - coords2[particle_type]
            squared_diffs.extend((diff ** 2).flatten())
    
    if not squared_diffs:
        return 0.0
    
    return np.sqrt(np.mean(squared_diffs))


def create_perturbation(ideal_coords, perturbation_magnitude, rng=None):
    """
    Create a perturbed version of the ideal coordinates.
    
    Args:
        ideal_coords: Dictionary of ideal coordinates
        perturbation_magnitude: Standard deviation of Gaussian noise to add
        rng: Random number generator (optional)
    
    Returns:
        Dictionary of perturbed coordinates
    """
    if rng is None:
        rng = np.random.default_rng()
    
    perturbed = {}
    for particle_type, coords in ideal_coords.items():
        noise = rng.normal(0, perturbation_magnitude, coords.shape)
        perturbed[particle_type] = coords + noise
    
    return perturbed


class TestRMSDvsScore(unittest.TestCase):
    
    def setUp(self):
        """Set up test data with ideal structure"""
        self.params = SystemParameters()
        self.ideal_coords = self.params.ideal_coordinates
        
        # Setup sigma values
        self.sigma_ranges = get_default_sigma_ranges()
        self.rng = np.random.default_rng(42)
        
        # Use mid-range sigma values for testing
        self.sigma = {}
        for pair_type, (low, high) in self.sigma_ranges.items():
            self.sigma[pair_type] = (low + high) / 2
        
        # Create sampler sequence
        self.sampler_sequence = ["pair_sampling", "tetramer_sampling", "octet_sampling"]
    
    def _create_state(self, coords, sampler_name):
        """Helper to create a SystemState with given coordinates and sampler"""
        state = SystemState(self.sampler_sequence, sampler_name)
        state.box_size = self.params.box_size
        state.update_positions(coords)
        state.sigma = self.sigma.copy()
        
        # Create sigma prior for the state
        pair_types = list(self.sigma_ranges.keys())
        state.sigma_prior = create_sigma_prior(
            pair_types=pair_types,
            sigma_ranges=self.sigma_ranges,
            gmm_file=None,
            prior_type="uniform"
        )
        state.sigma_range = self.sigma_ranges
        
        return state
    
    def test_ideal_structure_scores(self):
        """Test that ideal structure can be scored by all samplers"""
        print("\n" + "="*70)
        print("IDEAL STRUCTURE SCORES")
        print("="*70)
        
        # Score with pair sampler
        pair_state = self._create_state(self.ideal_coords, "pair_sampling")
        pair_total, pair_exvol, pair_pair, pair_prior = pair_score(pair_state)
        print(f"\nPair Sampler:")
        print(f"  Total Score: {pair_total:.3f}")
        print(f"  ExVol: {pair_exvol:.3f}, Pair: {pair_pair:.3f}, Prior: {pair_prior:.3f}")
        
        # Score with tetramer sampler
        tet_state = self._create_state(self.ideal_coords, "tetramer_sampling")
        tet_total, tet_exvol, tet_pair, tet_tet, tet_prior = tetramer_score(tet_state)
        print(f"\nTetramer Sampler:")
        print(f"  Total Score: {tet_total:.3f}")
        print(f"  ExVol: {tet_exvol:.3f}, Pair: {tet_pair:.3f}, Tetramer: {tet_tet:.3f}, Prior: {tet_prior:.3f}")
        
        # Score with octet sampler
        oct_state = self._create_state(self.ideal_coords, "octet_sampling")
        oct_total, oct_exvol, oct_pair, oct_tet, oct_oct, oct_prior = octet_score(oct_state)
        print(f"\nOctet Sampler:")
        print(f"  Total Score: {oct_total:.3f}")
        print(f"  ExVol: {oct_exvol:.3f}, Pair: {oct_pair:.3f}, Tetramer: {oct_tet:.3f}, Octet: {oct_oct:.3f}, Prior: {oct_prior:.3f}")
        
        # All scores should be finite
        self.assertTrue(np.isfinite(pair_total))
        self.assertTrue(np.isfinite(tet_total))
        self.assertTrue(np.isfinite(oct_total))
    
    def test_rmsd_vs_score_pair_sampler(self):
        """Generate RMSD vs Score graph for pair sampler"""
        print("\n" + "="*70)
        print("PAIR SAMPLER: RMSD vs SCORE")
        print("="*70)
        
        # Create perturbations with varying magnitudes
        perturbation_magnitudes = np.linspace(0, 50, 20)
        rmsd_values = []
        score_values = []
        
        for mag in perturbation_magnitudes:
            # Create perturbation
            perturbed_coords = create_perturbation(self.ideal_coords, mag, self.rng)
            
            # Calculate RMSD
            rmsd = calculate_rmsd(self.ideal_coords, perturbed_coords)
            
            # Score the perturbed structure
            state = self._create_state(perturbed_coords, "pair_sampling")
            total_score, _, _, _ = pair_score(state)
            
            rmsd_values.append(rmsd)
            score_values.append(total_score)
            
            if len(rmsd_values) % 5 == 0:
                print(f"  Perturbation mag={mag:.1f}, RMSD={rmsd:.2f}, Score={total_score:.2f}")
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.scatter(rmsd_values, score_values, alpha=0.6, s=50)
        plt.xlabel('RMSD from Ideal Structure (Å)', fontsize=12)
        plt.ylabel('Score (Negative Log Posterior)', fontsize=12)
        plt.title('Pair Sampler: RMSD vs Score', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Save plot
        output_dir = os.path.join(os.path.dirname(__file__), 'output')
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'rmsd_vs_score_pair.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n  Plot saved to: {plot_path}")
        
        # Verify we have data
        self.assertEqual(len(rmsd_values), len(perturbation_magnitudes))
        self.assertEqual(len(score_values), len(perturbation_magnitudes))
        
        # Verify scores generally increase with RMSD (with some tolerance for noise)
        # Just check that the last score is higher than the first
        self.assertGreater(score_values[-1], score_values[0])
    
    def test_rmsd_vs_score_tetramer_sampler(self):
        """Generate RMSD vs Score graph for tetramer sampler"""
        print("\n" + "="*70)
        print("TETRAMER SAMPLER: RMSD vs SCORE")
        print("="*70)
        
        # Create perturbations with varying magnitudes
        perturbation_magnitudes = np.linspace(0, 50, 20)
        rmsd_values = []
        score_values = []
        
        for mag in perturbation_magnitudes:
            # Create perturbation
            perturbed_coords = create_perturbation(self.ideal_coords, mag, self.rng)
            
            # Calculate RMSD
            rmsd = calculate_rmsd(self.ideal_coords, perturbed_coords)
            
            # Score the perturbed structure
            state = self._create_state(perturbed_coords, "tetramer_sampling")
            total_score, _, _, _, _ = tetramer_score(state)
            
            rmsd_values.append(rmsd)
            score_values.append(total_score)
            
            if len(rmsd_values) % 5 == 0:
                print(f"  Perturbation mag={mag:.1f}, RMSD={rmsd:.2f}, Score={total_score:.2f}")
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.scatter(rmsd_values, score_values, alpha=0.6, s=50, color='orange')
        plt.xlabel('RMSD from Ideal Structure (Å)', fontsize=12)
        plt.ylabel('Score (Negative Log Posterior)', fontsize=12)
        plt.title('Tetramer Sampler: RMSD vs Score', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Save plot
        output_dir = os.path.join(os.path.dirname(__file__), 'output')
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'rmsd_vs_score_tetramer.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n  Plot saved to: {plot_path}")
        
        # Verify we have data
        self.assertEqual(len(rmsd_values), len(perturbation_magnitudes))
        self.assertEqual(len(score_values), len(perturbation_magnitudes))
        
        # Verify scores generally increase with RMSD
        self.assertGreater(score_values[-1], score_values[0])
    
    def test_rmsd_vs_score_octet_sampler(self):
        """Generate RMSD vs Score graph for octet sampler"""
        print("\n" + "="*70)
        print("OCTET SAMPLER: RMSD vs SCORE")
        print("="*70)
        
        # Create perturbations with varying magnitudes
        perturbation_magnitudes = np.linspace(0, 50, 20)
        rmsd_values = []
        score_values = []
        
        for mag in perturbation_magnitudes:
            # Create perturbation
            perturbed_coords = create_perturbation(self.ideal_coords, mag, self.rng)
            
            # Calculate RMSD
            rmsd = calculate_rmsd(self.ideal_coords, perturbed_coords)
            
            # Score the perturbed structure
            state = self._create_state(perturbed_coords, "octet_sampling")
            total_score, _, _, _, _, _ = octet_score(state)
            
            rmsd_values.append(rmsd)
            score_values.append(total_score)
            
            if len(rmsd_values) % 5 == 0:
                print(f"  Perturbation mag={mag:.1f}, RMSD={rmsd:.2f}, Score={total_score:.2f}")
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.scatter(rmsd_values, score_values, alpha=0.6, s=50, color='green')
        plt.xlabel('RMSD from Ideal Structure (Å)', fontsize=12)
        plt.ylabel('Score (Negative Log Posterior)', fontsize=12)
        plt.title('Octet Sampler: RMSD vs Score', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Save plot
        output_dir = os.path.join(os.path.dirname(__file__), 'output')
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'rmsd_vs_score_octet.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n  Plot saved to: {plot_path}")
        
        # Verify we have data
        self.assertEqual(len(rmsd_values), len(perturbation_magnitudes))
        self.assertEqual(len(score_values), len(perturbation_magnitudes))
        
        # Verify scores generally increase with RMSD
        self.assertGreater(score_values[-1], score_values[0])
    
    def test_rmsd_vs_score_all_samplers(self):
        """Generate combined RMSD vs Score graph for all samplers"""
        print("\n" + "="*70)
        print("ALL SAMPLERS: RMSD vs SCORE (COMBINED)")
        print("="*70)
        
        # Create perturbations with varying magnitudes
        perturbation_magnitudes = np.linspace(0, 50, 20)
        
        results = {
            'pair': {'rmsd': [], 'scores': [], 'color': 'blue'},
            'tetramer': {'rmsd': [], 'scores': [], 'color': 'orange'},
            'octet': {'rmsd': [], 'scores': [], 'color': 'green'}
        }
        
        for mag in perturbation_magnitudes:
            # Create perturbation
            perturbed_coords = create_perturbation(self.ideal_coords, mag, self.rng)
            
            # Calculate RMSD
            rmsd = calculate_rmsd(self.ideal_coords, perturbed_coords)
            
            # Score with pair sampler
            pair_state = self._create_state(perturbed_coords, "pair_sampling")
            pair_total, _, _, _ = pair_score(pair_state)
            results['pair']['rmsd'].append(rmsd)
            results['pair']['scores'].append(pair_total)
            
            # Score with tetramer sampler
            tet_state = self._create_state(perturbed_coords, "tetramer_sampling")
            tet_total, _, _, _, _ = tetramer_score(tet_state)
            results['tetramer']['rmsd'].append(rmsd)
            results['tetramer']['scores'].append(tet_total)
            
            # Score with octet sampler
            oct_state = self._create_state(perturbed_coords, "octet_sampling")
            oct_total, _, _, _, _, _ = octet_score(oct_state)
            results['octet']['rmsd'].append(rmsd)
            results['octet']['scores'].append(oct_total)
            
            if len(results['pair']['rmsd']) % 5 == 0:
                print(f"  Perturbation mag={mag:.1f}, RMSD={rmsd:.2f}")
                print(f"    Pair: {pair_total:.2f}, Tetramer: {tet_total:.2f}, Octet: {oct_total:.2f}")
        
        # Create combined plot
        plt.figure(figsize=(12, 8))
        
        for sampler_name, data in results.items():
            plt.scatter(data['rmsd'], data['scores'], 
                       alpha=0.6, s=50, color=data['color'], 
                       label=sampler_name.capitalize())
        
        plt.xlabel('RMSD from Ideal Structure (Å)', fontsize=12)
        plt.ylabel('Score (Negative Log Posterior)', fontsize=12)
        plt.title('RMSD vs Score for All Samplers', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Save plot
        output_dir = os.path.join(os.path.dirname(__file__), 'output')
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'rmsd_vs_score_all_samplers.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n  Combined plot saved to: {plot_path}")
        
        # Verify all data is collected
        for sampler_name in results.keys():
            self.assertEqual(len(results[sampler_name]['rmsd']), len(perturbation_magnitudes))
            self.assertEqual(len(results[sampler_name]['scores']), len(perturbation_magnitudes))


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
