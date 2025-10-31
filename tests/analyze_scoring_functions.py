"""
Score vs RMSD Analysis Tool

Analyzes scoring functions by:
1. Loading structures from trajectory files OR generating perturbations
2. Computing scores using all available scoring functions
3. Calculating RMSD metrics
4. Saving results for visualization

Usage:
    python tests/analyze_scoring_functions.py [trajectory_file.h5]
"""

import numpy as np
import os
import sys
import h5py
from pathlib import Path
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from core.parameters import SystemParameters
from core.state import SystemState
from core.sigma import initialize_sigma_dict
from scoring.pair_score import PairNLL
from scoring.tetramer_score import TetramerNLL
from scoring.octet_score import OctetNLL
from scoring.exvol_score import ExvolNLL
from scoring.full_score import FullNLL


class PerturbationAnalyzer:
    """
    Generates and analyzes molecular perturbations with comprehensive scoring.
    """
    
# In the __init__ method, after getting ideal coordinates (around line 70):

    def __init__(self, n_target=200, jitter_range=(0.1, 25.0), max_total_score=10000.0, 
                 trajectory_file=None, em_map_file=None):
        """
        Initialize analyzer with configuration parameters.
        
        Args:
            n_target: Number of structures to analyze/generate
            jitter_range: (min, max) range for perturbation intensity
            max_total_score: Maximum acceptable score for perturbations
            trajectory_file: Optional path to existing trajectory
            em_map_file: Optional path to EM density map for CCC scoring
        """
        # Configuration
        self.n_target = n_target
        self.jitter_range = jitter_range
        self.max_total_score = max_total_score
        self.incremental_steps = 5
        self.overlap_tolerance = 1.0
        self.max_relax_iters = 120
        self.pair_buffer = 0.25
        self.trajectory_file = trajectory_file
        self.em_map_file = em_map_file
        
        # System parameters
        self.params = SystemParameters()
        
        # Get ideal coordinates
        ideal_coords = self.params.latest_ideal()
        self.array_A = ideal_coords['A']
        self.array_B = ideal_coords['B']
        self.array_C = ideal_coords['C']
        self.n_A, self.n_B, self.n_C = len(self.array_A), len(self.array_B), len(self.array_C)
        
        # IMPORTANT: Get actual radii from parameters (not defaults!)
        self.radii = {
            'A': self.params.radii['A'],
            'B': self.params.radii['B'],
            'C': self.params.radii['C']
        }
        print(f"Using radii: A={self.radii['A']}, B={self.radii['B']}, C={self.radii['C']}")
        
        # Analysis data
        self.ref_coords = None
        self.results = []
        
        # Initialize sigma values
        self.sigma = initialize_sigma_dict()
        
        # Excluded volume kappa parameter
        self.exvol_kappa = 100.0

    def initialize_system(self):
        """Initialize reference coordinates."""
        self.ref_coords = np.vstack([self.array_A, self.array_B, self.array_C])
        print(f"Reference system initialized:")
        print(f"  A particles: {self.n_A}")
        print(f"  B particles: {self.n_B}")
        print(f"  C particles: {self.n_C}")
        print(f"  Total: {len(self.ref_coords)}")
        
    def load_trajectory_structures(self):
        """Load structures from an existing trajectory file, sorted by step number."""
        if not self.trajectory_file or not os.path.exists(self.trajectory_file):
            return []
            
        print(f"Loading structures from trajectory: {self.trajectory_file}")
        structures = []
        
        with h5py.File(self.trajectory_file, 'r') as f:
            if 'trajectory' not in f:
                print("Warning: No 'trajectory' group found in file")
                return []
                
            traj_grp = f['trajectory']
            
            # Get all state groups - sort by step number
            state_groups = []
            for state_name in traj_grp.keys():
                if state_name.startswith('state_'):
                    state_grp = traj_grp[state_name]
                    step = int(state_grp.attrs.get('step', 0))
                    state_groups.append((step, state_name, state_grp))
            
            # Sort by actual step number
            state_groups.sort(key=lambda x: x[0])
            
            print(f"Found {len(state_groups)} states in trajectory")
            
            for step, state_name, state_grp in state_groups:
                # Load positions
                positions = {}
                if 'positions' in state_grp:
                    pos_grp = state_grp['positions']
                    for comp in ['A', 'B', 'C']:
                        if comp in pos_grp:
                            dataset = pos_grp[comp]
                            if dataset.size > 0:
                                positions[comp] = np.array(dataset[:])
                
                # Load sigma values if available
                sigma_values = self.sigma.copy()
                if 'sigma' in state_grp:
                    sigma_grp = state_grp['sigma']
                    for key in sigma_grp.attrs:
                        sigma_values[key] = float(sigma_grp.attrs[key])
                
                # Only include if we have all required position data
                if len(positions) == 3 and all(comp in positions for comp in ['A', 'B', 'C']):
                    # Combine coordinates
                    coords = np.vstack([positions['A'], positions['B'], positions['C']])
                    
                    structure_data = {
                        'coordinates': coords,
                        'positions': positions,
                        'sigma': sigma_values,
                        'step': step
                    }
                    structures.append(structure_data)
                else:
                    print(f"Warning: Skipping state {state_name} due to incomplete position data")
                    
        print(f"Successfully loaded {len(structures)} structures from trajectory")
        return structures
        
    def generate_perturbation(self, intensity):
        """Generate a perturbed structure with given jitter intensity."""
        step_sigma = intensity / np.sqrt(self.incremental_steps)
        coords = self.ref_coords.copy()
        
        for _ in range(self.incremental_steps):
            coords += np.random.normal(scale=step_sigma, size=coords.shape)
            
        return self._resolve_clashes(coords)
        
    def _resolve_clashes(self, coords):
        """Resolve overlapping particles using iterative pair separation."""
        coords = coords.copy()
        
        # Create radii array
        radii_array = np.concatenate([
            np.full(self.n_A, self.params.radii['A']),
            np.full(self.n_B, self.params.radii['B']), 
            np.full(self.n_C, self.params.radii['C'])
        ])
        
        for _ in range(self.max_relax_iters):
            clashes = self._find_clashes(coords, radii_array)
            if not clashes:
                break
            coords = self._separate_clashing_pairs(coords, clashes)
            
        return coords
        
    def _find_clashes(self, coords, radii_array):
        """Identify clashing particle pairs."""
        distances = cdist(coords, coords)
        np.fill_diagonal(distances, np.inf)
        min_allowed = (radii_array[:, None] + radii_array) * self.overlap_tolerance
        
        clash_mask = distances < min_allowed
        if not np.any(clash_mask):
            return None
            
        return {
            'mask': clash_mask,
            'distances': distances, 
            'min_allowed': min_allowed
        }
        
    def _separate_clashing_pairs(self, coords, clashes):
        """Separate clashing pairs by moving them apart."""
        coords = coords.copy()
        i_idx, j_idx = np.where(clashes['mask'])
        processed = set()
        
        for i, j in zip(i_idx, j_idx):
            if i >= j or (i, j) in processed:
                continue
                
            distance = clashes['distances'][i, j]
            if distance < 1e-8:
                direction = np.random.normal(size=3)
                direction /= np.linalg.norm(direction)
            else:
                direction = (coords[j] - coords[i]) / distance
                
            needed_separation = (clashes['min_allowed'][i, j] + self.pair_buffer) - distance
            if needed_separation > 0:
                shift = 0.5 * needed_separation
                coords[i] -= direction * shift
                coords[j] += direction * shift
                
            processed.add((i, j))
            
        return coords
        
    def calculate_scores(self, positions, sigma_values=None):
        """
        Calculate ALL scoring functions for given coordinates.
        
        Args:
            positions: Dict with keys 'A', 'B', 'C' mapping to position arrays
            sigma_values: Optional dict of sigma values (uses default if None)
            
        Returns:
            Dict of score values
        """
        if sigma_values is None:
            sigma_values = self.sigma
        
        # Create SystemState for scoring
        state = SystemState()
        state.positions = positions
        state.sigma = sigma_values
        
        scores = {}
        
        # 1. Pair Score (includes pair distances + prior on sigma)
        pair_scorer = PairNLL(positions, sigma_values)
        scores['pair'] = pair_scorer.compute_score()
        
        # 2. Excluded Volume Score
        exvol_scorer = ExvolNLL(positions, self.exvol_kappa)
        scores['exvol'] = exvol_scorer.compute_score()
        
        # 3. Tetramer Score
        tetramer_scorer = TetramerNLL(state)
        scores['tetramer'] = tetramer_scorer.compute_score()
        
        # 4. Octet Score
        octet_scorer = OctetNLL(state)
        scores['octet'] = octet_scorer.compute_score()
        
        # 5. EM Density Score (if map file provided)
        if self.em_map_file and os.path.exists(self.em_map_file):
            try:
                # Create a minimal state with proper radii
                state.params = self.params  # Give it access to correct radii
                
                em_scorer = FullNLL(state, self.em_map_file, resolution=50.0, backend='cpu')
                ccc = em_scorer.calculate_ccc(positions, debug_logging=True)  # Enable debug for reference
                scores['ccc'] = ccc
                scores['em_score'] = 1.0 - ccc  # FIXED: Use 1 - CCC
                
                # Turn off debug after first call
                if hasattr(self, '_em_debug_done'):
                    em_scorer.calculate_ccc(positions, debug_logging=False)
                else:
                    self._em_debug_done = True
                    
            except Exception as e:
                print(f"Warning: Could not calculate EM score: {e}")
                import traceback
                traceback.print_exc()
                scores['ccc'] = 0.0
                scores['em_score'] = 2.0
        else:
            scores['ccc'] = 0.0
            scores['em_score'] = 2.0
        
        # 6. Combined scores
        scores['pair_exvol'] = scores['pair'] + scores['exvol']
        scores['tetramer_exvol'] = scores['tetramer'] + scores['exvol']
        scores['octet_exvol'] = scores['octet'] + scores['exvol']
        
        return scores
                
    def calculate_rmsd_metrics(self, coords):
        """Calculate both raw and aligned RMSD values."""
        raw_rmsd = np.sqrt(np.mean(np.sum((self.ref_coords - coords)**2, axis=1)))
        aligned_rmsd = self._calculate_aligned_rmsd(coords)
        return raw_rmsd, aligned_rmsd
        
    def _calculate_aligned_rmsd(self, coords):
        """Calculate RMSD after optimal alignment using Kabsch algorithm."""
        ref_centered = self.ref_coords - self.ref_coords.mean(0)
        coords_centered = coords - coords.mean(0)
        
        H = coords_centered.T @ ref_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
            
        aligned_coords = (R @ coords_centered.T).T + self.ref_coords.mean(0)
        return np.sqrt(np.mean(np.sum((self.ref_coords - aligned_coords)**2, axis=1)))
        
    def is_structure_acceptable(self, all_scores):
        """Determine if structure meets acceptance criteria based on any score."""
        acceptable_scores = [
            all_scores.get('pair_exvol', np.inf),
            all_scores.get('tetramer_exvol', np.inf),
            all_scores.get('octet_exvol', np.inf)
        ]
        return any(score <= self.max_total_score for score in acceptable_scores)
        
    def create_result_record(self, coords, positions, all_scores, rmsd_metrics, sigma_values,
                           intensity, attempt_num, is_reference=False, trajectory_step=None):
        """Create a standardized result record with ALL scores."""
        raw_rmsd, aligned_rmsd = rmsd_metrics
        
        record = {
            'coords_A': positions['A'],
            'coords_B': positions['B'], 
            'coords_C': positions['C'],
            'rmsd': aligned_rmsd,
            'rmsd_aligned': aligned_rmsd,
            'rmsd_raw': raw_rmsd,
            'intensity_requested': intensity if not is_reference else 0.0,
            'attempt_index': attempt_num,
            'is_reference': is_reference,
            'from_trajectory': trajectory_step is not None
        }
        
        # Add trajectory step if available
        if trajectory_step is not None:
            record['trajectory_step'] = trajectory_step
        
        # Add sigma values
        for key, val in sigma_values.items():
            record[f'sigma_{key}'] = val
        
        # Add all score types
        for score_name, score_value in all_scores.items():
            record[f'score_{score_name}'] = score_value
            
        return record
        
    def run_analysis(self):
        """Execute the complete perturbation analysis."""
        print("=" * 70)
        print("Perturbation Analysis - Multi-Score Evaluation")
        print("=" * 70)
        
        self.initialize_system()
        
        # ALWAYS start with reference structure as frame 0
        ref_positions = {
            'A': self.array_A,
            'B': self.array_B,
            'C': self.array_C
        }
        ref_scores = self.calculate_scores(ref_positions, self.sigma)
        ref_rmsd_metrics = self.calculate_rmsd_metrics(self.ref_coords)
        ref_record = self.create_result_record(
            self.ref_coords, ref_positions, ref_scores, ref_rmsd_metrics, self.sigma,
            intensity=0.0, attempt_num=0, is_reference=True
        )
        self.results = [ref_record]
        
        self._print_reference_info(ref_scores)
        
        # Check if we should load from trajectory or generate perturbations
        if self.trajectory_file:
            self._analyze_trajectory()
        else:
            self._generate_perturbations()
            
        self._print_final_summary()
        
    def _analyze_trajectory(self):
        """Analyze structures from trajectory file."""
        print(f"\nMODE: Analyzing existing trajectory")
        print(f"File: {self.trajectory_file}")
        
        structures = self.load_trajectory_structures()
        if not structures:
            print("Warning: No structures loaded from trajectory file")
            return
            
        # Limit to n_target structures if we have more
        if len(structures) > self.n_target:
            print(f"Limiting analysis to last {self.n_target} structures")
            structures = structures[-self.n_target:]
            
        processed = 0
        for i, structure_data in enumerate(structures):
            coords = structure_data['coordinates']
            positions = structure_data['positions']
            sigma_values = structure_data['sigma']
            step = structure_data['step']
            
            # Calculate scores
            all_scores = self.calculate_scores(positions, sigma_values)
            rmsd_metrics = self.calculate_rmsd_metrics(coords)
            
            record = self.create_result_record(
                coords, positions, all_scores, rmsd_metrics, sigma_values,
                intensity=0.0, attempt_num=i+1, is_reference=False,
                trajectory_step=step
            )
            self.results.append(record)
            processed += 1
            
            if (processed) % 50 == 0:
                self._log_trajectory_acceptance(record, processed, step)
                
        print(f"\nProcessed {processed} structures from trajectory")
        
    def _generate_perturbations(self):
        """Generate new perturbations."""
        print(f"\nMODE: Generating new perturbations")
        print(f"Target: {self.n_target} structures")
        
        attempts = 0
        max_attempts = self.n_target * 6
        
        while len(self.results) < (self.n_target + 1) and attempts < max_attempts:
            attempts += 1
            intensity = np.random.uniform(*self.jitter_range)
            
            # Generate perturbation
            pert_coords = self.generate_perturbation(intensity)
            
            # Split into positions dict
            positions = {
                'A': pert_coords[:self.n_A],
                'B': pert_coords[self.n_A:self.n_A+self.n_B],
                'C': pert_coords[self.n_A+self.n_B:]
            }
            
            # Calculate scores
            all_scores = self.calculate_scores(positions, self.sigma)
            
            if not self.is_structure_acceptable(all_scores):
                continue
                
            rmsd_metrics = self.calculate_rmsd_metrics(pert_coords)
            record = self.create_result_record(
                pert_coords, positions, all_scores, rmsd_metrics, self.sigma,
                intensity, attempts
            )
            self.results.append(record)
            
            if len(self.results) % 10 == 0:
                self._log_acceptance(record, len(self.results)-1, attempts)
        
        if len(self.results) < (self.n_target + 1):
            print(f"\nWarning: Only generated {len(self.results)-1}/{self.n_target} structures")
        
    def _print_reference_info(self, ref_scores):
        """Print information about the reference system."""
        print(f"\nReference Structure Scores:")
        print(f"  Pair:          {ref_scores.get('pair', 0.0):.2f}")
        print(f"  ExVol:         {ref_scores.get('exvol', 0.0):.2f}")
        print(f"  Pair+ExVol:    {ref_scores.get('pair_exvol', 0.0):.2f}")
        print(f"  Tetramer:      {ref_scores.get('tetramer', 0.0):.2f}")
        print(f"  Tetramer+ExVol:{ref_scores.get('tetramer_exvol', 0.0):.2f}")
        print(f"  Octet:         {ref_scores.get('octet', 0.0):.2f}")
        print(f"  Octet+ExVol:   {ref_scores.get('octet_exvol', 0.0):.2f}")
        if ref_scores.get('ccc', 0.0) != 0.0:
            print(f"  CCC:           {ref_scores.get('ccc', 0.0):.4f}")
            print(f"  EM Score:      {ref_scores.get('em_score', 0.0):.2f}")
        
        print(f"\nSettings:")
        print(f"  Overlap tolerance: {self.overlap_tolerance}")
        print(f"  Max total score:   {self.max_total_score}")
        print(f"  ExVol kappa:       {self.exvol_kappa}")
        if not self.trajectory_file:
            print(f"  Jitter range:      {self.jitter_range} Å")
            print(f"  Incremental steps: {self.incremental_steps}")
        
    def _log_acceptance(self, record, n_accepted, attempt_num):
        """Log details of an accepted perturbation."""
        print(f"Accepted {n_accepted:03d}/{self.n_target} | attempt {attempt_num:04d} | "
              f"int={record['intensity_requested']:5.1f} | "
              f"RMSD={record['rmsd_aligned']:6.2f} | "
              f"Pair={record['score_pair']:6.1f} | "
              f"Tet={record['score_tetramer']:6.1f} | "
              f"Oct={record['score_octet']:6.1f}")
              
    def _log_trajectory_acceptance(self, record, n_processed, step):
        """Log details of a trajectory structure."""
        print(f"Processed {n_processed:03d} | step {step:05d} | "
              f"RMSD={record['rmsd_aligned']:6.2f} | "
              f"Pair={record['score_pair']:6.1f} | "
              f"Tet={record['score_tetramer']:6.1f} | "
              f"Oct={record['score_octet']:6.1f}")
              
    def _print_final_summary(self):
        """Print comprehensive analysis summary."""
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        
        n_structures = len(self.results) - 1
        mode = "trajectory analysis" if self.trajectory_file else "perturbation generation"
        print(f"Mode: {mode}")
        print(f"Total structures: {len(self.results)} (including reference)")
        
        if len(self.results) <= 1:
            return
            
        # Extract metrics (excluding reference)
        structures_only = self.results[1:]
        
        # Collect score types
        score_types = ['pair', 'exvol', 'pair_exvol', 'tetramer', 'tetramer_exvol', 
                      'octet', 'octet_exvol']
        if self.em_map_file:
            score_types.extend(['ccc', 'em_score'])
        
        metrics = {
            'aligned_rmsd': np.array([r['rmsd_aligned'] for r in structures_only]),
            'raw_rmsd': np.array([r['rmsd_raw'] for r in structures_only])
        }
        
        # Add score metrics
        for score_type in score_types:
            key = f'score_{score_type}'
            if key in structures_only[0]:
                metrics[key] = np.array([r[key] for r in structures_only])
        
        # Print RMSD statistics
        print("\nRMSD Statistics:")
        for name in ['aligned_rmsd', 'raw_rmsd']:
            values = metrics[name]
            print(f"  {name.replace('_', ' ').title():15}: "
                  f"mean={values.mean():6.2f} ± {values.std():6.2f} "
                  f"(min={values.min():6.2f}, max={values.max():6.2f})")
        
        # Print score statistics
        print("\nScore Statistics:")
        for score_type in score_types:
            key = f'score_{score_type}'
            if key in metrics:
                values = metrics[key]
                # Check for constant scores
                if np.std(values) < 1e-10:
                    print(f"  {score_type.upper():15}: CONSTANT = {values[0]:.2f}")
                else:
                    print(f"  {score_type.upper():15}: "
                          f"mean={values.mean():8.2f} ± {values.std():8.2f} "
                          f"(min={values.min():8.2f}, max={values.max():8.2f})")
        
        # Print correlations
        if len(structures_only) > 1:
            print("\nCorrelations (Aligned RMSD vs Scores):")
            for score_type in score_types:
                key = f'score_{score_type}'
                if key in metrics:
                    # Check if score has variance
                    if np.std(metrics[key]) < 1e-10:
                        print(f"  vs {score_type.upper():15}: N/A (constant score)")
                    else:
                        # Suppress numpy warnings for this calculation
                        with np.errstate(invalid='ignore'):
                            corr = np.corrcoef(metrics['aligned_rmsd'], metrics[key])[0,1]
                        if np.isnan(corr):
                            print(f"  vs {score_type.upper():15}: N/A (insufficient variance)")
                        else:
                            print(f"  vs {score_type.upper():15}: {corr:6.3f}")
            
    def save_results(self, output_file=None):
        """Save results to HDF5 file with ALL scores."""
        if output_file is None:
            if self.trajectory_file:
                base_name = Path(self.trajectory_file).stem
                output_file = f"tests/output/{base_name}_score_analysis.h5"
            else:
                output_file = "tests/output/perturbation_score_analysis.h5"
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_file, 'w') as f:
            # Create groups
            groups = {
                'array_A': f.create_group('array_A'),
                'array_B': f.create_group('array_B'), 
                'array_C': f.create_group('array_C'),
                'metadata': f.create_group('metadata')
            }
            
            # Save each result
            for i, result in enumerate(self.results):
                frame_name = f"frame_{i:04d}"
                
                # Save coordinates
                groups['array_A'].create_dataset(frame_name, data=result['coords_A'])
                groups['array_B'].create_dataset(frame_name, data=result['coords_B'])
                groups['array_C'].create_dataset(frame_name, data=result['coords_C'])
                
                # Save metadata
                meta_group = groups['metadata'].create_group(frame_name)
                for key, value in result.items():
                    if not key.startswith("coords_"):
                        if isinstance(value, (str, bytes)):
                            meta_group.attrs[key] = str(value).encode('utf-8')
                        else:
                            meta_group.attrs[key] = value
            
            # Save global attributes
            f.attrs['num_frames'] = len(self.results)
            analysis_mode = "trajectory" if self.trajectory_file else "perturbation"
            f.attrs['description'] = f"Score analysis with all scoring functions ({analysis_mode})".encode('utf-8')
            f.attrs['overlap_tolerance'] = self.overlap_tolerance
            f.attrs['max_total_score'] = self.max_total_score
            f.attrs['exvol_kappa'] = self.exvol_kappa
            f.attrs['has_reference_frame'] = True
            f.attrs['analysis_mode'] = analysis_mode.encode('utf-8')
            
            if self.trajectory_file:
                f.attrs['source_trajectory'] = str(self.trajectory_file).encode('utf-8')
            if self.em_map_file:
                f.attrs['em_map_file'] = str(self.em_map_file).encode('utf-8')
            
        print(f"\n Saved results: {output_file}")
        print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")

    def save_plots(self, output_file=None):
        """
        Create PDF plots of Score vs RMSD for each sampler stage.
        
        Creates 4 plots matching the sampling pipeline:
        1. Pair Sampler: pair + exvol
        2. Tetramer Sampler: pair + tetramer + exvol
        3. Octet Sampler: pair + tetramer + octet + exvol
        4. Full Sampler: pair + tetramer + octet + em + exvol
        
        Each plot has a zoomed inset for low score range.
        """
        if output_file is None:
            if self.trajectory_file:
                base_name = Path(self.trajectory_file).stem
                output_file = f"tests/output/{base_name}_score_analysis.pdf"
            else:
                output_file = "tests/output/perturbation_score_analysis.pdf"
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract data (excluding reference structure)
        structures_only = self.results[1:] if len(self.results) > 1 else []
        
        if not structures_only:
            print("Warning: No structures to plot (only reference)")
            return
        
        # Collect metrics
        rmsd = np.array([r['rmsd_aligned'] for r in structures_only])
        
        scores = {
            'pair': np.array([r['score_pair'] for r in structures_only]),
            'exvol': np.array([r['score_exvol'] for r in structures_only]),
            'tetramer': np.array([r['score_tetramer'] for r in structures_only]),
            'octet': np.array([r['score_octet'] for r in structures_only]),
        }
        
        # Add EM score if available
        if self.em_map_file and structures_only[0].get('score_em_score') is not None:
            scores['em'] = np.array([r['score_em_score'] for r in structures_only])
        
        # Set style
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.4)
        
        # Create PDF with 4 pages
        with PdfPages(output_file) as pdf:
            # Page 1: Pair Sampler
            pair_score = scores['pair'] + scores['exvol']
            self._plot_sampler_score(rmsd, pair_score, "Pair Sampler", 
                                    "Pair + ExVol", pdf)
            
            # Page 2: Tetramer Sampler
            tetramer_score = scores['pair'] + scores['tetramer'] + scores['exvol']
            self._plot_sampler_score(rmsd, tetramer_score, "Tetramer Sampler",
                                    "Pair + Tetramer + ExVol", pdf)
            
            # Page 3: Octet Sampler
            octet_score = scores['pair'] + scores['tetramer'] + scores['octet'] + scores['exvol']
            self._plot_sampler_score(rmsd, octet_score, "Octet Sampler",
                                    "Pair + Tetramer + Octet + ExVol", pdf)
            
            # Page 4: Full Sampler
            if 'em' in scores:
                full_score = scores['pair'] + scores['tetramer'] + scores['octet'] + scores['em'] + scores['exvol']
                self._plot_sampler_score(rmsd, full_score, "Full Sampler",
                                        "Pair + Tetramer + Octet + EM + ExVol", pdf)
        
        print(f"\n Saved plots: {output_file}")
        print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")
    
    def _plot_sampler_score(self, rmsd, scores, title, score_label, pdf, zoom_max=1200):
        """
        Create a single score vs RMSD plot with zoomed inset.
        
        Args:
            rmsd: Array of RMSD values
            scores: Array of score values
            title: Plot title (e.g., "Pair Sampler")
            score_label: Y-axis label (e.g., "Pair + ExVol")
            pdf: PdfPages object
            zoom_max: Maximum score for zoomed inset
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Main plot - full range
        ax.scatter(rmsd, scores, alpha=0.5, s=30, color='steelblue', edgecolors='black', linewidths=0.3)
        
        # Calculate correlation
#        if len(rmsd) > 1:
#            corr = np.corrcoef(rmsd, scores)[0, 1]
#
            # Add trend line
#            z = np.polyfit(rmsd, scores, 1)
#            p = np.poly1d(z)
#            x_trend = np.linspace(rmsd.min(), rmsd.max(), 100)
#            ax.plot(x_trend, p(x_trend), "r-", alpha=0.7, linewidth=2,
#                   label=f'r = {corr:.3f}')
        
        ax.set_xlabel('Aligned RMSD (Å)', fontweight='bold')
        ax.set_ylabel(f'{score_label} Score', fontweight='bold')
        ax.set_title(title, fontweight='bold', fontsize=16, pad=15)
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Add zoomed inset
        # Position: top-right corner, 40% of width and height
        ax_inset = inset_axes(ax, width="40%", height="40%", loc='upper right',
                             borderpad=2)
        
        # Filter data for zoom
        zoom_mask = scores <= zoom_max
        if np.any(zoom_mask):
            rmsd_zoom = rmsd[zoom_mask]
            scores_zoom = scores[zoom_mask]
            
            ax_inset.scatter(rmsd_zoom, scores_zoom, alpha=0.5, s=20, 
                           color='steelblue', edgecolors='black', linewidths=0.3)
            
            # Add trend line for zoomed data
 #           if len(rmsd_zoom) > 1:
 #               z_zoom = np.polyfit(rmsd_zoom, scores_zoom, 1)
 #               p_zoom = np.poly1d(z_zoom)
 #               x_trend_zoom = np.linspace(rmsd_zoom.min(), rmsd_zoom.max(), 100)
 #               ax_inset.plot(x_trend_zoom, p_zoom(x_trend_zoom), "r-", 
 #                           alpha=0.7, linewidth=1.5)
            
            ax_inset.set_ylim(0, zoom_max)
            ax_inset.set_xlabel('RMSD (Å)', fontsize=10)
            ax_inset.set_ylabel('Score', fontsize=10)
            ax_inset.set_title(f'Zoomed (Score ≤ {zoom_max})', fontsize=10)
            ax_inset.grid(True, alpha=0.3)
        else:
            # If no points in zoom range, show message
            ax_inset.text(0.5, 0.5, f'No scores ≤ {zoom_max}',
                        ha='center', va='center', transform=ax_inset.transAxes)
            ax_inset.set_xticks([])
            ax_inset.set_yticks([])
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def save_results(self, output_file=None):
        """Save results to HDF5 file with ALL scores."""
        if output_file is None:
            if self.trajectory_file:
                base_name = Path(self.trajectory_file).stem
                output_file = f"tests/output/{base_name}_score_analysis.h5"
            else:
                output_file = "tests/output/perturbation_score_analysis.h5"
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_file, 'w') as f:
            # Create groups
            groups = {
                'array_A': f.create_group('array_A'),
                'array_B': f.create_group('array_B'), 
                'array_C': f.create_group('array_C'),
                'metadata': f.create_group('metadata')
            }
            
            # Save each result
            for i, result in enumerate(self.results):
                frame_name = f"frame_{i:04d}"
                
                # Save coordinates
                groups['array_A'].create_dataset(frame_name, data=result['coords_A'])
                groups['array_B'].create_dataset(frame_name, data=result['coords_B'])
                groups['array_C'].create_dataset(frame_name, data=result['coords_C'])
                
                # Save metadata
                meta_group = groups['metadata'].create_group(frame_name)
                for key, value in result.items():
                    if not key.startswith("coords_"):
                        if isinstance(value, (str, bytes)):
                            meta_group.attrs[key] = str(value).encode('utf-8')
                        else:
                            meta_group.attrs[key] = value
            
            # Save global attributes
            f.attrs['num_frames'] = len(self.results)
            analysis_mode = "trajectory" if self.trajectory_file else "perturbation"
            f.attrs['description'] = f"Score analysis with all scoring functions ({analysis_mode})".encode('utf-8')
            f.attrs['overlap_tolerance'] = self.overlap_tolerance
            f.attrs['max_total_score'] = self.max_total_score
            f.attrs['exvol_kappa'] = self.exvol_kappa
            f.attrs['has_reference_frame'] = True
            f.attrs['analysis_mode'] = analysis_mode.encode('utf-8')
            
            if self.trajectory_file:
                f.attrs['source_trajectory'] = str(self.trajectory_file).encode('utf-8')
            if self.em_map_file:
                f.attrs['em_map_file'] = str(self.em_map_file).encode('utf-8')
            
        print(f"\n✓ Saved results: {output_file}")
        print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")
        
        # Save plots
        plot_file = str(output_path.with_suffix('.pdf'))
        self.save_plots(plot_file)

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze scoring functions on structures')
    parser.add_argument('trajectory', nargs='?', help='Optional trajectory HDF5 file')
    parser.add_argument('--n-target', type=int, default=100, help='Number of structures to analyze/generate')
    parser.add_argument('--em-map', type=str, help='EM density map file for CCC scoring')
    parser.add_argument('--jitter-min', type=float, default=0.1, help='Minimum jitter intensity')
    parser.add_argument('--jitter-max', type=float, default=25.0, help='Maximum jitter intensity')
    parser.add_argument('--max-score', type=float, default=10000.0, help='Maximum acceptable score')
    parser.add_argument('--output', type=str, help='Output file path')
    
    args = parser.parse_args()
    
    # Check if trajectory file exists
    trajectory_file = None
    if args.trajectory:
        if os.path.exists(args.trajectory):
            trajectory_file = args.trajectory
            print(f"Will analyze trajectory from: {trajectory_file}")
        else:
            print(f"Error: Trajectory file not found: {args.trajectory}")
            sys.exit(1)
    else:
        print("No trajectory file specified - will generate new perturbations")
    
    # Create analyzer
    analyzer = PerturbationAnalyzer(
        n_target=args.n_target,
        jitter_range=(args.jitter_min, args.jitter_max),
        max_total_score=args.max_score,
        trajectory_file=trajectory_file,
        em_map_file=args.em_map
    )
    
    # Run analysis
    analyzer.run_analysis()
    
    # Save results
    analyzer.save_results(args.output)
    
    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()