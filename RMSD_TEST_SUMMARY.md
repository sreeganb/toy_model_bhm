# RMSD vs Score Test Summary

## Overview

This test suite creates graphs showing the relationship between RMSD (Root Mean Square Deviation) and scoring functions for different samplers in the toy_model_bhm repository.

## What was implemented:

### 1. Main Test File: `test_rmsd_vs_score.py`

Contains 5 test methods:

1. **`test_ideal_structure_scores`**: Verifies that the ideal structure can be scored by all samplers
2. **`test_rmsd_vs_score_pair_sampler`**: Generates RMSD vs Score graph for the pair sampler
3. **`test_rmsd_vs_score_tetramer_sampler`**: Generates RMSD vs Score graph for the tetramer sampler
4. **`test_rmsd_vs_score_octet_sampler`**: Generates RMSD vs Score graph for the octet sampler
5. **`test_rmsd_vs_score_all_samplers`**: Generates combined RMSD vs Score graph for all samplers

### 2. Utility Functions:

- **`calculate_rmsd(coords1, coords2)`**: Calculates RMSD between two coordinate dictionaries
- **`create_perturbation(ideal_coords, magnitude, rng)`**: Creates perturbed versions of ideal coordinates

### 3. Plot Viewer: `view_rmsd_plots.py`

A standalone utility to view generated plots without re-running the test suite.

### 4. Documentation: `README.md`

Comprehensive documentation explaining:
- What the test does
- How to run it
- Expected output
- Dependencies required

## Test Results

All 5 tests pass successfully:

```
test_ideal_structure_scores - OK
test_rmsd_vs_score_all_samplers - OK
test_rmsd_vs_score_octet_sampler - OK
test_rmsd_vs_score_pair_sampler - OK
test_rmsd_vs_score_tetramer_sampler - OK
```

## Generated Outputs

The test generates 4 PNG files in `tests/output/`:

1. `rmsd_vs_score_pair.png` - Blue scatter plot
2. `rmsd_vs_score_tetramer.png` - Orange scatter plot
3. `rmsd_vs_score_octet.png` - Green scatter plot
4. `rmsd_vs_score_all_samplers.png` - Combined plot with all three samplers

## Sample Scores

### Ideal Structure (RMSD = 0):

- **Pair Sampler**: Total Score = 66.555
  - ExVol: 0.000, Pair: 61.595, Prior: 4.960

- **Tetramer Sampler**: Total Score = 110.507
  - ExVol: 0.000, Pair: 61.595, Tetramer: 43.952, Prior: 4.960

- **Octet Sampler**: Total Score = 119.329
  - ExVol: 0.000, Pair: 61.595, Tetramer: 43.952, Octet: 8.822, Prior: 4.960

### Example Perturbation (RMSD ≈ 10.5 Å):

- **Pair Sampler**: Score = 230,136.92
- **Tetramer Sampler**: Score = 231,229.13
- **Octet Sampler**: Score = 231,275.17

The scores increase significantly with RMSD, demonstrating that the scoring functions correctly penalize deviations from the ideal structure.

## Key Observations

1. **Score increases with RMSD**: As structures deviate from the ideal (higher RMSD), scores (negative log posterior) increase, as expected.

2. **Sampler complexity**: 
   - Pair sampler has the simplest scoring (lowest scores for ideal)
   - Tetramer sampler adds tetramer formation terms
   - Octet sampler adds both tetramer and octet formation terms

3. **Validation**: The test validates that all scoring functions work correctly and penalize deviations appropriately.

## Dependencies

- numpy
- matplotlib
- scikit-learn (for sigma prior GMM fitting)
- h5py

## Running the Test

```bash
# Install dependencies
pip install numpy matplotlib scikit-learn h5py

# Run the test
cd /path/to/toy_model_bhm
python tests/test_rmsd_vs_score.py

# View generated plots
python tests/view_rmsd_plots.py
```

## Files Modified/Added

- `tests/test_rmsd_vs_score.py` (NEW) - Main test file
- `tests/view_rmsd_plots.py` (NEW) - Plot viewer utility
- `tests/README.md` (NEW) - Test documentation
- `.gitignore` (MODIFIED) - Added `tests/output/` to ignore generated plots
