# Tests

## RMSD vs Score Test

The `test_rmsd_vs_score.py` test generates graphs showing the relationship between RMSD (Root Mean Square Deviation) from the ideal structure and the scoring function for different samplers.

### What it does:

1. **Loads the ideal structure** from `SystemParameters`
2. **Scores the ideal structure** using each sampler's scoring function
3. **Creates perturbations** of the ideal structure with varying magnitudes
4. **Calculates RMSD** between each perturbation and the ideal structure
5. **Scores each perturbation** using different sampler scoring functions
6. **Generates plots** showing RMSD vs Score relationship

### Samplers tested:

- **Pair Sampler**: Uses pairwise distance restraints
- **Tetramer Sampler**: Includes pairwise + tetramer formation scores
- **Octet Sampler**: Includes pairwise + tetramer + octet formation scores

### Running the test:

```bash
# Install dependencies (if not already installed)
pip install numpy matplotlib scikit-learn h5py

# Run the test
cd /path/to/toy_model_bhm
python tests/test_rmsd_vs_score.py
```

### Output:

The test generates the following plots in `tests/output/`:

- `rmsd_vs_score_pair.png` - RMSD vs Score for Pair Sampler
- `rmsd_vs_score_tetramer.png` - RMSD vs Score for Tetramer Sampler
- `rmsd_vs_score_octet.png` - RMSD vs Score for Octet Sampler
- `rmsd_vs_score_all_samplers.png` - Combined plot for all samplers

### Expected behavior:

- As RMSD increases (structures get further from ideal), the score (negative log posterior) should generally increase
- This demonstrates that the scoring functions correctly penalize deviations from the ideal structure
- Different samplers may show different sensitivities to perturbations based on their scoring components

### Test framework:

The test uses Python's built-in `unittest` framework and can be run individually or as part of a test suite.
