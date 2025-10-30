#!/usr/bin/env python3
"""
Create HDF5 file with ideal system coordinates.

Usage:
    python tests/create_ideal_h5.py
    
Then convert to RMF3:
    python analysis/h5_to_rmf3.py tests/output/ideal_structure.h5 tests/output/ideal_structure.rmf3
"""

import sys
from pathlib import Path
import h5py
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.parameters import SystemParameters
from core.state import SystemState
from core.sigma import initialize_sigma_dict


def create_ideal_h5(output_file="tests/output/ideal_structure.h5"):
    """Create HDF5 file with ideal coordinates."""
    
    # Get ideal coordinates
    params = SystemParameters()
    ideal_coords = params.latest_ideal()
    
    # Initialize sigma
    sigma = initialize_sigma_dict()
    
    print("Creating ideal structure HDF5 file...")
    print(f"  A particles: {len(ideal_coords['A'])}")
    print(f"  B particles: {len(ideal_coords['B'])}")
    print(f"  C particles: {len(ideal_coords['C'])}")
    print(f"\nSigma values:")
    for key, val in sigma.items():
        print(f"  {key}: {val:.1f}")
    
    # Create output directory
    Path(output_file).parent.mkdir(exist_ok=True)
    
    # Create HDF5 file with same structure as trajectory files
    with h5py.File(output_file, 'w') as f:
        # Create trajectory group
        traj_grp = f.create_group('trajectory')
        
        # Create single state (state_00000)
        state_grp = traj_grp.create_group('state_00000')
        
        # Add positions
        pos_grp = state_grp.create_group('positions')
        for ptype in ['A', 'B', 'C']:
            if ptype in ideal_coords:
                coords = ideal_coords[ptype]
                pos_grp.create_dataset(ptype, data=coords, dtype='float64')
                print(f"  Saved {len(coords)} {ptype} particles")
        
        # Add sigma as attributes
        sigma_grp = state_grp.create_group('sigma')
        for key, val in sigma.items():
            sigma_grp.attrs[key] = val
        
        # Add metadata
        state_grp.attrs['step'] = 0
        state_grp.attrs['score'] = 0.0
    
    print(f"\n✓ Created: {output_file}")
    print(f"  File size: {Path(output_file).stat().st_size / 1024:.1f} KB")
    print(f"\nConvert to RMF3:")
    print(f"  python analysis/h5_to_rmf3.py {output_file} tests/output/ideal_structure.rmf3")


if __name__ == "__main__":
    create_ideal_h5()