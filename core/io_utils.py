# io_utils.py - Utilities for saving state information
import numpy as np
import h5py
from typing import Dict, Any, Optional, List

def save_state_to_disk(
    step: int, 
    positions: Dict[str, np.ndarray], 
    sigmas: Dict[str, float], 
    score: float,
    prior_score: float = 0.0, 
    pair_score: float = 0.0, 
    exvol_score: float = 0.0, 
    tet_score: float = 0.0, 
    oct_score: float = 0.0,
    full_score: float = 0.0,
    types: Optional[Dict] = None, 
    bead_numbers: Optional[Dict] = None, 
    traj_file: Optional[str] = None
) -> None:
    """
    Save state directly to an HDF5 file in a memory-efficient manner.
    
    Parameters:
        step: The current step number.
        positions: Dictionary mapping component names to position arrays.
        sigmas: Dictionary of sigma values.
        score: The total score (will be stored as 'total_score').
        prior_score, pair_score, exvol_score, tet_score, oct_score: Additional scores.
        types: Dictionary mapping bead indices to type names.
        bead_numbers: Dictionary mapping bead indices to bead numbers.
        traj_file: Path to the HDF5 file to write to.
    """
    # If types or bead_numbers are not provided, default to empty dictionaries.
    if types is None:
        types = {}
    if bead_numbers is None:
        bead_numbers = {}
    if traj_file is None:
        return

    try:
        with h5py.File(traj_file, 'a') as f:
            # Create or get the trajectory group.
            if 'trajectory' not in f:
                traj_grp = f.create_group('trajectory')
            else:
                traj_grp = f['trajectory']
            
            # Create a new state group named "state_XXXXX" where XXXXX is the step number zero-padded.
            state_name = f"state_{step:05d}"
            state_grp = traj_grp.create_group(state_name)
            
            # Save state attributes (same keys as before).
            state_grp.attrs["step"] = step
            state_grp.attrs["total_score"] = float(score)
            state_grp.attrs["prior_score"] = float(prior_score)
            state_grp.attrs["pair_score"] = float(pair_score)
            state_grp.attrs["exvol_score"] = float(exvol_score)
            state_grp.attrs["tet_score"] = float(tet_score)
            state_grp.attrs["oct_score"] = float(oct_score)
            state_grp.attrs["full_score"] = float(full_score)
            
            # Save sigma as a subgroup.
            sigma_grp = state_grp.create_group("sigma")
            for key, value in sigmas.items():
                sigma_grp.attrs[key] = float(value)
            
            # Save positions as datasets (with gzip compression).
            pos_grp = state_grp.create_group("positions")
            for comp, coords in positions.items():
                pos_grp.create_dataset(comp, data=coords.astype(np.float32), compression="gzip")
            
            # Save types and bead_numbers as datasets.
            types_keys = list(types.keys())
            types_vals = [types[k] for k in types_keys]
            state_grp.create_dataset("types_keys", data=np.array(types_keys, dtype="S"))
            state_grp.create_dataset("types_vals", data=np.array(types_vals, dtype="S"))
            
            bead_keys = list(bead_numbers.keys())
            bead_vals = [bead_numbers[k] for k in bead_keys]
            state_grp.create_dataset("bead_keys", data=np.array(bead_keys))
            state_grp.create_dataset("bead_vals", data=np.array(bead_vals))
    
    except Exception as e:
        print(f"Warning: Failed to save state to HDF5: {e}")
        
def load_trajectory_from_disk(
    traj_file: str, 
    step: int = -1,
    just_coordinates: bool = False,
    load_all: bool = False
) -> Dict[str, Any]:
    """
    Load trajectory data from an HDF5 file.
    
    Parameters:
        traj_file: Path to the HDF5 file to read from.
        step: The step number to load. If -1, load the last step. Ignored if load_all=True.
        just_coordinates: If True, return only the positions. If load_all=True, returns list of position dicts.
        load_all: If True, load all frames sorted by step number.

    Returns:
        If load_all=False: A dictionary containing the loaded trajectory data for the specified step.
        If load_all=True: A list of dictionaries, each containing data for one frame, sorted by step.
    """
    try:
        with h5py.File(traj_file, 'r') as f:
            if 'trajectory' not in f:
                raise ValueError("No trajectory group found in file.")
            traj_grp = f['trajectory']

            if load_all:
                # Load all frames sorted by step
                frames = sorted(
                    (k for k in traj_grp if k.startswith('state_')), 
                    key=lambda x: int(x.split('_')[1])
                )
                if not frames:
                    raise ValueError("No state frames found in trajectory group.")
                
                all_states = []
                for state_name in frames:
                    state_grp = traj_grp[state_name]
                    if just_coordinates:
                        # Only load positions
                        positions = {k: np.array(v) for k, v in state_grp["positions"].items()}
                        all_states.append(positions)
                    else:
                        # Load all attributes
                        sigmas = {k: float(v) for k, v in state_grp["sigma"].attrs.items()}
                        positions = {k: np.array(v) for k, v in state_grp["positions"].items()}
                        types_keys = np.array(state_grp["types_keys"])
                        types_vals = np.array(state_grp["types_vals"])
                        bead_keys = np.array(state_grp["bead_keys"])
                        bead_vals = np.array(state_grp["bead_vals"])
                        
                        state_data = {
                            "step": int(state_grp.attrs["step"]),
                            "total_score": float(state_grp.attrs["total_score"]),
                            "prior_score": float(state_grp.attrs["prior_score"]),
                            "pair_score": float(state_grp.attrs["pair_score"]),
                            "exvol_score": float(state_grp.attrs["exvol_score"]),
                            "tet_score": float(state_grp.attrs["tet_score"]),
                            "oct_score": float(state_grp.attrs["oct_score"]),
                            "full_score": float(state_grp.attrs.get("full_score", 0.0)),
                            "sigmas": sigmas,
                            "positions": positions,
                            "types": dict(zip(types_keys, types_vals)),
                            "bead_numbers": dict(zip(bead_keys, bead_vals))
                        }
                        all_states.append(state_data)
                
                return all_states
            
            else:
                # Load single frame
                if step == -1:
                    # Load the last step
                    frames = sorted(
                        (k for k in traj_grp if k.startswith('state_')), 
                        key=lambda x: int(x.split('_')[1])
                    )
                    if not frames:
                        raise ValueError("No state frames found in trajectory group.")
                    state_name = frames[-1]
                else:
                    state_name = f"state_{step:05d}"

                if state_name not in traj_grp:
                    raise ValueError(f"State {state_name} not found in trajectory group.")

                state_grp = traj_grp[state_name]
                
                if just_coordinates:
                    # Only load positions for the specified/last frame
                    positions = {k: np.array(v) for k, v in state_grp["positions"].items()}
                    return positions
                else:
                    # Load all attributes for the specified/last frame
                    sigmas = {k: float(v) for k, v in state_grp["sigma"].attrs.items()}
                    positions = {k: np.array(v) for k, v in state_grp["positions"].items()}
                    types_keys = np.array(state_grp["types_keys"])
                    types_vals = np.array(state_grp["types_vals"])
                    bead_keys = np.array(state_grp["bead_keys"])
                    bead_vals = np.array(state_grp["bead_vals"])
                    
                    return {
                        "step": int(state_grp.attrs["step"]),
                        "total_score": float(state_grp.attrs["total_score"]),
                        "prior_score": float(state_grp.attrs["prior_score"]),
                        "pair_score": float(state_grp.attrs["pair_score"]),
                        "exvol_score": float(state_grp.attrs["exvol_score"]),
                        "tet_score": float(state_grp.attrs["tet_score"]),
                        "oct_score": float(state_grp.attrs["oct_score"]),
                        "sigmas": sigmas,
                        "positions": positions,
                        "types": dict(zip(types_keys, types_vals)),
                        "bead_numbers": dict(zip(bead_keys, bead_vals))
                    }

    except Exception as e:
        print(f"Warning: Failed to load trajectory from HDF5: {e}")
        return {} if not load_all else []