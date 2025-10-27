# core/system.py
import os
import glob
import numpy as np
import h5py
import torch
from typing import Dict, Optional, Sequence
from core.state import SystemState
from core.parameters import SystemParameters
from core.io_utils import load_trajectory_from_disk  # Added import for trajectory loading

class SystemBuilder:
    """
    Class-based initializer for SystemState, supporting chained samplers
    (pair, tetramer, octet, full) with ideal, random, or trajectory inputs.
    """
    def __init__(
        self,
        params: SystemParameters,
        sampler_sequence: Sequence[str],
        current_sampler: str,
        source: str = "ideal",
        trajectory_folder: str = "output_analysis/sampler_results",
        trajectory_file: Optional[str] = None,
        frame: int = -1
    ):
        self.params = params
        self.sampler_sequence = list(sampler_sequence)
        self.current_sampler = current_sampler
        self.source = source
        self.trajectory_folder = trajectory_folder
        self.trajectory_file = trajectory_file
        self.frame = frame

    def build(self) -> SystemState:
        """
        Build and return a SystemState with positions set according to configuration.
        """
        # Initialize state with sequencing info
        state = SystemState(self.sampler_sequence, self.current_sampler)
        state.box_size = self.params.box_size

        # Determine and assign positions
        positions = self._get_positions()
        state.update_positions(positions)
        return state

    def _get_positions(self) -> Dict[str, np.ndarray]:
        """
        Select and load positions based on source and sampler order.
        """
        if self._use_trajectory():
            return self._load_trajectory()
        if self.source == "random":
            return self._random_positions()
        if self.source == "ideal":
            return self._ideal_positions()
        # default to random coordinates
        return self._random_positions()

    def _use_trajectory(self) -> bool:
        """
        Decide whether to load from trajectory: only allowed if not first sampler
        or if explicit trajectory_file provided, and source=="trajectory".
        """
        is_first = (self.sampler_sequence and self.sampler_sequence[0] == self.current_sampler)
        if self.source != "trajectory":
            return False
        if is_first and self.trajectory_file is None:
            # cannot use trajectory for first sampler without a file
            return False
        # if explicit file, use it
        if self.trajectory_file:
            return True
        # else auto-select from previous sampler
        return not is_first

    def _auto_select_file(self) -> str:
        """
        Find latest trajectory .h5 file from previous sampler in folder.
        """
        idx = self.sampler_sequence.index(self.current_sampler)
        prev = self.sampler_sequence[idx - 1]
        pattern = os.path.join(self.trajectory_folder, f"{prev}_trajectory_*.h5")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No trajectory files for '{prev}' in {self.trajectory_folder}")
        return files[-1]

    def _load_trajectory(self) -> Dict[str, np.ndarray]:
        """
        Load positions from HDF5 trajectory using the standardized io_utils function.
        """
        traj_file = self.trajectory_file or self._auto_select_file()
        # Use load_trajectory_from_disk for consistent loading (returns positions dict when just_coordinates=True)
        return load_trajectory_from_disk(traj_file, step=self.frame, just_coordinates=True)

    def _random_positions(self) -> Dict[str, np.ndarray]:
        """
        Generate uniformly random positions inside the box.
        """
        pos: Dict[str, np.ndarray] = {}
        for comp, count in self.params.component_counts.items():
            pos[comp] = np.random.uniform(
                low=0.0,
                high=self.params.box_size,
                size=(count, 3)
            )
        return pos

    def _ideal_positions(self) -> Dict[str, np.ndarray]:
        """
        Return the ideal coordinates from params.
        """
        ideal = self.params.ideal_coordinates
        return {k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else np.array(v))
                for k, v in ideal.items()}

def setup_system(
    params: SystemParameters,
    source: str = "ideal",
    sampler_sequence: Optional[Sequence[str]] = None,
    current_sampler: str = "pair_sampling",
    trajectory_folder: str = "output_analysis/sampler_results",
    trajectory_file: Optional[str] = None,
    frame: int = -1
) -> SystemState:
    """
    Convenience function to build a SystemState using SystemBuilder.
    
    Args:
        params: System parameters
        source: Source for positions ("ideal", "random", "trajectory")
        sampler_sequence: Sequence of samplers to run
        current_sampler: Current sampler being used
        trajectory_folder: Folder containing trajectory files
        trajectory_file: Specific trajectory file to load from
        frame: Frame to load from trajectory (-1 for last frame)
    
    Returns:
        SystemState object ready for simulation
    """
    # Default sampler sequence if not provided
    if sampler_sequence is None:
        sampler_sequence = ["pair_sampling", "tetramer_sampling", "octet_sampling"]
    
    builder = SystemBuilder(
        params=params,
        sampler_sequence=sampler_sequence,
        current_sampler=current_sampler,
        source=source,
        trajectory_folder=trajectory_folder,
        trajectory_file=trajectory_file,
        frame=frame
    )
    
    return builder.build()