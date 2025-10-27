"""
System State Container
======================
Lightweight container for system state passed between samplers.

IMPORTANT: Sigma initialization is now handled by sigma.py module.
This class only stores sigma values; it does NOT initialize them.
"""
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Sequence

class SystemState:
    """Container for system state (positions, sigma, metadata)"""

    def __init__(
        self,
        sampler_sequence: Optional[Sequence[str]] = None,
        current_sampler: Optional[str] = None
    ):
        # Core data
        self.positions: Dict[str, np.ndarray] = {}
        self.box_size: float = 0.0
        
        # Sigma parameters (initialized externally by pipeline via sigma.py)
        self.sigma: Dict[str, float] = {}
        self.sigma_range: Dict[str, Tuple[float, float]] = {}
        self.sigma_prior: Optional[Any] = None  # SigmaPrior object attached by pipeline
        
        # Sampler sequencing
        self.sampler_sequence: List[str] = list(sampler_sequence) if sampler_sequence else []
        self.current_sampler: Optional[str] = current_sampler
        
        # Cached structures (lazy evaluation)
        self._tetramers: Optional[List[Tuple]] = None
        self._octets: Optional[List[Tuple]] = None
        
        # Additional metadata
        self.metadata: Dict[str, Any] = {}
        
    def __getstate__(self):
        """Custom pickle support for multiprocessing"""
        return self.__dict__.copy()
    
    def __setstate__(self, state):
        """Custom unpickle support"""
        self.__dict__.update(state)
        # Ensure critical attributes exist
        if not hasattr(self, 'sigma'):
            self.sigma = {}
        if not hasattr(self, 'sigma_range'):
            self.sigma_range = {}
            
    def update_positions(self, new_positions: Dict[str, np.ndarray]) -> None:
        """Update positions and clear cached structures"""
        self.positions = {k: v.copy() for k, v in new_positions.items()}
        self._tetramers = None
        self._octets = None

    def update_sigma(self, new_sigma: Dict[str, float]) -> None:
        """Update sigma values"""
        self.sigma = {k: v for k, v in new_sigma.items()}

    def copy(self) -> 'SystemState':
        """Create a deep copy with all essential state"""
        state_copy = SystemState(self.sampler_sequence, self.current_sampler)
        state_copy.positions = {k: v.copy() for k, v in self.positions.items()}
        state_copy.sigma = dict(self.sigma)
        state_copy.sigma_range = dict(self.sigma_range)
        state_copy.box_size = self.box_size
        state_copy.metadata = dict(self.metadata)
        
        # Copy sigma_prior reference (same prior for all copies)
        if hasattr(self, 'sigma_prior'):
            state_copy.sigma_prior = self.sigma_prior
        
        return state_copy

    @property
    def tetramers(self) -> List[Tuple]:
        """Get tetramers (lazy evaluation)"""
        if self._tetramers is None and self._should_compute('tetramer'):
            from samplers.tetramer import get_tetramers
            self._tetramers = get_tetramers(self.positions)
        return self._tetramers or []

    @property
    def octets(self) -> List[Tuple]:
        """Get octets (lazy evaluation)"""
        if self._octets is None:
            from samplers.octet import get_octets
            self._octets, self._tetramers = get_octets(self.positions)
        return self._octets or []

    def _should_compute(self, target: str) -> bool:
        """Check if derived structure should be computed based on sampler sequence"""
        if not self.sampler_sequence or self.current_sampler is None:
            return True
        try:
            target_idx = self.sampler_sequence.index(target)
            current_idx = self.sampler_sequence.index(self.current_sampler)
            return target_idx <= current_idx
        except ValueError:
            return False