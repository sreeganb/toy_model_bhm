#============================================================================
# scoring/full_score.py
# EM density cross-correlation scoring function
#============================================================================
import numpy as np
import math
import mrcfile
import scipy.ndimage
from typing import Dict, Tuple
from types import SimpleNamespace

try:
    import cupy as cp
    import cupyx.scipy.ndimage
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

class FullNLL:
    def __init__(self, state, em_map_file: str, resolution: float = 50.0, backend: str = 'cpu'):
        """
        Initialize EM density scoring with system state.
        
        Args:
            state: SystemState object containing positions
            em_map_file: Path to MRC density map file
            resolution: Resolution for Gaussian blurring (Angstroms)
            backend: 'cpu' or 'gpu' for computation
        """
        self.state = state
        self.coordinates = state.positions
        self.em_map_file = em_map_file
        self.resolution = resolution
        self.backend = backend if (backend == 'gpu' and CUPY_AVAILABLE) else 'cpu'
        
        # Load and parse density map ONCE
        self.target_density_map = self._parse_density(em_map_file)
        self.bins = self._bins_from_density(self.target_density_map)
        self.voxel_size = self.target_density_map.voxel_size.x
        
        # Pre-compute sigma for Gaussian blurring
        self.sigma = self._resolution_to_sigma(self.resolution, self.voxel_size)
        
        # Pre-flatten target density for correlation (done ONCE)
        self.target_density_flat = self.target_density_map.data.flatten()
        
        # Calculate map bounds
        self.box_min = np.array([self.bins[0][0], self.bins[1][0], self.bins[2][0]])
        self.box_max = np.array([self.bins[0][-1], self.bins[1][-1], self.bins[2][-1]])
        
        # Get particle radii from state or use defaults
        if hasattr(state, 'params') and hasattr(state.params, 'radii'):
            self.radii = state.params.radii
        else:
            # quit the simulation in case radius is not found 
            raise ValueError("Particle radii not found in state parameters.")
            

        print_debug = False
        if print_debug == True:        
            print(f"FullNLL initialized:")
            print(f"  Map file: {em_map_file}")
            print(f"  Resolution: {resolution} Å")
            print(f"  Voxel size: {self.voxel_size} Å")
            print(f"  Sigma: {self.sigma:.3f}")
            print(f"  Backend: {self.backend}")

    @staticmethod
    def center_particles_to_density_com(positions: Dict[str, np.ndarray], 
                                       density_map, 
                                       box_min: np.ndarray, 
                                       box_max: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Center particles to match density center of mass.
        
        This is a static method so it can be called before creating the scorer.
        
        Args:
            positions: Dict mapping particle types to position arrays
            density_map: MRC density map object
            box_min: Minimum coordinates of density map
            box_max: Maximum coordinates of density map
            
        Returns:
            Centered positions dictionary
        """
        print("Centering particles to density map COM...")
        
        # Calculate particle COM
        all_coords = []
        for key in ['A', 'B', 'C']:
            if key in positions and len(positions[key]) > 0:
                all_coords.append(positions[key])
        
        if not all_coords:
            return positions
            
        combined = np.vstack(all_coords)
        particle_com = np.mean(combined, axis=0)
        
        # Calculate density COM
        data = density_map.data
        nz, ny, nx = data.shape
        
        # Create coordinate grids
        x_coords = np.linspace(box_min[0], box_max[0], nx)
        y_coords = np.linspace(box_min[1], box_max[1], ny) 
        z_coords = np.linspace(box_min[2], box_max[2], nz)
        
        X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        
        # Calculate weighted COM
        total_density = np.sum(data)
        if total_density > 1e-10:
            # data.T for correct indexing with meshgrid
            density_com_x = np.sum(X * data.T) / total_density
            density_com_y = np.sum(Y * data.T) / total_density
            density_com_z = np.sum(Z * data.T) / total_density
            density_com = np.array([density_com_x, density_com_y, density_com_z])
            
            # Calculate required translation
            translation = density_com - particle_com
            
            print(f"  Particle COM: [{particle_com[0]:.2f}, {particle_com[1]:.2f}, {particle_com[2]:.2f}]")
            print(f"  Density COM:  [{density_com[0]:.2f}, {density_com[1]:.2f}, {density_com[2]:.2f}]")
            print(f"  Translation:  [{translation[0]:.2f}, {translation[1]:.2f}, {translation[2]:.2f}]")
            
            # Apply translation
            centered_positions = {}
            for key in positions:
                if isinstance(positions[key], np.ndarray) and len(positions[key]) > 0:
                    centered_positions[key] = positions[key] + translation
                else:
                    centered_positions[key] = positions[key]
                    
            return centered_positions
        
        print("Warning: Density map has zero total density, skipping centering")
        return positions

    def _parse_density(self, fname: str):
        """Parse MRC density file."""
        density = mrcfile.open(fname, permissive=True)
        voxel_size = density.voxel_size
        
        if not (np.allclose(voxel_size.x, voxel_size.y) and 
                np.allclose(voxel_size.x, voxel_size.z)):
            raise ValueError("Non-uniform grids are not supported")
        
        return density

    def _bins_from_density(self, density) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate coordinate bins from density map (centered at origin)."""
        nx, ny, nz = density.header.nx, density.header.ny, density.header.nz
        vx, vy, vz = density.voxel_size.x, density.voxel_size.y, density.voxel_size.z
        
        # Create bins centered at origin (matching reference implementation)
        binsx = (np.linspace(0, nx, nx + 1) - nx/2) * vx
        binsy = (np.linspace(0, ny, ny + 1) - ny/2) * vy
        binsz = (np.linspace(0, nz, nz + 1) - nz/2) * vz
        
        return (binsx, binsy, binsz)

    def _resolution_to_sigma(self, resolution: float, pixel_size: float) -> float:
        """Convert resolution to sigma for Gaussian blurring."""
        return resolution / (4 * math.sqrt(2 * math.log(2))) / pixel_size

    def _calc_projection_cpu(self, coords, weights):
        """Calculate projection using CPU (optimized - bins and sigma are pre-computed)."""
        img, _ = np.histogramdd(coords, weights=weights, bins=self.bins)
        img = np.swapaxes(img, 0, 2)
        
        img_smoothed = scipy.ndimage.gaussian_filter(
            img, self.sigma, truncate=4
        ).astype(np.float32)
        
        return img_smoothed

    def _calc_projection_gpu(self, coords, weights):
        """Calculate projection using GPU (optimized - bins and sigma are pre-computed)."""
        img, _ = cp.histogramdd(coords, weights=weights, bins=self.bins)
        img = cp.swapaxes(img, 0, 2)
        
        img_smoothed = cupyx.scipy.ndimage.gaussian_filter(
            img, self.sigma, truncate=4
        ).astype(np.float32)
        
        return img_smoothed

    def _pairwise_correlation_cpu(self, A, B) -> float:
        """Calculate pairwise correlation using CPU (exactly as reference)."""
        am = A - np.mean(A)
        bm = B - np.mean(B)
        return np.sum(am * bm) / (np.sqrt(np.sum(am**2)) * np.sqrt(np.sum(bm**2)))

    def _pairwise_correlation_gpu(self, A, B) -> float:
        """Calculate pairwise correlation using GPU (exactly as reference)."""
        am = A - cp.mean(A)
        bm = B - cp.mean(B)
        return cp.sum(am * bm) / (cp.sqrt(cp.sum(am**2)) * cp.sqrt(cp.sum(bm**2)))

    def check_particles_bounds(self):
        '''
        Take the map bounds, take the particle coordinates. 
        Check if any particle is outside the map bounds.
        '''
        for ptype in ['A', 'B', 'C']:
            if ptype in self.coordinates and len(self.coordinates[ptype]) > 0:
                coords = self.coordinates[ptype]
                if (np.any(coords < self.box_min) or np.any(coords > self.box_max)):
                    print(f"WARNING: Some particles of type {ptype} are outside the map bounds!")

#    def add_slope(self):
#        '''
#        Add a small linear term to particles that are outside the map bounds.
#        '''

    def calculate_ccc(self, positions: Dict[str, np.ndarray], debug_logging: bool = False) -> float:
        """
        Calculate cross-correlation coefficient between model and density map.
        
        Args:
            positions: Dict mapping particle types to position arrays
            debug_logging: If True, print diagnostic information
            
        Returns:
            CCC score (range: -1 to 1, higher is better)
        """
        # Collect all coordinates and radii
        all_coords = []
        all_radii = []
        
        for ptype in ['A', 'B', 'C']:
            if ptype in positions and len(positions[ptype]) > 0:
                all_coords.append(positions[ptype])
                radius = self.radii.get(ptype, 5.0)
                all_radii.append(np.full(len(positions[ptype]), radius))
        
        if not all_coords:
            if debug_logging:
                print("WARNING: No coordinates provided!")
            return 0.0
        
        sphere_coords = np.vstack(all_coords)
        sphere_radii = np.concatenate(all_radii)
        
        debug_logging = False
        if debug_logging:
            print(f"\nCCC Calculation Debug:")
            print(f"  Total particles: {len(sphere_coords)}")
            print(f"  Coordinate bounds:")
            print(f"    X: [{sphere_coords[:, 0].min():.2f}, {sphere_coords[:, 0].max():.2f}]")
            print(f"    Y: [{sphere_coords[:, 1].min():.2f}, {sphere_coords[:, 1].max():.2f}]")
            print(f"    Z: [{sphere_coords[:, 2].min():.2f}, {sphere_coords[:, 2].max():.2f}]")
            print(f"  Map bounds:")
            print(f"    X: [{self.box_min[0]:.2f}, {self.box_max[0]:.2f}]")
            print(f"    Y: [{self.box_min[1]:.2f}, {self.box_max[1]:.2f}]")
            print(f"    Z: [{self.box_min[2]:.2f}, {self.box_max[2]:.2f}]")
            print(f"  Radii: {np.unique(sphere_radii)}")
        
        # Use volume (radius^3) as weight (exactly as reference)
        weights = sphere_radii**3
        
        if debug_logging:
            print(f"  Weight range: [{weights.min():.2f}, {weights.max():.2f}]")
        
        # Calculate projection
        if self.backend == 'gpu':
            coords_gpu = cp.asarray(sphere_coords)
            weights_gpu = cp.asarray(weights)
            
            projection = self._calc_projection_gpu(coords_gpu, weights_gpu)
            density_data_gpu = cp.asarray(self.target_density_flat)
            
            ccc = self._pairwise_correlation_gpu(
                projection.flatten(), 
                density_data_gpu
            )
            ccc = float(cp.asnumpy(ccc))
        else:
            projection = self._calc_projection_cpu(sphere_coords, weights)
            
            ccc = self._pairwise_correlation_cpu(
                projection.flatten(),
                self.target_density_flat
            )
        debug_logging = False
        if debug_logging:
            print(f"  Model projection stats:")
            print(f"    Shape: {projection.shape}")
            print(f"    Mean: {np.mean(projection):.6e}")
            print(f"    Std: {np.std(projection):.6e}")
            print(f"    Min: {np.min(projection):.6e}")
            print(f"    Max: {np.max(projection):.6e}")
            print(f"    Non-zero voxels: {np.count_nonzero(projection)}/{projection.size}")
            
            print(f"  Target density stats:")
            print(f"    Shape: {self.target_density_map.data.shape}")
            print(f"    Mean: {np.mean(self.target_density_map.data):.6e}")
            print(f"    Std: {np.std(self.target_density_map.data):.6e}")
            print(f"    Non-zero voxels: {np.count_nonzero(self.target_density_map.data)}/{self.target_density_map.data.size}")
            
            print(f"  CCC result: {ccc:.6f}")
        
        return ccc
    
    def compute_score(self) -> float:
        """
        Compute 1 - CCC as score (for minimization).
        
        Score ranges from 0 to 2:
        - CCC = 1.0 (perfect match) → score = 0.0 (minimum)
        - CCC = 0.0 (no correlation) → score = 1.0
        - CCC = -1.0 (anti-correlated) → score = 2.0 (maximum)
        
        Returns:
            1 - CCC score (range: 0 to 2, lower is better)
        """
        ccc = self.calculate_ccc(self.coordinates, debug_logging=False)
        
        # Return 1 - CCC so that minimization improves fit
        return 1.0 - ccc

    def compute_score_with_breakdown(self) -> Tuple[float, Dict[str, float]]:
        """
        Compute EM score with detailed breakdown.
        
        Returns:
            (total_score, breakdown_dict)
            
        breakdown_dict contains:
            - 'ccc': Raw cross-correlation coefficient
            - 'score': 1 - CCC (for minimization)
            - 'n_particles': Total number of particles
            - 'map_bounds': Map coordinate bounds
        """
        ccc = self.calculate_ccc(self.coordinates, debug_logging=False)
        
        # Count particles
        n_particles = sum(
            len(self.coordinates[ptype]) 
            for ptype in ['A', 'B', 'C'] 
            if ptype in self.coordinates
        )
        
        breakdown = {
            'ccc': ccc,
            'score': 1.0 - ccc,
            'n_particles': n_particles,
            'map_bounds': {
                'x_min': float(self.box_min[0]),
                'x_max': float(self.box_max[0]),
                'y_min': float(self.box_min[1]),
                'y_max': float(self.box_max[1]),
                'z_min': float(self.box_min[2]),
                'z_max': float(self.box_max[2])
            }
        }
        
        return 1.0 - ccc, breakdown
#============================================================================