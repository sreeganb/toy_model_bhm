#============================================================================
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
        
        # Load and parse density map
        self.target_density_map = self._parse_density(em_map_file)
        self.bins = self._bins_from_density(self.target_density_map)
        
        # Calculate map bounds
        self.box_min = np.array([b[0] for b in self.bins])
        self.box_max = np.array([b[-1] for b in self.bins])
        
        # Get particle radii from state or use defaults
        if hasattr(state, 'params') and hasattr(state.params, 'radii'):
            self.radii = state.params.radii
        else:
            self.radii = {'A': 5.0, 'B': 5.0, 'C': 5.0}

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
            # Note: data.T for correct indexing with meshgrid
            density_com_x = np.sum(X * data.T) / total_density
            density_com_y = np.sum(Y * data.T) / total_density
            density_com_z = np.sum(Z * data.T) / total_density
            density_com = np.array([density_com_x, density_com_y, density_com_z])
            
            # Calculate required translation
            translation = density_com - particle_com
            
            print(f"Centering particles to density COM:")
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
        
        # Calculate map bounds (centered at origin)
        x_extent = nx * vx / 2
        y_extent = ny * vy / 2
        z_extent = nz * vz / 2
        
        binsx = np.linspace(-x_extent, x_extent, nx + 1)
        binsy = np.linspace(-y_extent, y_extent, ny + 1)
        binsz = np.linspace(-z_extent, z_extent, nz + 1)
        
        return (binsx, binsy, binsz)

    def _resolution_to_sigma(self, resolution: float, pixel_size: float) -> float:
        """Convert resolution to sigma for Gaussian blurring."""
        return resolution / (4 * math.sqrt(2 * math.log(2))) / pixel_size

    def _calc_projection_cpu(self, coords, weights, bins):
        """Calculate projection using CPU."""
        img, _ = np.histogramdd(coords, weights=weights, bins=bins)
        img = np.swapaxes(img, 0, 2)
        
        voxel_size = bins[0][1] - bins[0][0]
        sigma = self._resolution_to_sigma(self.resolution, voxel_size)
        
        img_smoothed = scipy.ndimage.gaussian_filter(
            img, sigma, truncate=4
        ).astype(np.float32)
        
        return img_smoothed

    def _calc_projection_gpu(self, coords, weights, bins):
        """Calculate projection using GPU."""
        img, _ = cp.histogramdd(coords, weights=weights, bins=bins)
        img = cp.swapaxes(img, 0, 2)
        
        voxel_size = float(bins[0][1] - bins[0][0])
        sigma = self._resolution_to_sigma(self.resolution, voxel_size)
        
        img_smoothed = cupyx.scipy.ndimage.gaussian_filter(
            img, sigma, truncate=4
        ).astype(np.float32)
        
        return img_smoothed

    def _pairwise_correlation_cpu(self, A, B) -> float:
        """Calculate pairwise correlation using CPU with NaN handling."""
        if len(A) == 0 or len(B) == 0:
            return 0.0
        
        # Center the arrays
        am = A - np.mean(A)
        bm = B - np.mean(B)
        
        # Calculate standard deviations
        std_a = np.sqrt(np.sum(am**2))
        std_b = np.sqrt(np.sum(bm**2))
        
        # Check for zero variance
        if std_a < 1e-10 or std_b < 1e-10:
            return 0.0
        
        # Calculate correlation
        correlation = np.sum(am * bm) / (std_a * std_b)
        
        # Check for NaN
        if not np.isfinite(correlation):
            return 0.0
        
        return float(correlation)

    def _pairwise_correlation_gpu(self, A, B) -> float:
        """Calculate pairwise correlation using GPU."""
        am = A - cp.mean(A)
        bm = B - cp.mean(B)
        
        correlation = cp.sum(am * bm) / (
            cp.sqrt(cp.sum(am**2)) * cp.sqrt(cp.sum(bm**2))
        )
        
        return float(cp.asnumpy(correlation))

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
            return 0.0
        
        sphere_coords = np.vstack(all_coords)
        sphere_radii = np.concatenate(all_radii)
        
        # Use volume (radius^3) as weight
        weights = sphere_radii**3
        
        # Calculate projection
        if self.backend == 'gpu':
            coords_gpu = cp.asarray(sphere_coords)
            weights_gpu = cp.asarray(weights)
            bins_gpu = [cp.asarray(b) for b in self.bins]
            
            projection = self._calc_projection_gpu(coords_gpu, weights_gpu, bins_gpu)
            density_data = cp.asarray(self.target_density_map.data)
            
            ccc = self._pairwise_correlation_gpu(
                projection.flatten(), 
                density_data.flatten()
            )
        else:
            projection = self._calc_projection_cpu(sphere_coords, weights, self.bins)
            
            ccc = self._pairwise_correlation_cpu(
                projection.flatten(),
                self.target_density_map.data.flatten()
            )
        
        if debug_logging:
            print(f"CCC score: {ccc:.6f}")
            print(f"  Model projection: mean={np.mean(projection):.3e}, std={np.std(projection):.3e}")
            print(f"  Target density: mean={np.mean(self.target_density_map.data):.3e}")
        
        return ccc

    def compute_score(self) -> float:
        """
        Compute negative CCC as score (for minimization).
        
        Returns:
            Negative cross-correlation coefficient (range: -1 to 1, lower is better)
        """
        ccc = self.calculate_ccc(self.coordinates, debug_logging=False)
        
        # Return negative CCC so that minimization improves fit
        return -ccc

    def compute_score_with_breakdown(self) -> Tuple[float, Dict[str, float]]:
        """
        Compute EM score with detailed breakdown.
        
        Returns:
            (total_score, breakdown_dict)
            
        breakdown_dict contains:
            - 'ccc': Raw cross-correlation coefficient
            - 'score': Negative CCC (for minimization)
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
            'score': -ccc,
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
        
        return -ccc, breakdown
#============================================================================