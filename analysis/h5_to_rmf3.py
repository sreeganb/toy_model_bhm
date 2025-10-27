import h5py
import numpy as np
import IMP
import IMP.core
import IMP.atom
import IMP.algebra
import IMP.rmf  
import RMF
import sys

def convert_hdf5_to_rmf3(hdf5_file, rmf3_file, radii=None, colors=None):
    """
    Convert HDF5 trajectory to RMF3 format (sampler-agnostic).
    
    Works with any trajectory saved by save_state_to_disk():
        trajectory/state_XXXXX/positions/{A,B,C,...}
    
    Args:
        hdf5_file: Path to input HDF5 file
        rmf3_file: Path to output RMF3 file
        radii: Optional dict of radii per particle type (default: from sigma in first state)
        colors: Optional dict of IMP.display.Color per type (default: Red, Green, Blue for A,B,C)
    """
    # Default radii and colors
    if radii is None:
        radii = {'A': 24.0, 'B': 14.0, 'C': 16.0}
    if colors is None:
        colors = {
            'A': IMP.display.Color(1.0, 0.0, 0.0),  # Red
            'B': IMP.display.Color(0.0, 1.0, 0.0),  # Green  
            'C': IMP.display.Color(0.0, 0.0, 1.0)   # Blue
        }
    
    with h5py.File(hdf5_file, 'r') as f:
        traj_grp = f['trajectory']
        states = sorted([k for k in traj_grp.keys() if k.startswith('state_')],
                       key=lambda x: int(x.split('_')[1]))
        
        if not states:
            print("No states found in trajectory.")
            return
        
        print(f"Found {len(states)} states in trajectory")
        
        # Read first state to determine structure
        first_state = traj_grp[states[0]]
        pos_grp = first_state['positions']
        
        # Get particle types in consistent order
        particle_types_in_file = sorted(pos_grp.keys())
        
        # Override radii from sigma if available
        if 'sigma' in first_state:
            sigma_grp = first_state['sigma']
            for ptype in sigma_grp.attrs:
                radii[ptype] = float(sigma_grp.attrs[ptype])
        
        # Build particle list
        particle_types = []
        initial_coords = []
        
        for ptype in particle_types_in_file:
            coords = pos_grp[ptype][:]
            n_particles = len(coords)
            particle_types.extend([ptype] * n_particles)
            initial_coords.extend(coords)
            print(f"  Type {ptype}: {n_particles} particles, radius={radii.get(ptype, 1.0):.1f}")
        
        initial_coords = np.array(initial_coords)
        n_total = len(initial_coords)
        print(f"Total particles: {n_total}")
        
        # Create IMP Model
        model = IMP.Model()
        
        # Create root hierarchy
        p_root = IMP.Particle(model)
        root_h = IMP.atom.Hierarchy.setup_particle(p_root)
        p_root.set_name("root")
        
        # Create IMP particles
        particles = []
        particle_counter = {pt: 0 for pt in set(particle_types)}
        
        for i, ptype in enumerate(particle_types):
            # Create particle
            p = IMP.Particle(model)
            particle_name = f"{ptype}_{particle_counter[ptype]}"
            p.set_name(particle_name)
            particle_counter[ptype] += 1
            
            # Setup XYZR (coordinates and radius)
            xyzr = IMP.core.XYZR.setup_particle(p)
            coord = initial_coords[i]
            xyzr.set_coordinates(IMP.algebra.Vector3D(coord[0], coord[1], coord[2]))
            xyzr.set_radius(radii.get(ptype, 1.0))
            xyzr.set_coordinates_are_optimized(True)
            
            # Setup mass
            IMP.atom.Mass.setup_particle(p, 1.0)
            
            # Setup color if available
            if ptype in colors:
                IMP.display.Colored.setup_particle(p, colors[ptype])
            
            # Add to hierarchy
            h = IMP.atom.Hierarchy.setup_particle(p)
            root_h.add_child(h)
            
            particles.append(p)
        
        # Create RMF file
        rmf = RMF.create_rmf_file(rmf3_file)
        rmf.set_description("Trajectory converted from HDF5 to RMF3")
        
        # Add hierarchy and enable features
        IMP.rmf.add_hierarchy(rmf, root_h)
        IMP.rmf.add_restraints(rmf, [])  # Enable color support
        
        # Save all frames
        print(f"Converting {len(states)} frames...")
        for frame_idx, state_name in enumerate(states):
            if frame_idx % 50 == 0:
                print(f"  Processing frame {frame_idx+1}/{len(states)}")
            
            state_grp = traj_grp[state_name]
            pos_grp = state_grp['positions']
            
            # Collect coordinates in same order as particle list
            all_coords = []
            for ptype in particle_types_in_file:
                if ptype in pos_grp:
                    coords = pos_grp[ptype][:]
                    all_coords.extend(coords)
            
            all_coords = np.array(all_coords)
            
            # Update particle coordinates
            for j, p in enumerate(particles):
                coord = all_coords[j]
                xyzr = IMP.core.XYZR(p)
                xyzr.set_coordinates(IMP.algebra.Vector3D(coord[0], coord[1], coord[2]))
            
            # Update model and save frame
            model.update()
            IMP.rmf.save_frame(rmf, f"frame_{frame_idx}")
        
        # Close RMF file
        rmf.close()
        del rmf
        
        print(f"\nConversion complete! RMF3 file saved: {rmf3_file}")
        if colors:
            color_info = ", ".join([f"{k}={('Red' if c.get_red()==1.0 else 'Green' if c.get_green()==1.0 else 'Blue')}" 
                                   for k, c in colors.items()])
            print(f"Colors applied: {color_info}")
        print(f"\nOpen with: chimerax {rmf3_file}")
        print("In ChimeraX, use: coordset slider #1")

def inspect_hdf5_structure(hdf5_file):
    """Inspect the structure of an HDF5 file to understand its layout."""
    def print_structure(name, obj):
        print(f"{name}: {type(obj)}")
        if hasattr(obj, 'attrs') and len(obj.attrs) > 0:
            print(f"  Attributes: {dict(obj.attrs)}")
        if hasattr(obj, 'shape'):
            print(f"  Shape: {obj.shape}, Dtype: {obj.dtype}")
    
    print(f"=== HDF5 Structure: {hdf5_file} ===")
    with h5py.File(hdf5_file, 'r') as f:
        f.visititems(print_structure)

# Example usage:
if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--inspect":
        # Usage: python h5_to_rmf3.py --inspect <file>
        if len(sys.argv) == 3:
            inspect_hdf5_structure(sys.argv[2])
        else:
            print("Usage: python h5_to_rmf3.py --inspect <hdf5_file>")
    elif len(sys.argv) < 3:
        print("Usage: python h5_to_rmf3.py <input_hdf5_file> <output_rmf3_file>")
        print("   or: python h5_to_rmf3.py --inspect <hdf5_file>")
        sys.exit(1)
    else:
        input_hdf5 = sys.argv[1]
        output_rmf3 = sys.argv[2]
        
        # Optional: custom radii/colors
        custom_radii = {'A': 24.0, 'B': 14.0, 'C': 16.0}
        custom_colors = {
            'A': IMP.display.Color(1.0, 0.0, 0.0),  # Red
            'B': IMP.display.Color(0.0, 1.0, 0.0),  # Green
            'C': IMP.display.Color(0.0, 0.0, 1.0)   # Blue
        }
        
        convert_hdf5_to_rmf3(input_hdf5, output_rmf3, radii=custom_radii, colors=custom_colors)