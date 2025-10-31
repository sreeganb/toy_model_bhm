from core.parameters import SystemParameters
from core.system import setup_system
from samplers.pair import run_pair_sampling
from samplers.tetramer import run_tetramer_sampling
from samplers.octet import run_octet_sampling
from samplers.full import run_full_sampling, set_em_map_config
from pipeline import SamplerPipeline

# Set EM map configuration ONCE before pipeline (no state needed yet)
#set_em_map_config(map_file="simulated_target_density.mrc", resolution=50.0, backend='cpu')
set_em_map_config(map_file="target_map.mrc", resolution=50.0, backend='cpu')

def main():
    # Initialize system parameters
    params = SystemParameters()
    
    # Define sampler sequence
    sampler_sequence = ["pair_sampling", "tetramer_sampling", "octet_sampling", "full_sampling"]

    # Setup initial system state with sequence
    system_state = setup_system(
        params=params,
        source="random",
        sampler_sequence=sampler_sequence,
        current_sampler=sampler_sequence[0]
    )
    
    # Create and run pipeline
    pipeline = SamplerPipeline(system_state, prior_type="inv_gamma")
    
    # Add sampling stages
    pipeline.add_stage(
        run_pair_sampling,
        n_steps=30000,
        save_freq=100,
        temp_start=10.0,
        temp_end=1.0
    )
    
    pipeline.add_stage(
        run_tetramer_sampling,
        n_steps=15000,
        save_freq=100,
        temp_start=10.0,
        temp_end=1.0
    )
    
    pipeline.add_stage(
        run_full_sampling,
        n_steps=750,
        save_freq=1,
        temp_start=10.0,
        temp_end=1.0,
        center_to_density=True
    )
    
    # Run the pipeline with multiple chains
    results = pipeline.run(
        output_base="output",
        n_chains=8,
        use_replica_exchange=False
    )
    
    print("Simulation complete!")

if __name__ == "__main__":
    main()
