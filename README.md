## Toy Model Problem for Bayesian Hierarchical Modeling

### Overview
The **toy model** serves as a simplified test system to explore how **Bayesian Hierarchical Modeling (BHM)** can represent and sample structural ensembles of multi-component assemblies.

### Running the Simulation
To start the sampling of spherical particles, run:
```bash
python run_simulation.py

Sampling Levels

The simulation supports four hierarchical levels of sampling, each corresponding to increasingly complex collective moves:

Particle-Level Sampling

Moves individual particles.

Adds pairwise distance restraints between them.

Tetrameric-Level Sampling

Groups particles into tetramers.

Applies collective “tetrameric” moves and scores them appropriately.

Octet-Level Sampling

Samples larger assemblies (octets).

Evaluates their structural configurations collectively.

Global-Level Sampling

Applies cryo-EM restraints to pack particles into the target EM density.

Represents large-scale, system-wide reorganization moves.

Control Parameters

All sampling levels and restraint combinations can be configured within the run_simulation.py script.
