**Toy Model Problem for Bayesian Hierarchical Modeling**
python run_simulation.py to run the sampling of spherical particles
There are 4 different levels of sampling possible, one is with moving individual particles and adding pair restraints between them,
second is using tetrameric moves and scoring those appropriately, third is making octet moves and the global moves correspond 
to using the EM restraint and packing the particles into a cryo-em model density. Each of these can be controlled in the run_simulation script.
