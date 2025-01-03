# The restarted Lanczos method for matrix functions

This repository contains the code for most experiments of my MSc thesis in Scientific Computing at TU Berlin.

## Top level folders

- `experiments/`: Lanczos method experiments and visualizations. Also visualizations for experiments
  from `gautschiIntegrators`.
- `gautschiIntegrators/`: Git submodule.
- `pywkm/`: Local copy of the `pyWkm` repository.
- `semi_md/`: Molecular Dynamics code and experiments.
- `src/`: Numpy/ Scipy version of the (restarted) Lanczos method.
- `src_jax/`: Jax version of the Lanczos method.
- `test_np/`: Integration tests for the Numpy Lanczos method.

## Requirements

A specification of one of the Conda environments that I used can be found in `environment.yml`.
This environment is prepared to run almost everything in this repository, except ASE based molecular dynamics
simulations.
If only the Lanczos method experiments are of interest, much simpler environments should work.
For ASE MD simulations try the `ase_environment.yml`.

In order to run the semi-analytical molecular dynamics simulations, the `pyWkm` repository has be copied to the `pywkm`
folder.
Also, the recursive pulling has to be enabled to receive the `gautschiIntegrator` git submodule.

## Lanczos method

The (restarted) Lanczos method for matrix functions is implemented in `src/matfuncb/np_funm.py` in the
function `lanczos_method`.
See the various files in `experiments/` for examples on how to use the method.
The restarting procedure is implemented as described by
> M. Eiermann and O. G. Ernst, “A Restarted Krylov Subspace Method for the Evaluation of Matrix Functions,” *SIAM J.
Numer. Anal.*, vol. 44, no. 6, pp. 2481–2504, Dec. 2006, doi: 10.1137/050633846.

## Molecular Dynamics

In `semi_md/` have made multiple attempts at implementing the semi-analytic MD as described by
> D. L. Michels and M. Desbrun, “A semi-analytical approach to molecular dynamics,” *Journal of Computational Physics*,
> vol. 303, pp. 336–354, Dec. 2015, doi: 10.1016/j.jcp.2015.10.009.

The final version uses only OpenMM as its MD engine.

The system being simulated in my thesis is prepared in `preparation/protein_prep.py`.
A regular OpenMM Verlet simulation can be performed by calling `openmm_md/openmm_protein.py`.
The semi-analytical integrator version is run through `openmm_md/openmm_protein_sa.py`, with step size, step number and
output name changed as necessary.
This makes use of the custom OpenMM integrator `SemiAnalyticIntegrator` found in `openmm_md/SAIntegrator.py`.

In my thesis I analyze the convergence of the Wave-Kernel functions applied to a matrix extracted from the MD
simulation.
This is done in `semi_md/matrix_analysis/bounds_cos_sqrt.py`.
In the same folder there are also experiments comparing the use of `pyWkm` and diagonalization in the restarted Lanczos
method.


