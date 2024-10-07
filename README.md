# idp-design

This repository contains code corresponding to the paper titled "Generalized design of sequence-ensemble-function relationships for intrinsically disordered proteins." We provide code corresponding to two example optimizations -- radius of gyration (Rg) and salt sensor optimizations. Additional code is available upon request, and will be made public given sufficient demand.

![Animation](img/pseq_animation.gif)

## Installation

First, create a fresh conda environment (via `mamba` or `conda`). All code was tested with Python version 3.10.13:
```
mamba create -n <ENV-NAME> python=3.10.13
```
Next, activate your new environment:
```
mamba activate ENV-NAME
```
Then, navigate to this directory and install the required packages. Note that by default, we install the CUDA-compatible version of JAX. If you would like to install the CPU-only version, please remove the corresponding line from `requirements.txt` and install JAX manually (see [here](https://jax.readthedocs.io/en/latest/installation.html) for instructions):
```
cd path/to/idp-design
pip install -r requirements.txt
```
Lastly, install [sparrow](https://github.com/idptools/sparrow) via
```
pip install git+https://git@github.com/idptools/sparrow.git@a770f78013e6399d992e53921540e559defef94b
```
You'll also want to make a directory to store all your results. The default location for saving results is `idp-design/output`. So, run
```
mkdir output
```

## Usage

All design scripts save results in a specified directory in an `output` directory. Before you design any IDPs, please `cd path/to/idp-design && mkdir output`.

To design an IDP with a target Rg, simply run:
```
python3 -m experiments.design_rg --run-name <RUN-NAME> --seq-length <LENGTH> --target-rg <TARGET-VALUE>
```
where `TARGET-VALUE` is the target Rg (in Angstroms) and `LENGTH` is the length of the IDP. Results will be stored in `output/RUN-NAME`.

To design an IDP that either expands or contracts depending on changes in salt concentration, run
```
python3 -m experiments.design_rg_salt_sensor --run-name <RUN-NAME> --seq-length <LENGTH> --salt-lo 150 --salt-hi 450 --mode MODE
```
where `MODE` is either `expander` or `contractor` and `LENGTH` is the length of the IDP. Results will be stored in `output/RUN-NAME`.
By default, the low and high salt concentrations are 150 mM and 450 mM but these can be changed with the `--salt-lo` and `--salt-hi` arguments.
