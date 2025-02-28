# idp-design

This repository contains code corresponding to the paper titled **"[Generalized design of sequence-ensemble-function relationships for intrinsically disordered proteins](https://doi.org/10.1101/2024.10.10.617695)."**
We provide code for **four example optimizations**:
- **Radius of Gyration (Rg) optimization**
- **Salt sensor optimization**
- **Charge-constrained Rg optimization**
- **Binder optimization**

Additional code is available upon request.

![Animation](img/pseq_animation.gif)


# **Installation**

## **0. Create a New Environment**

We recommend starting with fresh environemnt (via `mamba` or `conda`). All code was tested with Python version `3.10.13`:
```sh
mamba create -n <ENV-NAME> python=3.10.13
mamba activate <ENV-NAME>
```

## **1. Clone the Repository**
Next, clone the repository:
```sh
git clone https://github.com/rkruegs123/idp-design.git
cd idp-design
```

## **2. Install Dependencies**
You may install the required dependencies via
```sh
pip install -r requirements.txt
```
Note that by default, we install the CUDA-compatible version of JAX.
If you would like to install the CPU-only version, please replace `jax[cuda]==0.4.31` with `jax==0.4.31`.

You may then install the package **editable mode** via:
```sh
pip install -e .
```
This allows you to modify the code and have changes reflected immediately without reinstalling.

Lastly, install [`sparrow`](https://github.com/idptools/sparrow) via
```sh
pip install git+https://git@github.com/idptools/sparrow.git@a770f78013e6399d992e53921540e559defef94b
```


## **Testing**
To ensure everything is working correctly, run:
```sh
pytest tests/
```
This will execute all tests inside the `tests/` directory.


# **Documentation**

The project documentation is generated using **Sphinx**. To build the documentation locally, run:
```sh
cd docs
make html
```
This will generate the HTML documentation inside the `docs/build/html/` directory. Open `index.html` in a web browser to view the documentation.

For any modifications to the documentation, edit the source files inside `docs/`, then rebuild using `make html`.


# **Usage**

All design scripts save results in a specified directory within the `output` folder.
**Before running any designs, create an output directory:**
```sh
mkdir output
```

All design scripts take a core set of arguments, including:
- `--n-eq-steps`: The number of equilibration timesteps per simulation
- `--n-sample-steps`: The number of timesteps for sampling reference states
- `--sample-every`: The timestep frequency for sampling representative states.
- `--n-iters`: Number of iterations of gradient descent. Note that this also sets the timescale of annealing a probabilistic sequence to a discrete sequence.
- `--lr`: The learning rate.
- `--optimizer-type`: The choice of optimizer (e.g. `adam`, `lamb`).

Additional arguments include the temperature, timestep, and diffusion coefficient. Use `--help` for more details.
Below, we list a subset of arguments that are particularly pertinent to each experiment.

See the documentation for specific example specifications for each example.

## **Design an IDP with a Target Rg**
To design an IDP with a target **radius of gyration (Rg)**:
```sh
python3 -m experiments.design_rg \
    --run-name <RUN-NAME> \
    --seq-length <LENGTH> \
    --target-rg <TARGET-VALUE>
```
- `TARGET-VALUE`: The target Rg in Angstroms.
- `LENGTH`: The length of the IDP.
- Results will be stored in `output/RUN-NAME`.


## **Design an IDP as a Salt Sensor**
To design an IDP that **expands or contracts based on salt concentration**:
```sh
python3 -m experiments.design_rg_salt_sensor \
    --run-name <RUN-NAME> \
    --seq-length <LENGTH> \
    --salt-lo 150 \
    --salt-hi 450 \
    --mode <MODE>
```
- `MODE`: Choose `"expander"` or `"contractor"`.
- `LENGTH`: The length of the IDP.
- Results will be stored in `output/RUN-NAME`.

By default, salt concentrations are:
  - **Low salt**: 150 mM (`--salt-lo 150`)
  - **High salt**: 450 mM (`--salt-hi 450`)

You can adjust these values using the corresponding flags.

## **Design an IDP binder for a given IDP substrate**
To design an IDP that **strongly binds a second, fixed IDP** with sequence `<SUBSTRATE>`:
```sh
python3 -m experiments.design_binder \
    --run-name <RUN-NAME> \
    --substrate <SUBSTRATE> \
    --binder-length <BINDER-LENGTH> \
    --n-devices <N-DEVICES> \
    --n-sims-per-device <N-SIMS-PER-DEVICE> \
    --max-dist <MAX-DIST> \
    --spring-k <SPRING-K>
```
- `BINDER-LENGTH`: the length of the optimized binder.
- Results will be stored in `output/RUN-NAME`.

Unlike previous experiments, this script permits the distribution of simulations across multiple devices.
Additionally, we employ a bias potential to limit the maximum interstrand distance between the substrate and binder. This bias potential is controlled by `--max-dist` andd `--spring-k`.

## **Design an IDP with a Target Rg constrained to a desired charge distribution**
To design an IDP with a target **radius of gyration (Rg)** and a target **charge distribution**:
```sh
python3 -m experiments.design_rg_charge_constrained \
    --run-name <RUN-NAME> \
    --target-rg <TARGET-VALUE> \
    --min-pos-charge-ratio <TARGET-POS-CHARGE-RATIO> \
    --min-neg-charge-ratio <TARGET-NEG-CHARGE-RATIO> \
    --seq-length <LENGTH> \
    --histidine-not-charged
```
- `TARGET-POS-CHARGE-RATIO`: minimum fraction of the sequence that must be positively charged.
- `TARGET-NEG-CHARGE-RATIO`: minimum fraction of the sequence that must be negatively charged.
- Results will be stored in `output/RUN-NAME`.

Note that `TARGET-POS-CHARGE-RATIO + TARGET-NEG-CHARGE-RATIO` cannot exceed `1.0`.
In practice, we find improved performance if their sum is slightly less than `1.0`.
If `--histidine-not-charged` is not set, histidine will be considered a positively charged
residue.


## **Contributing**
If you have suggestions, feel free to **open an issue** or **submit a pull request**.
To maintain code quality standards, please ensure that any contributed code passes linting checks via `ruff check idp_design/`.


## **Reproducing Figures**

We provide plotting code for main figures in `figures/plot_figures.ipynb`.
This notebook has several additional dependencies, installable via:
```sh
pip install seaborn
pip install logomaker
mamba install jupyter
```



# Citation

If you wish to cite this work, please cite the following .bib:
```
@article{krueger2024generalized,
  title={Generalized design of sequence-ensemble-function relationships for intrinsically disordered proteins},
  author={Krueger, Ryan and Brenner, Michael P and Shrinivas, Krishna},
  journal={bioRxiv},
  pages={2024--10},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```
