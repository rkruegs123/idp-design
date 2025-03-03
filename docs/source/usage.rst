Usage
=====

.. _installation:

Installation
------------

0. Create a New Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^

We recommend starting with fresh environemnt (via :code:`mamba` or :code:`conda`). All code was tested with Python version :code:`3.10.13`:

.. code-block:: console

   mamba create -n <ENV-NAME> python=3.10.13
   mamba activate <ENV-NAME>

1. Clone the Repository
^^^^^^^^^^^^^^^^^^^^^^^

Next, clone the repository:

.. code-block:: console

    git clone https://github.com/rkruegs123/idp-design.git
    cd idp-design


2. Install Dependencies
^^^^^^^^^^^^^^^^^^^^^^^

You may install the required dependencies via

.. code-block:: console

    pip install -r requirements.txt

Note that by default, we install the CUDA-compatible version of JAX.
If you would like to install the CPU-only version, please replace :code:`jax[cuda]==0.4.31` with :code:`jax==0.4.31`.

You then must install `sparrow <https://github.com/idptools/sparrow/>`_ via

.. code-block:: console

    pip install git+https://git@github.com/idptools/sparrow.git@a770f78013e6399d992e53921540e559defef94b

Finally, you may then install the package in **editable mode** via:

.. code-block:: console

    pip install -e .

This allows you to modify the code and have changes reflected immediately without reinstalling.



Testing
-------

To ensure everything is working correctly, run:

.. code-block:: console

    pytest tests/

This will execute all tests inside the :code:`tests/` directory.





Examples
--------

We provide code for **four example optimizations**:

- **Radius of Gyration (Rg) optimization**
- **Salt sensor optimization**
- **Charge-constrained Rg optimization**
- **Binder optimization**


All design scripts save results in a specified directory within the :code:`output` folder.
**Before running any designs, create an output directory:**

.. code-block:: console

    mkdir output



Design an IDP with a Target Rg
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To design an IDP with a target **radius of gyration (Rg)**:

.. code-block:: console

    python3 -m examples.design_rg \
        --run-name <RUN-NAME> \
        --seq-length <LENGTH> \
        --target-rg <TARGET-VALUE>

- :code:`TARGET-VALUE`: The target Rg in Angstroms.
- :code:`LENGTH`: The length of the IDP.
- Results will be stored in :code:`output/RUN-NAME`.

For example, to design an IDP of length ``50`` with ``Rg = 20``, run

.. code-block:: console

    python3 -m examples.design_rg \
        --run-name <RUN-NAME> \
        --seq-length 50 \
        --target-rg 20.0 \
        --min-neff-factor 0.90 \
        --n-iters 50



Design an IDP as a Salt Sensor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To design an IDP that **expands or contracts based on salt concentration**:

.. code-block:: console

    python3 -m examples.design_rg_salt_sensor \
        --run-name <RUN-NAME> \
        --seq-length <LENGTH> \
        --salt-lo 150 \
        --salt-hi 450 \
        --mode <MODE>

- :code:`MODE`: Choose :code:`expander` or :code:`contractor`.
- :code:`LENGTH`: The length of the IDP.
- Results will be stored in :code:`output/RUN-NAME`.


By default, salt concentrations are:

- **Low salt**: 150 mM (:code:`--salt-lo 150`)
- **High salt**: 450 mM (:code:`--salt-hi 450`)

You can adjust these values using the corresponding flags.

For example, to design an IDP of length ``50`` that contracts upon an increase in salt concentration, run

.. code-block:: console

    python3 -m examples.design_rg_salt_sensor \
        --run-name <RUN-NAME> \
        --seq-length 50 \
        --n-iters 100 \
        --mode contractor \
        --salt-lo 150 \
        --salt-hi 450


Design an IDP binder for a given IDP substrate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To design an IDP that **strongly binds a second, fixed IDP** with sequence ``<SUBSTRATE>``:

.. code-block:: console

    python3 -m examples.design_binder \
        --run-name <RUN-NAME> \
        --substrate <SUBSTRATE> \
        --binder-length <BINDER-LENGTH> \
        --n-devices <N-DEVICES> \
        --n-sims-per-device <N-SIMS-PER-DEVICE> \
        --max-dist <MAX-DIST> \
        --spring-k <SPRING-K>

- :code:`BINDER-LENGTH`: the length of the optimized binder.
- Results will be stored in :code:`output/RUN-NAME`.

Unlike previous experiments, this script permits the distribution of simulations across multiple devices.
Additionally, we employ a bias potential to limit the maximum interstrand distance between the substrate and binder. This bias potential is controlled by :code:`--max-dist` andd :code:`--spring-k`.

For example, to design an IDP of length ``30`` strongly binds a polyR sequence of the same length, run

.. code-block:: console

    python3 -m examples.design_binder \
    --run-name <RUN-NAME> \
    --substrate RRRRRRRRRRRRRRRRRRRRRRRRRRRRRR \
    --binder-length 30 \
    --n-sims-per-device 5 \
    --max-dist 150.0 \
    --spring-k 10.0 \
    --n-devices 2

This command assumes that you are on a machine with 2 GPUs.
Change ``--n-devices`` accordingly.

Design an IDP with a Target Rg constrained to a desired charge distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To design an IDP with a target **radius of gyration (Rg)** and a target **charge distribution**:

.. code-block:: console

    python3 -m examples.design_rg_charge_constrained \
        --run-name <RUN-NAME> \
        --target-rg <TARGET-VALUE> \
        --min-pos-charge-ratio <TARGET-POS-CHARGE-RATIO> \
        --min-neg-charge-ratio <TARGET-NEG-CHARGE-RATIO> \
        --seq-length <LENGTH> \
        --histidine-not-charged

- :code:`TARGET-POS-CHARGE-RATIO`: minimum fraction of the sequence that must be positively charged.
- :code:`TARGET-NEG-CHARGE-RATIO`: minimum fraction of the sequence that must be negatively charged.
- Results will be stored in :code:`output/RUN-NAME`.

Note that :code:`TARGET-POS-CHARGE-RATIO + TARGET-NEG-CHARGE-RATIO` cannot exceed :code:`1.0`.
In practice, we find improved performance if their sum is slightly less than :code:`1.0`.
If :code:`--histidine-not-charged` is not set, histidine will be considered a positively charged
residue.

For example, to design an IDP of length ``50`` with ``Rg = 10.0`` and a ``+/-`` charge distribution of ``50/50``, run

.. code-block:: console

    python3 -m examples.design_rg_charge_constrained \
            --run-name test-charge-constrained-k1 \
            --target-rg 20.0 \
            --min-pos-charge-ratio 0.495 \
            --min-neg-charge-ratio 0.495 \
            --n-iters 200 \
            --seq-length 50 \
            --histidine-not-charged \
            --key 1
