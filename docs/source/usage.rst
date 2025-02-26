Usage
=====

.. _installation:

Installation
------------

To use `idp-design`, first install it using pip:

.. code-block:: console

   git clone https://github.com/rkruegs123/idp-design.git
   cd idp-design
   pip install -e .

There are two packages you must install manually -- `sparrow` and `JAX-MD`:

.. code-block:: console

   pip install git+https://git@github.com/idptools/sparrow.git@a770f78013e6399d992e53921540e559defef94b
   pip install https://github.com/jax-md/jax-md/archive/main.zip
