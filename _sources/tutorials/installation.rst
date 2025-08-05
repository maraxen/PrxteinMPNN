Installation Guide
==================

This guide covers the installation of PrxteinMPNN and its dependencies.

Requirements
------------

* Python >= 3.11
* JAX ecosystem (jax, jaxlib, flax)
* Additional scientific computing libraries

Basic Installation
------------------

Install from source:

.. code-block:: bash

   git clone <repository-url>
   cd PrxteinMPNN
   pip install -e .

Development Installation
------------------------

For development work, install with development dependencies:

.. code-block:: bash

   pip install -e ".[dev]"

This includes tools for:

* Testing (pytest, pytest-cov)
* Linting (ruff)
* Type checking (pyright) 
* Documentation (sphinx, sphinx-rtd-theme)

Verification
------------

Verify your installation:

.. code-block:: python

   import prxteinmpnn
   print(prxteinmpnn.__version__)

Troubleshooting
---------------

Common installation issues and solutions will be documented here.