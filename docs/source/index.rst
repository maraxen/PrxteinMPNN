PrxteinMPNN Documentation
=========================

.. image:: https://img.shields.io/badge/coverage-90%25-brightgreen.svg
  :target: https://github.com/yourusername/PrxteinMPNN/actions/workflows/pytest.yml
  :alt: Test coverage

`Run on Colab <https://colab.research.google.com/github/maraxen/PrxteinMPNN/blob/main/examples/example_notebook.ipynb>`_

Welcome to PrxteinMPNN's documentation! This project provides a functional interface for ProteinMPNN, leveraging the JAX ecosystem for accelerated computation.

.. toctree::
  :maxdepth: 2
  :caption: Contents:

  api/modules
  tutorials/index
  examples/index

Key Features
------------

* **Increased Transparency:** Clear and functional interface for ProteinMPNN
* **JAX Compatibility:** Efficient computation with JAX's functional programming paradigm
* **Modular Design:** Easy updates and extensions
* **Performance Optimization:** Utilizes JAX's JIT compilation and vectorization
* **Validated Implementation:** >0.95 correlation with ColabDesign across all decoding paths

Validation
----------

PrxteinMPNN has been rigorously validated against the original `ColabDesign ProteinMPNN <https://github.com/sokrypton/ColabDesign>`_ implementation:

.. list-table:: Equivalence Test Results
   :widths: 40 30 30
   :header-rows: 1

   * - Decoding Path
     - Correlation
     - Status
   * - Unconditional
     - 0.984
     - ✅ Validated
   * - Conditional
     - 0.958-0.984
     - ✅ Validated
   * - Autoregressive
     - 0.953-0.970
     - ✅ Validated

All three decoding paths achieve **>0.95 Pearson correlation** with ColabDesign outputs, ensuring faithful reproduction of the original model's behavior.

For detailed validation results and testing procedures, see `FINAL_VALIDATION_RESULTS.md <https://github.com/maraxen/PrxteinMPNN/blob/main/docs/FINAL_VALIDATION_RESULTS.md>`_.

Installation
------------

.. code-block:: bash

  pip install -e .

For development:

.. code-block:: bash

  pip install -e ".[dev]"

Use notes
------------
