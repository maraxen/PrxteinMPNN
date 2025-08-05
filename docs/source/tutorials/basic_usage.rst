Basic Usage
===========

This tutorial covers the fundamental usage patterns of PrxteinMPNN.

Overview
--------

PrxteinMPNN provides a JAX-compatible functional interface for protein design
using the ProteinMPNN model.

Core Concepts
-------------

* **Functional Design**: All operations follow JAX's functional programming paradigm
* **Immutable Data**: Protein structures and model states are immutable
* **JAX Transformations**: Compatible with jit, vmap, and scan

Basic Workflow
--------------

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from prxteinmpnn
   
   
   # Load protein structure
   # (API details to be finalized)
   
   # Design sequences
   # (Implementation examples to be added)

Performance Tips
----------------

* Use JAX transformations for batch processing
* Leverage JIT compilation for repeated operations
* Consider memory usage with large protein structures

Next Steps
----------

* Explore the :doc:`../examples/index` for complete workflows
* Read the :doc:`../api/modules` for detailed API documentation