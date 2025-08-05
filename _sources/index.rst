PrxteinMPNN Documentation
=========================

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

Installation
------------

.. code-block:: bash

   pip install -e .

For development:

.. code-block:: bash

   pip install -e ".[dev]"

Use notes
------------
```python
!pip install nest_asyncio
import nest_asyncio
nest_asyncio.apply()
```

is needed if running `foldcomp_utils.model_from_id` in a notebook


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`