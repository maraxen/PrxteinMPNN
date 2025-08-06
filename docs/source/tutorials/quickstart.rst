Quick Start
===========

This guide will get you started with PrxteinMPNN in just a few minutes.

Installation
------------

Install PrxteinMPNN using pip:

.. code-block:: bash

   pip install -e .

Basic Usage
-----------

Here's a simple example to get you started:

.. code-block:: python

    import jax
    import jax.numpy as jnp

    from prxteinmpnn.mpnn import ProteinMPNNModelVersion, ModelWeights, get_mpnn_model
    from prxteinmpnn.io import from_structure_file

    # Load a protein structure
    protein_structure = from_structure_file(filename="path/to/structure.pdb")

    # Get the MPNN model
    model = get_mpnn_model(
        model_version=ProteinMPNNModelVersion.V_48_020,
        model_weights=ModelWeights.DEFAULT
    )

    # Get the model inputs for the given structure
    model_inputs = protein_structure_to_model_inputs(protein_structure)
    
    # Score sequences
    from prxteinmpnn.scoring.score import make_score_sequence
    from prxteinmpnn.utils.decoding_order import random_decoding_order
    key = jax.random.PRNGKey(0)  # Random key for JAX operations
    score_sequence = make_score_sequence(parameters, random_decoding_order, model_inputs=model_inputs)
    original_sequence_score, original_sequence_logits, original_decoding_order_used = 
      score_sequence(
        key,
        inputs.sequence
      )
    other_sequence = "SOMEOTHERSEQUENCE"
    other_sequence_score, other_sequence_logits, new_decoding_order = score_sequence(
      key,
      other_sequence
    )

Next Steps
----------

* Read the :doc:`installation` guide for detailed setup instructions
* Explore :doc:`basic_usage` for more comprehensive examples
* Check out the :doc:`../api/modules` for complete API reference