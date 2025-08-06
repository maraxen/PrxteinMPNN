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

  from prxteinmpnn.mpnn import ProteinMPNNModelVersion, ModelWeights, get_mpnn_model
  from prxteinmpnn.io import from_structure_file, protein_structure_to_model_inputs

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
  from prxteinmpnn.io import string_to_protein_sequence
  other_sequence_jax = string_to_protein_sequence(other_sequence)
  other_sequence_score, other_sequence_logits, new_decoding_order = score_sequence(
    key,
    other_sequence_jax
  )

  # Sample sequences
  from prxteinmpnn.sampling import make_sample_sequences, SamplingEnum
  sample_sequence = make_sample_sequences(
      parameters,
      random_decoding_order,
      sampling_strategy=SamplingEnum.TEMPERATURE,
      model_inputs=model_inputs
  )
  TEMPERATURE = 0.1  # Example temperature for sampling
  sampled_sequence, logits, decoding_order_used = sample_sequence(
      key,
      inputs.sequence,
      hyperparameters=(TEMPERATURE,),
      iterations=100
  )



Performance Tips
----------------

* Use JAX transformations for batch processing
* Leverage JIT compilation for repeated operations
* Consider memory usage with large protein structures

Next Steps
----------

* Explore the :doc:`../examples/index` for complete workflows
* Read the :doc:`../api/modules` for detailed API documentation