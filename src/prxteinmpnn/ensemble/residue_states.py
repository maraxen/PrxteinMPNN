"""Utility for deriving residue states from ensemble using MPNN logits or features."""

from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from prxteinmpnn.sampling.conditional_logits import make_conditional_logits_fn
from prxteinmpnn.utils.data_structures import (
  Protein,
)
from prxteinmpnn.utils.decoding_order import DecodingOrderFn
from prxteinmpnn.utils.types import (
  EdgeFeatures,
  Logits,
  ModelParameters,
  NodeFeatures,
  OneHotProteinSequence,
  ProteinSequence,
)

ResidueStates = tuple[Logits, NodeFeatures, EdgeFeatures]


def residue_states_from_ensemble(
  prng_key: PRNGKeyArray,
  ensemble: Protein,
  model_parameters: ModelParameters,
  decoding_order_fn: DecodingOrderFn | None,
  reference_sequence: ProteinSequence | OneHotProteinSequence | None = None,
) -> ResidueStates:
  """Derive residue states from a protein ensemble using a specified model.

  This function first aligns all structures in the ensemble to create a common
  residue frame of reference. It then computes logits for each structure, using
  the sequence of a single reference structure across all conformations. This
  isolates the impact of conformational changes on the model's predictions.

  Args:
    prng_key (PRNGKeyArray): Pseudo-random number generator key for stochastic operations.
    ensemble (Protein): A stacked ensemble of protein structures to process.
    model_parameters (ModelParameters): Parameters for the neural network model used to compute
      logits.
    decoding_order_fn (DecodingOrderFn): Function specifying the order in which residues are
      decoded.
    reference_sequence (ProteinSequence | None, optional): Optional reference sequence to use
      for all structures. If None, the sequence from the first structure in the ensemble is used

  Returns:
    ResidueStates: An iterable container with computed logits for each residue in each frame of the
      ensemble.

  Raises:
    ValueError: If the ensemble is empty or improperly formatted.
    TypeError: If input types do not match expected signatures.

  Notes:
    - The function splits the PRNG key for each frame in the ensemble to ensure independent
      stochasticity.
    - The logits are computed conditionally for each frame using the provided model parameters and
      decoding order.
    - The output is a generator yielding logits for each frame in the ensemble.

  """
  if ensemble.coordinates.shape[0] == 0:
    return (
      jnp.array([]),
      jnp.array([]),
      jnp.array([]),
    )

  get_conditional_logits = make_conditional_logits_fn(
    model_parameters=model_parameters,
    decoding_order_fn=decoding_order_fn,
  )

  n_frames = ensemble.coordinates.shape[0]
  static_args = (None, 48, None)

  if reference_sequence is not None:
    cond_logits_fn = partial(
      get_conditional_logits,
      sequence=reference_sequence,  # type: ignore[call]
    )
    return jax.lax.map(
      lambda xs: cond_logits_fn(*xs, *static_args),  # type: ignore[call]
      (
        jax.random.split(prng_key, n_frames),
        ensemble.coordinates,
        ensemble.atom_mask,
        ensemble.residue_index,
        ensemble.chain_index,
      ),
    )

  return jax.lax.map(
    lambda xs: get_conditional_logits(*xs, *static_args),  # type: ignore[call]
    (
      jax.random.split(prng_key, n_frames),
      ensemble.coordinates,
      ensemble.one_hot_sequence,
      ensemble.atom_mask,
      ensemble.residue_index,
      ensemble.chain_index,
    ),
  )
