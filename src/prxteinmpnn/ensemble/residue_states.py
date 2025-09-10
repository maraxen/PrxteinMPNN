"""Utility for deriving residue states from ensemble using MPNN logits or features."""

from collections.abc import AsyncGenerator

import jax
from jaxtyping import PRNGKeyArray

from prxteinmpnn.sampling.conditional_logits import make_conditional_logits_fn
from prxteinmpnn.utils.data_structures import (
  ProteinEnsemble,
)
from prxteinmpnn.utils.decoding_order import DecodingOrderFn
from prxteinmpnn.utils.types import (
  EdgeFeatures,
  InputBias,
  Logits,
  ModelParameters,
  NodeFeatures,
  ProteinSequence,
)

ResidueStates = AsyncGenerator[tuple[ProteinSequence, tuple[Logits, NodeFeatures, EdgeFeatures]]]


async def residue_states_from_ensemble(
  prng_key: PRNGKeyArray,
  model_parameters: ModelParameters,
  ensemble: ProteinEnsemble,
  decoding_order_fn: DecodingOrderFn | None,
  bias: InputBias | None = None,
  reference_sequence: ProteinSequence | None = None,
) -> ResidueStates:
  """Derive residue states from a protein ensemble using a specified model.

  This function first aligns all structures in the ensemble to create a common
  residue frame of reference. It then computes logits for each structure, using
  the sequence of a single reference structure across all conformations. This
  isolates the impact of conformational changes on the model's predictions.

  Args:
    prng_key (PRNGKeyArray): Pseudo-random number generator key for stochastic operations.
    model_parameters (ModelParameters): Parameters for the neural network model used to compute
      logits.
    decoding_order_fn (DecodingOrderFn): Function specifying the order in which residues are
      decoded.
    ensemble (ProteinEnsemble): An iterable of ProteinStructure objects representing the ensemble.
    bias (InputBias | None, optional): Optional bias to be applied to the logits computation.
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
  get_conditional_logits = make_conditional_logits_fn(
    model_parameters=model_parameters,
    decoding_order_fn=decoding_order_fn,
  )

  i = 0
  async for entry in ensemble:
    if entry is None:
      msg = "Ensemble contains None entries."
      raise ValueError(msg)

    frame_key = jax.random.fold_in(prng_key, i)
    frame, _ = entry
    if reference_sequence is None:
      reference_sequence = frame.aatype

    features = get_conditional_logits(
      frame_key,
      frame.coordinates,
      reference_sequence,
      frame.atom_mask[:, 0],
      frame.residue_index,
      frame.chain_index,
      bias,
      48,
      None,
    )
    yield (
      reference_sequence,
      features,
    )
    i += 1
