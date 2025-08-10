"""Utility for deriving residue states from trajectory using MPNN logits."""

import itertools
from collections.abc import Iterator
from functools import partial

import jax
from jaxtyping import PRNGKeyArray

from prxteinmpnn.sampling.conditional_logits import make_conditional_logits_fn
from prxteinmpnn.utils.data_structures import ProteinEnsemble, ProteinStructure
from prxteinmpnn.utils.decoding_order import DecodingOrder, DecodingOrderFn
from prxteinmpnn.utils.types import (
  EdgeFeatures,
  InputBias,
  Logits,
  ModelParameters,
  NodeFeatures,
  ProteinSequence,
)

ResidueStates = Iterator[
  tuple[ProteinSequence, tuple[Logits, NodeFeatures, EdgeFeatures], DecodingOrder]
]


def get_ensemble_info(ensemble: ProteinEnsemble) -> tuple[ProteinEnsemble, ProteinStructure, int]:
  """Extract the first frame and count frames in an iterator without deep copying.

  Args:
    ensemble: ProteinEnsemble iterator.

  Returns:
    tuple: (new iterator over ensemble, first frame, number of frames)

  Example:
    >>> ensemble, first_frame, n_frames = get_ensemble_info(ensemble)

  """
  ensemble_iter = iter(ensemble)
  try:
    first_frame = next(ensemble_iter)
  except StopIteration as e:
    msg = "Ensemble is empty."
    raise ValueError(msg) from e
  new_ensemble = itertools.chain([first_frame], ensemble_iter)
  n_frames = 1 + sum(1 for _ in ensemble_iter)
  new_ensemble = itertools.chain([first_frame], iter(ensemble))
  return new_ensemble, first_frame, n_frames


def residue_states_from_ensemble(
  prng_key: PRNGKeyArray,
  model_parameters: ModelParameters,
  decoding_order_fn: DecodingOrderFn,
  ensemble: ProteinEnsemble,
  bias: InputBias | None = None,
) -> ResidueStates:
  """Derive residue states from a protein ensemble using a specified model.

  Args:
    prng_key (PRNGKeyArray): Pseudo-random number generator key for stochastic operations.
    model_parameters (ModelParameters): Parameters for the neural network model used to compute
      logits.
    decoding_order_fn (DecodingOrderFn): Function specifying the order in which residues are
      decoded.
    ensemble (ProteinEnsemble): An iterable of ProteinStructure objects representing the ensemble.
    bias (InputBias | None, optional): Optional bias to be applied to the logits computation.

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
  ensemble_iter = iter(ensemble)
  try:
    first_frame = next(ensemble_iter)
  except StopIteration as e:
    msg = "Ensemble is empty."
    raise ValueError(msg) from e

  ensemble = itertools.chain([first_frame], ensemble_iter)

  get_conditional_logits = make_conditional_logits_fn(
    model_parameters=model_parameters,
    decoding_order_fn=decoding_order_fn,
  )
  logits_fn = partial(
    get_conditional_logits,
    sequence=first_frame.aatype,  # type: ignore[call-arg]
    mask=first_frame.atom_mask[:, 1],  # type: ignore[call-arg]
    residue_index=first_frame.residue_index,  # type: ignore[call-arg]
    chain_index=first_frame.chain_index,  # type: ignore[call-arg]
    bias=bias,  # type: ignore[call-arg]
  )

  def state_generator() -> ResidueStates:
    """Generate residue states for each frame in the ensemble."""
    for i, frame in enumerate(ensemble):
      frame_key = jax.random.fold_in(prng_key, i)
      features = logits_fn(
        prng_key=frame_key,  # type: ignore[arg-type]
        structure_coordinates=frame.coordinates,
      )
      yield features

  return state_generator()
