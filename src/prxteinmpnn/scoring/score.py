"""Score a given sequence on a structure using the ProteinMPNN model."""

import inspect
from collections.abc import Callable
from functools import partial
from typing import Literal, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float, PRNGKeyArray

from prxteinmpnn.model import PrxteinMPNN
from prxteinmpnn.run.averaging import make_encoding_sampling_split_fn
from prxteinmpnn.utils.autoregression import generate_ar_mask
from prxteinmpnn.utils.decoding_order import DecodingOrderFn, random_decoding_order

_DEFAULT_DECODING_ORDER_FN = cast("DecodingOrderFn", random_decoding_order)
from prxteinmpnn.utils.types import (
  AlphaCarbonMask,
  AutoRegressiveMask,
  BackboneNoise,
  ChainIndex,
  DecodingOrder,
  Logits,
  OneHotProteinSequence,
  ProteinSequence,
  ResidueIndex,
  StructureAtomicCoordinates,
)

ScoringFn = Callable[
  [
    PRNGKeyArray,
    ProteinSequence,
    StructureAtomicCoordinates,
    AlphaCarbonMask,
    ResidueIndex,
    ChainIndex,
    int,
    BackboneNoise | None,
    AutoRegressiveMask | None,
    jax.Array | None,
    jax.Array | None,
    Literal["arithmetic_mean", "geometric_mean", "product"],
    Float,
  ],
  tuple[Float, Logits, DecodingOrder],
]


SCORE_EPS = 1e-8


def score_sequence_with_encoding(
  model: PrxteinMPNN,
  sequence: ProteinSequence,
  encoding: tuple,
  tie_group_map: jax.Array | None = None,
  multi_state_strategy: Literal["arithmetic_mean", "geometric_mean", "product"] = "arithmetic_mean",
  multi_state_temperature: float = 1.0,
) -> tuple[Float, Logits, DecodingOrder]:
  """Score a sequence on a structure using pre-computed encodings."""
  _, _, decode_fn = make_encoding_sampling_split_fn(model)

  if sequence.ndim == 1:
    sequence = jax.nn.one_hot(sequence, num_classes=21)

  seq_len = sequence.shape[0]
  ar_mask = jnp.zeros((seq_len, seq_len), dtype=jnp.int32)

  logits = decode_fn(encoding, sequence, ar_mask)
  if tie_group_map is not None:
    strategy_map = {"arithmetic_mean": 0, "geometric_mean": 1, "product": 2}
    strategy_idx = jnp.asarray(strategy_map[multi_state_strategy], dtype=jnp.int32)
    logits = PrxteinMPNN._apply_multistate_to_all_logits(  # noqa: SLF001
      logits,
      tie_group_map,
      strategy_idx,
      multi_state_temperature,
    )

  log_probability = jax.nn.log_softmax(logits, axis=-1)[..., :20]
  score = -(sequence[..., :20] * log_probability).sum(-1)
  mask = encoding[3]  # mask is the 4th element in the encoding tuple
  masked_score_sum = (score * mask).sum(-1)
  mask_sum = mask.sum() + SCORE_EPS

  return masked_score_sum / mask_sum, logits, jnp.arange(seq_len)


def make_score_fn(
  model: PrxteinMPNN,
  decoding_order_fn: DecodingOrderFn = _DEFAULT_DECODING_ORDER_FN,
  _num_encoder_layers: int = 3,
  _num_decoder_layers: int = 3,
  inference: bool = True,  # noqa: FBT001, FBT002
) -> ScoringFn:
  """Create a function to score a sequence on a structure using PrxteinMPNN.

  Args:
    model: A PrxteinMPNN Equinox model instance.
    decoding_order_fn: Function to generate decoding order (default: random).
    _num_encoder_layers: Deprecated, ignored (kept for API compatibility).
    _num_decoder_layers: Deprecated, ignored (kept for API compatibility).
    inference: If True (default), applies eqx.nn.inference_mode to disable
        dropout for deterministic and faster scoring. Set to False only if
        scoring during training where dropout regularization is desired.

  Returns:
    A function that scores sequences on structures.

  Example:
    >>> from prxteinmpnn.io.weights import load_model
    >>> model = load_model()
    >>> score_fn = make_score_sequence(model)
    >>> score, logits, order = score_fn(key, seq, coords, mask, res_idx, chain_idx)

  """

  if inference:
    model = eqx.nn.inference_mode(model, value=True)

  supports_multi_state_temperature = (
    "multi_state_temperature" in inspect.signature(model.__call__).parameters
  )

  @partial(jax.jit, static_argnames=("_k_neighbors", "multi_state_strategy"))
  def score_sequence(
    prng_key: PRNGKeyArray,
    sequence: ProteinSequence | OneHotProteinSequence,
    structure_coordinates: StructureAtomicCoordinates,
    mask: AlphaCarbonMask,
    residue_index: ResidueIndex,
    chain_index: ChainIndex,
    _k_neighbors: int = 48,
    backbone_noise: BackboneNoise | None = None,
    ar_mask: AutoRegressiveMask | None = None,
    structure_mapping: jax.Array | None = None,
    tie_group_map: jax.Array | None = None,
    multi_state_strategy: Literal["arithmetic_mean", "geometric_mean", "product"] = "arithmetic_mean",
    multi_state_temperature: Float = 1.0,
  ) -> tuple[Float, Logits, DecodingOrder]:
    """Score a sequence on a structure using the ProteinMPNN model.

    Args:
      prng_key: JAX random key.
      sequence: Protein sequence (integer indices or one-hot).
      structure_coordinates: Atomic coordinates (N, 4, 3).
      mask: Alpha carbon mask indicating valid residues.
      residue_index: Residue indices.
      chain_index: Chain indices.
      _k_neighbors: Deprecated, model handles internally (kept for API compatibility).
      backbone_noise: Optional noise for backbone coordinates.
      ar_mask: Optional custom autoregressive mask.
      structure_mapping: Optional (N,) array mapping each residue to a structure ID.
                   When provided (multi-state mode), prevents cross-structure
                   neighbors to avoid information leakage between conformational states.
      tie_group_map: Optional (N,) array mapping each position to tied groups.
                    When provided, logits are combined within groups before scoring.
      multi_state_strategy: Strategy for combining logits across tied positions.
      multi_state_temperature: Temperature for geometric_mean strategy.


    Returns:
      Tuple of (average score, logits, decoding order).

    Example:
      >>> score, logits, order = score_sequence(
      ...     key, seq, coords, mask, res_idx, chain_idx
      ... )

    """
    decoding_order, prng_key = decoding_order_fn(prng_key, sequence.shape[0], None, None)
    autoregressive_mask = (
      cast("Callable", generate_ar_mask)(decoding_order) if ar_mask is None else ar_mask
    )

    # Ensure sequence is one-hot encoded
    if sequence.ndim == 1:
      sequence = jax.nn.one_hot(sequence, num_classes=21)

    # Run model in conditional mode (scoring a given sequence)
    if supports_multi_state_temperature:
      _, logits = model(
        structure_coordinates,
        mask,
        residue_index,
        chain_index,
        decoding_approach="conditional",
        prng_key=prng_key,
        ar_mask=autoregressive_mask,
        one_hot_sequence=sequence,
        temperature=0.0,  # Not used in conditional mode
        bias=None,  # No bias in scoring
        backbone_noise=backbone_noise,
        structure_mapping=structure_mapping,
        tie_group_map=tie_group_map,
        multi_state_strategy=multi_state_strategy,
        multi_state_temperature=multi_state_temperature,
      )
    else:
      _, logits = model(
        structure_coordinates,
        mask,
        residue_index,
        chain_index,
        decoding_approach="conditional",
        prng_key=prng_key,
        ar_mask=autoregressive_mask,
        one_hot_sequence=sequence,
        temperature=0.0,  # Not used in conditional mode
        bias=None,  # No bias in scoring
        backbone_noise=backbone_noise,
        structure_mapping=structure_mapping,
        tie_group_map=tie_group_map,
        multi_state_strategy=multi_state_strategy,
      )

    # Compute score from logits
    log_probability = jax.nn.log_softmax(logits, axis=-1)[..., :20]
    score = -(sequence[..., :20] * log_probability).sum(-1)
    masked_score_sum = (score * mask).sum(-1)
    mask_sum = mask.sum() + SCORE_EPS

    return masked_score_sum / mask_sum, logits, decoding_order

  return cast("ScoringFn", score_sequence)


make_score_sequence = make_score_fn
