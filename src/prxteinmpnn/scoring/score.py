"""Score a given sequence on a structure using the ProteinMPNN model."""

from collections.abc import Callable
from functools import partial

import jax
from jaxtyping import Float, PRNGKeyArray

from prxteinmpnn.model import PrxteinMPNN
from prxteinmpnn.utils.autoregression import generate_ar_mask
from prxteinmpnn.utils.decoding_order import DecodingOrderFn, random_decoding_order
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
  ],
  tuple[Float, Logits, DecodingOrder],
]


SCORE_EPS = 1e-8


def make_score_sequence(
  model: PrxteinMPNN,
  decoding_order_fn: DecodingOrderFn = random_decoding_order,
  _num_encoder_layers: int = 3,
  _num_decoder_layers: int = 3,
) -> ScoringFn:
  """Create a function to score a sequence on a structure using PrxteinMPNN.

  Args:
    model: A PrxteinMPNN Equinox model instance.
    decoding_order_fn: Function to generate decoding order (default: random).
    _num_encoder_layers: Deprecated, ignored (kept for API compatibility).
    _num_decoder_layers: Deprecated, ignored (kept for API compatibility).

  Returns:
    A function that scores sequences on structures.

  Example:
    >>> from prxteinmpnn.io.weights import load_model
    >>> model = load_model()
    >>> score_fn = make_score_sequence(model)
    >>> score, logits, order = score_fn(key, seq, coords, mask, res_idx, chain_idx)

  """

  @partial(jax.jit, static_argnames=("_k_neighbors",))
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

    Returns:
      Tuple of (average score, logits, decoding order).

    Example:
      >>> score, logits, order = score_sequence(
      ...     key, seq, coords, mask, res_idx, chain_idx
      ... )

    """
    decoding_order, prng_key = decoding_order_fn(prng_key, sequence.shape[0])
    autoregressive_mask = generate_ar_mask(decoding_order) if ar_mask is None else ar_mask

    # Ensure sequence is one-hot encoded
    if sequence.ndim == 1:
      sequence = jax.nn.one_hot(sequence, num_classes=21)

    # Run model in conditional mode (scoring a given sequence)
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
    )

    # Compute score from logits
    log_probability = jax.nn.log_softmax(logits, axis=-1)[..., :20]
    score = -(sequence[..., :20] * log_probability).sum(-1)
    masked_score_sum = (score * mask).sum(-1)
    mask_sum = mask.sum() + SCORE_EPS

    return masked_score_sum / mask_sum, logits, decoding_order

  return score_sequence
