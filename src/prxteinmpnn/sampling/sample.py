"""Factory for creating sequence sampling functions for PrxteinMPNN."""

from collections.abc import Callable
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Float, PRNGKeyArray

from prxteinmpnn.model import PrxteinMPNN
from prxteinmpnn.utils.autoregression import generate_ar_mask
from prxteinmpnn.utils.decoding_order import DecodingOrderFn, random_decoding_order
from prxteinmpnn.utils.types import (
  AlphaCarbonMask,
  BackboneNoise,
  ChainIndex,
  DecodingOrder,
  InputBias,
  Logits,
  ProteinSequence,
  ResidueIndex,
  StructureAtomicCoordinates,
)

SamplerFn = Callable[..., tuple[ProteinSequence, Logits, DecodingOrder]]


def make_sample_sequences(
  model: PrxteinMPNN,
  decoding_order_fn: DecodingOrderFn = random_decoding_order,
  sampling_strategy: Literal["temperature"] = "temperature",
  _num_encoder_layers: int = 3,
  _num_decoder_layers: int = 3,
) -> SamplerFn:
  """Create a function to sample sequences from a structure using PrxteinMPNN.

  Args:
    model: A PrxteinMPNN Equinox model instance.
    decoding_order_fn: Function to generate decoding order (default: random).
    sampling_strategy: Sampling strategy, currently only "temperature" is supported.
    _num_encoder_layers: Deprecated, ignored (kept for API compatibility).
    _num_decoder_layers: Deprecated, ignored (kept for API compatibility).

  Returns:
    A function that samples sequences from structures.

  Example:
    >>> from prxteinmpnn.io.weights import load_model
    >>> model = load_model()
    >>> sample_fn = make_sample_sequences(model)
    >>> seq, logits, order = sample_fn(key, coords, mask, res_idx, chain_idx)

  """
  if sampling_strategy != "temperature":
    msg = f"Unsupported sampling strategy: {sampling_strategy}"
    raise NotImplementedError(msg)

  @partial(jax.jit, static_argnames=("k_neighbors",))
  def sample_sequences(
    prng_key: PRNGKeyArray,
    structure_coordinates: StructureAtomicCoordinates,
    mask: AlphaCarbonMask,
    residue_index: ResidueIndex,
    chain_index: ChainIndex,
    _k_neighbors: int = 48,
    bias: InputBias | None = None,
    fixed_positions: jnp.ndarray | None = None,
    backbone_noise: BackboneNoise | None = None,
    temperature: Float | None = None,
  ) -> tuple[ProteinSequence, Logits, DecodingOrder]:
    """Sample a sequence from a structure using the ProteinMPNN model.

    Args:
      prng_key: JAX random key.
      structure_coordinates: Atomic coordinates (N, 4, 3).
      mask: Alpha carbon mask indicating valid residues.
      residue_index: Residue indices.
      chain_index: Chain indices.
      _k_neighbors: Deprecated, model handles internally (kept for API compatibility).
      bias: Optional bias to add to logits (N, 21).
      fixed_positions: Optional mask for positions to keep fixed (not implemented yet).
      backbone_noise: Optional noise for backbone coordinates.
      temperature: Temperature for sampling (default: 1.0).

    Returns:
      Tuple of (sampled sequence, logits, decoding order).

    Example:
      >>> seq, logits, order = sample_sequences(
      ...     key, coords, mask, res_idx, chain_idx, temperature=0.1
      ... )

    """
    del fixed_positions  # Not yet implemented

    # Set default temperature
    if temperature is None:
      temperature = jnp.array(1.0, dtype=jnp.float32)

    # Generate decoding order
    decoding_order, prng_key = decoding_order_fn(prng_key, structure_coordinates.shape[0])
    autoregressive_mask = generate_ar_mask(decoding_order)

    # Run model in autoregressive mode (sampling)
    sampled_sequence, logits = model(
      structure_coordinates,
      mask,
      residue_index,
      chain_index,
      decoding_approach="autoregressive",
      prng_key=prng_key,
      ar_mask=autoregressive_mask,
      temperature=temperature,
      bias=bias,
      backbone_noise=backbone_noise,
    )

    # Convert one-hot to integer sequence if needed
    one_hot_ndim = 2
    if sampled_sequence.ndim == one_hot_ndim:
      sampled_sequence = sampled_sequence.argmax(axis=-1).astype(jnp.int8)

    return sampled_sequence, logits, decoding_order

  return sample_sequences
