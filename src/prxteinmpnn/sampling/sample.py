"""Factory for creating sequence sampling functions for PrxteinMPNN."""

from collections.abc import Callable
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int, PRNGKeyArray

from prxteinmpnn.model import PrxteinMPNN
from prxteinmpnn.sampling.conditional_logits import make_encoding_conditional_logits_split_fn
from prxteinmpnn.sampling.ste_optimize import make_optimize_sequence_fn
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
  sampling_strategy: Literal["temperature", "straight_through"] = "temperature",
  _num_encoder_layers: int = 3,
  _num_decoder_layers: int = 3,
) -> SamplerFn:
  """Create a function to sample sequences from a structure using PrxteinMPNN.

  Args:
    model: A PrxteinMPNN Equinox model instance.
    decoding_order_fn: Function to generate decoding order (default: random).
    sampling_strategy: Sampling strategy - "temperature" or "straight_through".
    _num_encoder_layers: Deprecated, ignored (kept for API compatibility).
    _num_decoder_layers: Deprecated, ignored (kept for API compatibility).

  Returns:
    A function that samples sequences from structures.

  Example:
    >>> from prxteinmpnn.io.weights import load_model
    >>> model = load_model()
    >>> sample_fn = make_sample_sequences(model, sampling_strategy="temperature")
    >>> seq, logits, order = sample_fn(key, coords, mask, res_idx, chain_idx)
    >>>
    >>> # For optimization
    >>> optimize_fn = make_sample_sequences(model, sampling_strategy="straight_through")
    >>> seq, logits, order = optimize_fn(
    ...     key, coords, mask, res_idx, chain_idx,
    ...     iterations=100, learning_rate=0.01
    ... )

  """
  # Create optimization function if needed
  if sampling_strategy == "straight_through":
    optimize_fn = make_optimize_sequence_fn(model, decoding_order_fn)

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
      iterations: Int | None = None,
      learning_rate: Float | None = None,
      temperature: Float | None = None,
    ) -> tuple[ProteinSequence, Logits, DecodingOrder]:
      """Optimize a sequence using straight-through estimation.

      Args:
        prng_key: JAX random key.
        structure_coordinates: Atomic coordinates (N, 4, 3).
        mask: Alpha carbon mask indicating valid residues.
        residue_index: Residue indices.
        chain_index: Chain indices.
        _k_neighbors: Deprecated (kept for API compatibility).
        bias: Not used in straight_through mode.
        fixed_positions: Not implemented yet.
        backbone_noise: Optional noise for backbone coordinates.
        iterations: Number of optimization steps (default: 100).
        learning_rate: Learning rate for optimization (default: 0.01).
        temperature: Temperature for STE sampling (default: 1.0).

      Returns:
        Tuple of (optimized sequence, final logits, decoding order).

      """
      del bias, fixed_positions, _k_neighbors  # Not used in optimization

      # Set defaults
      if iterations is None:
        iterations = jnp.array(100, dtype=jnp.int32)
      if learning_rate is None:
        learning_rate = jnp.array(0.01, dtype=jnp.float32)
      if temperature is None:
        temperature = jnp.array(1.0, dtype=jnp.float32)

      # Generate decoding order for return value
      decoding_order, prng_key = decoding_order_fn(prng_key, structure_coordinates.shape[0])

      # Run optimization
      optimized_sequence, final_logits, _ = optimize_fn(
        prng_key,
        structure_coordinates,
        mask,
        residue_index,
        chain_index,
        iterations,
        learning_rate,
        temperature,
        backbone_noise,
      )

      return optimized_sequence, final_logits, decoding_order

    return sample_sequences

  # Temperature sampling (default)
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


def make_encoding_sampling_split_fn(
  model_parameters: PrxteinMPNN,
  decoding_order_fn: DecodingOrderFn | None = None,
  sampling_strategy: Literal["temperature", "straight_through"] = "temperature",
) -> tuple[Callable, Callable]:
  """Create separate encoding and sampling functions for averaged encodings.

  This splits the sampling process into two parts:
  1. Encoding: Structure -> Encoder features (can be averaged across noise levels)
  2. Sampling: (Encoder features, PRNGKey) -> Sequences

  This separation allows:
  - Averaging encoder features across multiple noise levels
  - Efficient reuse of encoder output for multiple samples
  - Lower memory usage when sampling many sequences

  Args:
    model_parameters: A PrxteinMPNN Equinox model instance.
    decoding_order_fn: Function to generate decoding order (default: random).
    sampling_strategy: Sampling strategy - "temperature" or "straight_through".

  Returns:
    Tuple of (encode_fn, sample_fn) where:
      - encode_fn: Computes encoder features from structure
      - sample_fn: Samples sequences from cached encoder features

  Example:
    >>> from prxteinmpnn.io.weights import load_model
    >>> model = load_model()
    >>> encode_fn, sample_fn = make_encoding_sampling_split_fn(
    ...     model, sampling_strategy="temperature"
    ... )
    >>> # Encode once
    >>> encoding = encode_fn(
    ...     key, coords, mask, res_idx, chain_idx,
    ...     k_neighbors=48, backbone_noise=0.1
    ... )
    >>> # Sample multiple sequences using same encoding
    >>> seq1 = sample_fn(key1, encoding, order1, temperature=0.1)
    >>> seq2 = sample_fn(key2, encoding, order2, temperature=0.5)

  """
  del sampling_strategy, decoding_order_fn  # Currently only temperature sampling supported

  # Get the encoder/decoder split functions
  encode_logits_fn, decode_logits_fn = make_encoding_conditional_logits_split_fn(
    model_parameters,
  )

  @partial(jax.jit, static_argnames=("k_neighbors",))
  def encode_fn(
    prng_key: PRNGKeyArray,
    structure_coordinates: StructureAtomicCoordinates,
    mask: AlphaCarbonMask,
    residue_index: ResidueIndex,
    chain_index: ChainIndex,
    k_neighbors: int = 48,
    backbone_noise: BackboneNoise | None = None,
  ) -> tuple:
    """Encode structure to get encoder features.

    Args:
      prng_key: JAX random key for feature extraction.
      structure_coordinates: Atomic coordinates (N, 4, 3).
      mask: Alpha carbon mask indicating valid residues.
      residue_index: Residue indices.
      chain_index: Chain indices.
      k_neighbors: Number of nearest neighbors (currently not used, for API compat).
      backbone_noise: Optional noise for backbone coordinates.

    Returns:
      Encoder features tuple that can be passed to sample_fn.

    """
    del k_neighbors  # Not used in current implementation

    # Run encoder (encode_fn expects structure,mask,res_idx,chain_idx,...)
    return encode_logits_fn(
      structure_coordinates,
      mask,
      residue_index,
      chain_index,
      backbone_noise=backbone_noise,
      prng_key=prng_key,
    )

  @partial(jax.jit)
  def sample_fn(
    prng_key: PRNGKeyArray,
    encoded_features: tuple,
    decoding_order: DecodingOrder,
    bias: InputBias | None = None,
    iterations: Int | None = None,
    learning_rate: Float | None = None,
    temperature: Float | None = None,
    sampling_strategy: Literal["temperature", "straight_through"] = "temperature",
  ) -> ProteinSequence:
    """Sample a sequence from cached encoder features.

    Args:
      prng_key: JAX random key for sampling.
      encoded_features: Encoder features from encode_fn.
      decoding_order: Decoding order for autoregressive sampling.
      bias: Optional bias to add to logits (N, 21).
      iterations: For STE optimization (not used in temperature sampling).
      learning_rate: For STE optimization (not used in temperature sampling).
      temperature: Temperature for sampling (default: 1.0).
      sampling_strategy: "temperature" or "straight_through" (currently only temperature).

    Returns:
      Sampled sequence as integer array (N,).

    """
    del iterations, learning_rate, sampling_strategy  # Not used in current implementation

    # Set default temperature
    if temperature is None:
      temperature = jnp.array(1.0, dtype=jnp.float32)

    # Generate autoregressive mask from decoding order
    autoregressive_mask = generate_ar_mask(decoding_order)

    # Initialize sequence with random amino acids (for one-hot encoding)
    seq_length = autoregressive_mask.shape[0]
    _, prng_key = jax.random.split(prng_key)
    initial_seq = jax.random.randint(
      prng_key,
      shape=(seq_length,),
      minval=0,
      maxval=21,
      dtype=jnp.int32,
    )

    # Autoregressive sampling loop
    def sample_step(
      i: int,
      state: tuple[ProteinSequence, PRNGKeyArray],
    ) -> tuple[ProteinSequence, PRNGKeyArray]:
      """Sample one position in the decoding order."""
      sequence, key = state

      # Decode current sequence to get logits
      logits = decode_logits_fn(encoded_features, sequence, autoregressive_mask)

      # Get position to sample
      pos = decoding_order[i]

      # Apply bias if provided
      if bias is not None:
        logits = logits.at[pos].add(bias[pos])

      # Apply temperature and sample
      scaled_logits = logits[pos] / temperature
      key, subkey = jax.random.split(key)
      sampled_aa = jax.random.categorical(subkey, scaled_logits).astype(jnp.int8)

      # Update sequence
      updated_seq = sequence.at[pos].set(sampled_aa)

      return updated_seq, key

    # Run autoregressive sampling
    final_seq, _ = jax.lax.fori_loop(
      0,
      seq_length,
      sample_step,
      (initial_seq, prng_key),
    )

    return final_seq

  return encode_fn, sample_fn
