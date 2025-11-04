"""Factory for creating sequence sampling and optimization functions for PrxteinMPNN."""

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Literal, cast

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int, PRNGKeyArray

from prxteinmpnn.sampling.adapter import get_decoder_fn, get_encoder_fn
from prxteinmpnn.utils.autoregression import generate_ar_mask
from prxteinmpnn.utils.decoding_order import DecodingOrderFn
from prxteinmpnn.utils.types import (
  AlphaCarbonMask,
  BackboneNoise,
  ChainIndex,
  DecodingOrder,
  InputBias,
  Logits,
  Model,
  ModelParameters,
  ProteinSequence,
  ResidueIndex,
  StructureAtomicCoordinates,
)

from .initialize import sampling_encode
from .sampling_step import preload_sampling_step_decoder
from .ste_optimize import make_optimize_sequence_fn

"""Factory for creating sequence sampling and optimization functions."""


if TYPE_CHECKING:
  from prxteinmpnn.types import (
    RunAutoregressiveDecoderFn,
    RunConditionalDecoderFn,
  )
# Simplified type hints
SamplerInputs = tuple[
  PRNGKeyArray,
  StructureAtomicCoordinates,
  AlphaCarbonMask,
  ResidueIndex,
  ChainIndex,
  int,
  InputBias | None,
  Int | None,
  BackboneNoise | None,
  Int | None,
  Float | None,
  Float | None,
]
SamplerFn = Callable[..., tuple[ProteinSequence, Logits, DecodingOrder]]
EncodingSamplerFn = Callable[
  [
    PRNGKeyArray,
    StructureAtomicCoordinates,
    AlphaCarbonMask,
    ResidueIndex,
    ChainIndex,
    int,
    BackboneNoise | None,
  ],
  tuple[ModelParameters, DecodingOrder],
]


def make_sample_sequences(
  model: Model,
  decoding_order_fn: DecodingOrderFn,
  sampling_strategy: Literal["temperature", "straight_through"] = "temperature",
  num_encoder_layers: int = 3,
  num_decoder_layers: int = 3,
) -> SamplerFn:
  """Create a function to sample or optimize sequences from a structure."""
  encoder = get_encoder_fn(
    model,
    attention_mask_type="cross",
    num_encoder_layers=num_encoder_layers,
  )

  conditional_decoder: RunConditionalDecoderFn = cast(
    "RunConditionalDecoderFn",
    get_decoder_fn(
      model,
      attention_mask_type="conditional",
      decoding_approach="conditional",
      num_decoder_layers=num_decoder_layers,
    ),
  )

  autoregressive_decoder = cast(
    "RunAutoregressiveDecoderFn",
    get_decoder_fn(
      model,
      attention_mask_type=None,
      decoding_approach="autoregressive",
      num_decoder_layers=num_decoder_layers,
    ),
  )

  # For now, internal functions still use ModelParameters PyTree
  # In future refactoring, these will also be updated to use Model union type

  # Extract model parameters for functions that haven't been migrated yet
  # For Equinox models, we'll need to handle this differently in the future
  # For now, we assume PyTree models for these internal functions
  model_params = model  # type: ignore[assignment]

  sample_model_pass = sampling_encode(encoder=encoder, decoding_order_fn=decoding_order_fn)

  optimize_seq_fn = make_optimize_sequence_fn(
    decoder=conditional_decoder,
    decoding_order_fn=decoding_order_fn,
    model_parameters=model_params,  # type: ignore[arg-type]
  )

  # --- Top-level imports for group-based sampling ---

  # --- Top-level imports for group-based sampling ---
  from prxteinmpnn.sampling.sampling_step import group_sampling_step
  from prxteinmpnn.utils.autoregression import (
    get_decoding_step_map,
    make_autoregressive_mask,
  )
  from prxteinmpnn.utils.decoding_order import get_decoding_order

  @partial(jax.jit, static_argnames=("k_neighbors",))
  def sample_or_optimize_fn(
    prng_key: PRNGKeyArray,
    structure_coordinates: StructureAtomicCoordinates,
    mask: AlphaCarbonMask,
    residue_index: ResidueIndex,
    chain_index: ChainIndex,
    k_neighbors: int = 48,
    bias: InputBias | None = None,
    fixed_positions: Int | None = None,
    backbone_noise: BackboneNoise | None = None,
    iterations: Int | None = None,
    learning_rate: Float | None = None,
    temperature: Float | None = None,
    tie_group_map: jnp.ndarray | None = None,
    spec: object | None = None,
  ) -> tuple[ProteinSequence, Logits, DecodingOrder]:
    """Dispatches to either the optimization or sampling function."""
    bias, fixed_positions, iterations, learning_rate, temperature = (
      bias
      if bias is not None
      else jnp.zeros((structure_coordinates.shape[0], 21), dtype=jnp.float32),
      fixed_positions if fixed_positions is not None else jnp.array([], dtype=jnp.int32),
      iterations if iterations is not None else 1,
      learning_rate if learning_rate is not None else 1e-4,
      temperature if temperature is not None else 1.0,
    )

    (
      node_features,
      edge_features,
      neighbor_indices,
      decoding_order,
      _,
      next_rng_key,
    ) = sample_model_pass(
      prng_key,
      model_params,  # type: ignore[arg-type]
      structure_coordinates,
      mask,
      residue_index,
      chain_index,
      None,
      k_neighbors,
      backbone_noise,
    )

    # If tie_group_map is provided, use group-based decoding
    if tie_group_map is not None and spec is not None:
      # Step 1: get group_decoding_order
      group_decoding_order = get_decoding_order(tie_group_map, next_rng_key)
      # Step 2: get decoding_step_map
      decoding_step_map = get_decoding_step_map(tie_group_map, group_decoding_order)
      # Step 3: get AR mask
      ar_mask = make_autoregressive_mask(decoding_step_map)
      # Step 4: scan over group_decoding_order
      carry = (
        next_rng_key,
        jnp.zeros_like(residue_index),
        autoregressive_decoder,
        tie_group_map,
        ar_mask,
        temperature,
      )

      def scan_body(
        carry: tuple,
        group_id: jnp.ndarray,
      ) -> tuple[tuple, None]:
        return group_sampling_step(carry, group_id)

      carry, _ = jax.lax.scan(scan_body, carry, group_decoding_order)
      _, output_sequence, *_ = carry
      # output_sequence may be a tuple, ensure correct type
      if isinstance(output_sequence, tuple):
        output_sequence = output_sequence[0]
      output_logits = jnp.zeros((output_sequence.shape[0], 21))  # Placeholder
      return output_sequence, output_logits, decoding_order

    if sampling_strategy == "straight_through":
      output_sequence, output_logits = optimize_seq_fn(
        next_rng_key,
        node_features,
        edge_features,
        neighbor_indices,
        mask,
        iterations,
        learning_rate,
        temperature,
      )
      output_sequence = output_sequence.argmax(axis=-1).astype(jnp.int8)
      return output_sequence, output_logits, decoding_order
    sample_model_pass_fn = partial(
      sample_model_pass,
      model_parameters=model_params,  # type: ignore[arg-type]
      structure_coordinates=structure_coordinates,  # type: ignore[arg-type]
      mask=mask,  # type: ignore[arg-type]
      residue_index=residue_index,  # type: ignore[arg-type]
      chain_index=chain_index,  # type: ignore[arg-type]
      k_neighbors=k_neighbors,  # type: ignore[arg-type]
      backbone_noise=backbone_noise,  # type: ignore[arg-type]
    )
    sample_step = preload_sampling_step_decoder(
      autoregressive_decoder,
      sample_model_pass_fn,
      sampling_strategy=sampling_strategy,
      temperature=temperature,
    )
    _, output_sequence, output_logits = sample_step(prng_key=next_rng_key, bias=bias)
    return output_sequence, output_logits, decoding_order

  return sample_or_optimize_fn  # type: ignore[return-value]


def make_encoding_sampling_split_fn(
  model: Model,
  decoding_order_fn: DecodingOrderFn,
  sampling_strategy: Literal["temperature", "straight_through"] = "temperature",
  num_decoder_layers: int = 3,
) -> tuple[EncodingSamplerFn, Callable]:
  """Create functions for encoding and sampling, intended for averaging encodings.

  This function returns two functions: `encode` and `sample_from_features`.
  `encode` runs the encoder part of the model to get structural features.
  `sample_from_features` runs sampling on provided features.
  This separation allows for averaging the results of `encode` from multiple runs
  (e.g., with different noise) before a single `sample_from_features` call.

  Args:
    model: A Model (either PyTree parameters or PrxteinMPNN Equinox module).
    decoding_order_fn: A function that generates the decoding order.
    sampling_strategy: The sampling strategy to use ("temperature" or "straight_through").
    num_decoder_layers: The number of decoder layers to use. Defaults to 3.

  Returns:
    A tuple containing two functions: (`encode`, `sample_from_features`).

  """
  # Extract model parameters for functions that haven't been migrated yet
  model_params = model  # type: ignore[assignment]

  encoder = get_encoder_fn(
    model,
    attention_mask_type="cross",
    num_encoder_layers=3,
  )
  conditional_decoder = cast(
    "RunConditionalDecoderFn",
    get_decoder_fn(
      model,
      attention_mask_type="conditional",
      decoding_approach="conditional",
      num_decoder_layers=num_decoder_layers,
    ),
  )

  autoregressive_decoder = cast(
    "RunAutoregressiveDecoderFn",
    get_decoder_fn(
      model,
      attention_mask_type=None,
      decoding_approach="autoregressive",
      num_decoder_layers=num_decoder_layers,
    ),
  )

  sample_model_pass = sampling_encode(encoder=encoder, decoding_order_fn=decoding_order_fn)

  optimize_seq_fn = make_optimize_sequence_fn(
    decoder=conditional_decoder,
    decoding_order_fn=decoding_order_fn,
    model_parameters=model_params,  # type: ignore[arg-type]
  )

  @partial(jax.jit, static_argnames=("k_neighbors",))
  def encode(
    prng_key: PRNGKeyArray,
    structure_coordinates: StructureAtomicCoordinates,
    mask: AlphaCarbonMask,
    residue_index: ResidueIndex,
    chain_index: ChainIndex,
    k_neighbors: int = 48,
    backbone_noise: BackboneNoise | None = None,
  ) -> tuple[ModelParameters, DecodingOrder]:
    """Encode the structure to get features for sampling.

    Returns:
        A tuple of (encoded_features, decoding_order) where encoded_features is a
        dict-like PyTree containing node_features, edge_features, neighbor_indices, and mask.

    """
    if backbone_noise is None:
      backbone_noise = jnp.array(0.0, dtype=jnp.float32)

    (
      node_features,
      edge_features,
      neighbor_indices,
      decoding_order,
      _,
      _,
    ) = sample_model_pass(
      prng_key,
      model_params,  # type: ignore[arg-type]
      structure_coordinates,
      mask,
      residue_index,
      chain_index,
      None,
      k_neighbors,
      backbone_noise,
    )

    # Return features as a dict-like structure
    encoded_features = {
      "node_features": node_features,
      "edge_features": edge_features,
      "neighbor_indices": neighbor_indices,
      "mask": mask,
    }

    return encoded_features, decoding_order  # type: ignore[return-value]

  @partial(jax.jit, static_argnames=("sampling_strategy",))
  def sample_from_features(
    prng_key: PRNGKeyArray,
    encoded_features: ModelParameters,
    decoding_order: DecodingOrder,
    bias: InputBias | None = None,
    iterations: Int | None = None,
    learning_rate: Float | None = None,
    temperature: Float | None = None,
    sampling_strategy: Literal["temperature", "straight_through"] = sampling_strategy,
  ) -> tuple[ProteinSequence, Logits]:
    """Sample sequences given encoded features.

    Args:
        prng_key: Random key for sampling.
        encoded_features: Dict containing node_features, edge_features,
            neighbor_indices, and mask from the encode function.
        decoding_order: The decoding order to use.
        bias: Optional bias to add to logits.
        iterations: Number of iterations for straight_through optimization.
        learning_rate: Learning rate for straight_through optimization.
        temperature: Temperature for sampling.
        sampling_strategy: "temperature" or "straight_through".

    Returns:
        A tuple of (sampled_sequence, logits).

    """
    node_features = encoded_features["node_features"]
    edge_features = encoded_features["edge_features"]
    neighbor_indices = encoded_features["neighbor_indices"]
    mask = encoded_features["mask"]

    # Set defaults
    bias = bias if bias is not None else jnp.zeros((node_features.shape[0], 21), dtype=jnp.float32)
    iterations = iterations if iterations is not None else 1
    learning_rate = learning_rate if learning_rate is not None else 1e-4
    temperature = temperature if temperature is not None else 1.0

    if sampling_strategy == "straight_through":
      output_sequence, output_logits = optimize_seq_fn(
        prng_key,
        node_features,
        edge_features,
        neighbor_indices,
        mask,
        iterations,
        learning_rate,
        temperature,
      )
      output_sequence = output_sequence.argmax(axis=-1).astype(jnp.int8)
      return output_sequence, output_logits

    # Temperature sampling - use autoregressive decoder with preloaded sampling step
    # Create a partial function that returns the pre-encoded features
    ar_mask = generate_ar_mask(decoding_order)

    dummy_sample_model_pass_fn = partial(
      lambda *_args, **_kwargs: (
        node_features,
        edge_features,
        neighbor_indices,
        mask,
        ar_mask,
        prng_key,
      ),
    )

    sample_step = preload_sampling_step_decoder(
      autoregressive_decoder,
      dummy_sample_model_pass_fn,
      sampling_strategy=sampling_strategy,
      temperature=temperature,
    )

    _, output_sequence, output_logits = sample_step(prng_key=prng_key, bias=bias)
    return output_sequence, output_logits

  return encode, sample_from_features
