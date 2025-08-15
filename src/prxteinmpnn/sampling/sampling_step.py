"""Sample sequences from a structure using the ProteinMPNN model.

prxteinmpnn.sampling.sampling
"""

from collections.abc import Callable
from functools import partial
from typing import Any, cast

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from prxteinmpnn.model.decoder import RunAutoregressiveDecoderFn, RunConditionalDecoderFn
from prxteinmpnn.utils.types import (
  CEELoss,
  Logits,
  ModelParameters,
  NodeFeatures,
  ProteinSequence,
  SequenceEdgeFeatures,
)

from .config import SamplingConfig, SamplingEnum
from .initialize import SamplingModelPassOutput
from .ste import ste_loss

SampleModelPassOnlyPRNGFn = Callable[[PRNGKeyArray], SamplingModelPassOutput]

SamplingStepState = tuple[
  PRNGKeyArray,
  SequenceEdgeFeatures,
  NodeFeatures,
  ProteinSequence,
  Logits,
]
SamplingStepInput = tuple[
  Any,
  ...,
]

SamplingStepFn = Callable[
  [*SamplingStepInput],
  SamplingStepState,
]


def temperature_sample(
  prng_key: PRNGKeyArray,
  decoder: RunAutoregressiveDecoderFn,
  sample_model_pass_fn_only_prng: SampleModelPassOnlyPRNGFn,
  temperature: float,
) -> SamplingStepState:
  """Single autoregressive sampling step with temperature scaling.

  Args:
    prng_key: Random key for JAX operations.
    decoder: Decoder function to update node features. Preloaded autoregressive decoder.
    sample_model_pass_fn_only_prng: Function to run a single pass through the model.
    temperature: Temperature for scaling logits.

  Returns:
    Updated carry state and None for scan output.

  Example:
    carry = (rng_key, edge_features, node_features, sequence, logits)
    sample_step = partial(
      sample_temperature_step,
      decoder=decoder,
      neighbor_indices=neighbor_indices,
      mask=mask,
      autoregressive_mask=autoregressive_mask,
      model_parameters=model_parameters,
      temperature=temperature,
    )
    final_carry, _ = jax.lax.fori_loop(
      0,
      iterations,
      sample_step,
      carry,
    )

  """
  (
    node_features,
    edge_features,
    neighbor_indices,
    mask,
    autoregressive_mask,
    decoding_key,
  ) = sample_model_pass_fn_only_prng(
    prng_key,
  )

  output_sequence, logits = decoder(
    decoding_key,
    node_features,
    edge_features,
    neighbor_indices,
    mask,
    autoregressive_mask,
    temperature,
  )

  output_sequence = output_sequence.argmax(axis=-1).astype(jnp.int8)

  return prng_key, edge_features, node_features, output_sequence, logits


def ste_sample(
  initial_carry: SamplingStepState,
  decoder: RunAutoregressiveDecoderFn,
  sample_model_pass_fn_only_prng: SampleModelPassOnlyPRNGFn,
  learning_rate: float,
  iterations: int,
) -> SamplingStepState:
  """Autoregressive sampling with straight-through estimator.

  Args:
    initial_carry: Tuple containing initial state (rng_key, edge_features, node_features, sequence,
      logits).
    decoder: Decoder function to update node features.
    model_parameters: Model parameters for the model.
    sample_model_pass_fn_only_prng: Function to run a single pass through the model.
    learning_rate: Learning rate for updating logits.
    target_logits: Target logits for the straight-through estimator.
    iterations: Number of iterations for the straight-through estimator.


  Returns:
    Updated carry state and None for scan output.

  Example:
    carry = (rng_key, edge_features, node_features, sequence, logits)
    sample_step = partial(
      sample_straight_through_estimator_step,
      decoder=decoder,
      neighbor_indices=neighbor_indices,
      mask=mask,
      autoregressive_mask=autoregressive_mask,
      model_parameters=model_parameters,
      learning_rate=learning_rate,
    )
    final_carry, _ = jax.lax.fori_loop(
      0,
      iterations,
      sample_step,
      carry,

  """

  def sampling_step(_i: int, carry: SamplingStepState) -> SamplingStepState:
    """Single sampling step for the straight-through estimator."""
    (
      current_key,
      edge_features,
      node_features,
      _sequence,
      current_logits,
    ) = carry

    (
      node_features,
      edge_features,
      neighbor_indices,
      mask,
      autoregressive_mask,
      next_rng_key,
    ) = sample_model_pass_fn_only_prng(
      current_key,
    )

    @jax.jit
    def loss_fn(input_logits: Logits) -> CEELoss:
      """Compute the loss for the straight-through estimator."""
      output_logits, _ = decoder(
        next_rng_key,
        node_features,
        edge_features,
        neighbor_indices,
        mask,
        autoregressive_mask,
        1.0,
      )

      return ste_loss(output_logits, input_logits, mask)

    _, grad = jax.value_and_grad(loss_fn, has_aux=False)(
      current_logits,
    )

    updated_logits = current_logits - learning_rate * grad

    updated_sequence = updated_logits.argmax(axis=-1).astype(jnp.int8)
    return (
      next_rng_key,
      edge_features,
      node_features,
      updated_sequence,
      updated_logits,
    )

  return jax.lax.fori_loop(0, iterations, sampling_step, initial_carry)


def preload_sampling_step_decoder(
  decoder: RunConditionalDecoderFn | RunAutoregressiveDecoderFn,
  sample_model_pass_fn_only_prng: SampleModelPassOnlyPRNGFn,
  model_parameters: ModelParameters,
  sampling_config: SamplingConfig,
) -> SamplingStepFn:
  """Preload the sampling step decoder."""
  match sampling_config.sampling_strategy:
    case SamplingEnum.TEMPERATURE:
      """Get the temperature sampling step function."""
      decoder = cast("RunAutoregressiveDecoderFn", decoder)
      decoding_loaded_step_fn = partial(
        temperature_sample,
        decoder=decoder,
        sample_model_pass_fn_only_prng=sample_model_pass_fn_only_prng,
        temperature=sampling_config.temperature,  # type: ignore[arg-type]
      )
    case SamplingEnum.STRAIGHT_THROUGH:
      """Get the straight-through sampling step function."""
      decoder = cast("RunAutoregressiveDecoderFn", decoder)
      decoding_loaded_step_fn = partial(
        ste_sample,
        decoder=decoder,
        sample_model_pass_fn_only_prng=sample_model_pass_fn_only_prng,
        learning_rate=sampling_config.learning_rate,  # type: ignore[arg-type]
        iterations=sampling_config.iterations,  # type: ignore[arg-type]
      )
    case SamplingEnum.BEAM_SEARCH:
      """Beam search sampling is not implemented yet."""
      msg = "Beam search sampling is not implemented yet."
      raise NotImplementedError(msg)
    case SamplingEnum.GREEDY:
      """Greedy sampling is not implemented yet."""
      msg = "Greedy sampling is not implemented yet."
      raise NotImplementedError(msg)
    case SamplingEnum.TOP_K:
      """Top-k sampling is not implemented yet."""
      msg = "Top-k sampling is not implemented yet."
      raise NotImplementedError(msg)
    case _:
      """Raise an error for unknown sampling strategies."""
      msg = f"Unknown sampling strategy: {sampling_config.sampling_strategy}"
      raise ValueError(msg)

  return lambda *args: decoding_loaded_step_fn(*args)
