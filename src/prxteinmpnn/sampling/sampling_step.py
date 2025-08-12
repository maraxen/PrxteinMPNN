"""Sample sequences from a structure using the ProteinMPNN model.

prxteinmpnn.sampling.sampling
"""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from prxteinmpnn.model.decoder import RunConditionalDecoderFn
from prxteinmpnn.model.projection import final_projection
from prxteinmpnn.utils.types import (
  CEELoss,
  Logits,
  ModelParameters,
  NodeFeatures,
  OneHotProteinSequence,
  ProteinSequence,
  SequenceEdgeFeatures,
)

from .config import SamplingConfig, SamplingEnum
from .initialize import SamplingModelPassOutput
from .ste import ste_loss, straight_through_estimator

SamplingStepState = tuple[
  PRNGKeyArray,
  SequenceEdgeFeatures,
  NodeFeatures,
  ProteinSequence,
  Logits,
]
SamplingStepInput = tuple[int, SamplingStepState]

SamplingStepFn = Callable[
  [*SamplingStepInput],
  SamplingStepState,
]

SampleModelPassOnlyPRNGFn = Callable[[PRNGKeyArray], SamplingModelPassOutput]


def sample_temperature_step(
  _i: int,
  carry: SamplingStepState,
  decoder: RunConditionalDecoderFn,
  model_parameters: ModelParameters,
  sample_model_pass_fn_only_prng: SampleModelPassOnlyPRNGFn,
  temperature: float,
) -> SamplingStepState:
  """Single autoregressive sampling step with temperature scaling.

  Args:
    _i: Current iteration.
    carry: Tuple containing current state (rng_key, edge_features, node_features, sequence, logits).
    decoder: Decoder function to update node features.
    model_parameters: Model parameters for the model.
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
  current_key, edge_features, node_features, _, _ = carry

  sequence = jnp.zeros((node_features.shape[0], 21), dtype=jnp.float32)
  logits = jnp.zeros((node_features.shape[0], 21), dtype=jnp.float32)

  (
    node_features,
    edge_features,
    neighbor_indices,
    mask,
    autoregressive_mask,
    next_prng_key,
  ) = sample_model_pass_fn_only_prng(
    current_key,
  )

  decoding_order = jnp.argsort(jnp.sum(autoregressive_mask, axis=1))

  def update_sequence(
    i: int,
    inner_carry: tuple[PRNGKeyArray, OneHotProteinSequence, Logits],
  ) -> tuple[PRNGKeyArray, ProteinSequence, Logits]:
    """Update the sequence at the current position."""
    current_prng_key, sequence, logits = inner_carry
    position = decoding_order[i]
    updated_node_features = decoder(
      node_features,
      edge_features,
      neighbor_indices,
      mask,
      autoregressive_mask,
      sequence,
    )
    logits = final_projection(model_parameters, updated_node_features)
    temperature_key, next_prng_key = jax.random.split(current_prng_key)
    position_logits = (logits[position] / temperature) + jax.random.gumbel(
      temperature_key,
      logits[position].shape,
    )
    sequence = jax.nn.one_hot(position_logits[..., :20].argmax(-1), 21)
    logits = logits.at[position].set(position_logits)
    return next_prng_key, sequence, logits

  next_prng_key, sequence, logits = jax.lax.fori_loop(
    0,
    sequence.shape[0],
    update_sequence,
    (next_prng_key, sequence, logits),
  )

  return next_prng_key, edge_features, node_features, sequence, logits


def sample_straight_through_estimator_step(
  _i: int,
  carry: SamplingStepState,
  decoder: RunConditionalDecoderFn,
  model_parameters: ModelParameters,
  sample_model_pass_fn_only_prng: SampleModelPassOnlyPRNGFn,
  learning_rate: float,
  target_logits: Logits,
) -> SamplingStepState:
  """Single autoregressive sampling step with straight-through estimator.

  Args:
    _i: Current iteration.
    carry: Tuple containing current state (rng_key, edge_features, node_features, sequence, logits).
    decoder: Decoder function to update node features.
    model_parameters: Model parameters for the model.
    sample_model_pass_fn_only_prng: Function to run a single pass through the model.
    learning_rate: Learning rate for updating logits.
    target_logits: Target logits for the straight-through estimator.

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
  current_key, edge_features, node_features, _sequence, current_logits = carry

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
  def loss_fn(input_logits: Logits) -> tuple[CEELoss, tuple[NodeFeatures, Logits]]:
    """Compute the loss for the straight-through estimator."""
    ste_logits = straight_through_estimator(input_logits)
    updated_node_features = decoder(
      node_features,
      edge_features,
      neighbor_indices,
      mask,
      autoregressive_mask,
      ste_logits,
    )
    output_logits = final_projection(model_parameters, updated_node_features)

    return ste_loss(output_logits, target_logits, mask), (updated_node_features, output_logits)

  (_, (new_node_features, _)), grad = jax.value_and_grad(loss_fn, has_aux=True)(
    current_logits,
  )

  updated_logits = current_logits - learning_rate * grad

  updated_sequence = updated_logits.argmax(axis=-1).astype(jnp.int8)  # type: ignore[no-any-return]
  return (
    next_rng_key,
    edge_features,
    new_node_features,
    updated_sequence,
    updated_logits,
  )


def preload_sampling_step_decoder(
  decoder: RunConditionalDecoderFn,
  sample_model_pass_fn_only_prng: SampleModelPassOnlyPRNGFn,
  model_parameters: ModelParameters,
  sampling_config: SamplingConfig,
) -> SamplingStepFn:
  """Preload the sampling step decoder."""
  match sampling_config.sampling_strategy:
    case SamplingEnum.TEMPERATURE:
      """Get the temperature sampling step function."""
      decoding_loaded_step_fn = partial(
        sample_temperature_step,
        decoder=decoder,
        sample_model_pass_fn_only_prng=sample_model_pass_fn_only_prng,
        temperature=sampling_config.temperature,  # type: ignore[arg-type]
      )
    case SamplingEnum.STRAIGHT_THROUGH:
      """Get the straight-through sampling step function."""
      decoding_loaded_step_fn = partial(
        sample_straight_through_estimator_step,
        decoder=decoder,
        sample_model_pass_fn_only_prng=sample_model_pass_fn_only_prng,
        learning_rate=sampling_config.learning_rate,  # type: ignore[arg-type]
        target_logits=sampling_config.target_logits,  # type: ignore[arg-type]
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

  return partial(
    decoding_loaded_step_fn,
    model_parameters=model_parameters,
  )
