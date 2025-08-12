"""Sample sequences from a structure using the ProteinMPNN model.

prxteinmpnn.sampling.sampling
"""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import optax
from jaxtyping import PRNGKeyArray

from prxteinmpnn.model.decoder import RunConditionalDecoderFn
from prxteinmpnn.model.projection import final_projection
from prxteinmpnn.utils.types import (
  AtomMask,
  AutoRegressiveMask,
  CEELoss,
  Logits,
  ModelParameters,
  NeighborIndices,
  NodeFeatures,
  ProteinSequence,
  SamplingHyperparameters,
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
  optax.OptState | None,
]
SamplingStepInput = tuple[int, SamplingStepState]

SamplingStepFn = Callable[
  [*SamplingStepInput],
  SamplingStepState,
]

SampleModelPassOnlyPRNGFn = Callable[[PRNGKeyArray], SamplingModelPassOutput]


def sample_temperature_step(
  i: int,
  carry: SamplingStepState,
  decoder: RunConditionalDecoderFn,
  neighbor_indices: NeighborIndices,
  mask: AtomMask,
  autoregressive_mask: AutoRegressiveMask,
  model_parameters: ModelParameters,
  hyperparameters: SamplingHyperparameters = (1.0,),
) -> SamplingStepState:
  """Single autoregressive sampling step with temperature scaling.

  Args:
    i: Current iteration.
    carry: Tuple containing current state (rng_key, edge_features, node_features, sequence, logits).
    decoder: Decoder function to update node features.
    neighbor_indices: Indices of neighboring nodes.
    mask: Atom mask for valid atoms.
    autoregressive_mask: Mask for autoregressive decoding.
    model_parameters: Model parameters for the model.
    hyperparameters: Hyperparameters for sampling. In this case, the temperature for scaling
      logits.

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
  temperature = hyperparameters[0]
  prng_key, edge_features, node_features, sequence, logits, _ = carry

  current_prng_key, next_prng_key = jax.random.split(prng_key)

  updated_node_features = decoder(
    node_features,
    edge_features,
    neighbor_indices,
    mask,
    autoregressive_mask,
    sequence,
  )

  node_features = updated_node_features

  logits = final_projection(model_parameters, updated_node_features)

  logits = logits / temperature + jax.random.gumbel(current_prng_key, logits.shape)

  sampled_aa = logits[:20].argmax()

  s_i = jax.nn.one_hot(sampled_aa, 21)

  sequence = sequence.at[i].set(s_i).astype(jnp.int8)
  logits = logits.at[i].set(logits)

  return next_prng_key, edge_features, node_features, sequence, logits, None


def sample_straight_through_estimator_step(
  _i: int,
  carry: SamplingStepState,
  decoder: RunConditionalDecoderFn,
  model_parameters: ModelParameters,
  sample_model_pass_fn_only_prng: SampleModelPassOnlyPRNGFn,
  optimizer: optax.GradientTransformation,
  target_logits: Logits,
) -> SamplingStepState:
  """Single autoregressive sampling step with straight-through estimator.

  Args:
    _i: Current iteration.
    carry: Tuple containing current state (rng_key, edge_features, node_features, sequence, logits).
    decoder: Decoder function to update node features.
    model_parameters: Model parameters for the model.
    sample_model_pass_fn_only_prng: Function to run a single pass through the model.
    optimizer: Optax optimizer for updating logits.
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
  current_key, edge_features, node_features, _sequence, current_logits, opt_state = carry

  jax.debug.print(
    "➡️ Iteration {_i}, Current sequence: {_sequence}",
    _i=_i,
    _sequence=_sequence[:10],
  )
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

  updates, new_opt_state = optimizer.update(grad, opt_state)  # type: ignore[arg-type]
  updated_logits = optax.apply_updates(current_logits, updates)

  updated_sequence = updated_logits.argmax(axis=-1).astype(jnp.int8)  # type: ignore[no-any-return]
  return (
    next_rng_key,
    edge_features,
    new_node_features,
    updated_sequence,
    updated_logits,  # type: ignore[no-any-return]
    new_opt_state,
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
      msg = "Temperature sampling is not implemented yet."
      raise NotImplementedError(msg)
    case SamplingEnum.STRAIGHT_THROUGH:
      """Get the straight-through sampling step function."""
      decoding_loaded_step_fn = partial(
        sample_straight_through_estimator_step,
        decoder=decoder,
        sample_model_pass_fn_only_prng=sample_model_pass_fn_only_prng,
        optimizer=sampling_config.optimizer,  # type: ignore[arg-type]
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
