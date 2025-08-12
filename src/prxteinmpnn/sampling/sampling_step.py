"""Sample sequences from a structure using the ProteinMPNN model.

prxteinmpnn.sampling.sampling
"""

import enum
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
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


class SamplingEnum(enum.Enum):
  """Enum for different sampling strategies."""

  GREEDY = "greedy"
  TOP_K = "top_k"
  TOP_P = "top_p"
  TEMPERATURE = "temperature"
  BEAM_SEARCH = "beam_search"
  STRAIGHT_THROUGH = "straight_through"


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
  prng_key, edge_features, node_features, sequence, logits = carry

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

  return next_prng_key, edge_features, node_features, sequence, logits


def sample_straight_through_estimator_step(
  _i: int,
  carry: SamplingStepState,
  decoder: RunConditionalDecoderFn,
  model_parameters: ModelParameters,
  sample_model_pass_fn_only_prng: SampleModelPassOnlyPRNGFn,
  hyperparameters: tuple[float, Logits],
) -> SamplingStepState:
  """Single autoregressive sampling step with straight-through estimator.

  Args:
    _i: Current iteration.
    carry: Tuple containing current state (rng_key, edge_features, node_features, sequence, logits).
    decoder: Decoder function to update node features.
    neighbor_indices: Indices of neighboring nodes.
    mask: Atom mask for valid atoms.
    autoregressive_mask: Mask for autoregressive decoding.
    model_parameters: Model parameters for the model.
    sample_model_pass_fn_only_prng: Function that takes a PRNG key and returns the model pass
      output.
    hyperparameters: Hyperparameters for the straight-through estimator. In this case, the
      learning rate in the first position and the target logits in the second position.

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
  learning_rate, _target_logits = hyperparameters[0], hyperparameters[1]
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

    return ste_loss(output_logits, input_logits, mask), (updated_node_features, output_logits)

  (_, (new_node_features, _)), grad = jax.value_and_grad(loss_fn, has_aux=True)(
    current_logits,
  )

  grad_norm = jnp.linalg.norm(grad)
  has_nan = jnp.isnan(grad_norm)
  jax.debug.print(
    "➡️ Iteration {_i}, Grad Norm: {n}, Is NaN: {nan}",
    _i=_i,
    n=grad_norm,
    nan=has_nan,
  )
  clip_threshold = 1.0
  grad = jnp.where(grad_norm > clip_threshold, grad * (clip_threshold / grad_norm), grad)

  updated_logits = current_logits - learning_rate * grad

  updated_sequence = updated_logits.argmax(axis=-1).astype(jnp.int8)
  return next_rng_key, edge_features, new_node_features, updated_sequence, updated_logits


def preload_sampling_step_decoder(
  decoder: RunConditionalDecoderFn,
  sample_model_pass_fn_only_prng: SampleModelPassOnlyPRNGFn,
  sampling_strategy: SamplingEnum,
) -> Callable[
  [
    ModelParameters,
    SamplingHyperparameters,
  ],
  SamplingStepFn,
]:
  """Preload the sampling step decoder."""
  match sampling_strategy:
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
      msg = f"Unknown sampling strategy: {sampling_strategy}"
      raise ValueError(msg)

  def sampling_step_fn(
    model_parameters: ModelParameters,
    hyperparameters: SamplingHyperparameters,
  ) -> SamplingStepFn:
    """Get the sampling step function based on the sampling strategy."""
    return partial(
      decoding_loaded_step_fn,
      model_parameters=model_parameters,
      hyperparameters=hyperparameters,
    )

  return sampling_step_fn
