import jax


def group_sampling_step(
  carry,
  group_id_to_decode,
):
  """Sampling step for tied group logit averaging.

  Args:
    carry: tuple (key, S, model, tie_group_map, autoregressive_mask, temperature)
    group_id_to_decode: int, group id for this step

  Returns:
    new_carry, None

  """
  key, S, model, tie_group_map, autoregressive_mask, temperature = carry
  logits = model(S, autoregressive_mask)
  group_mask = tie_group_map == group_id_to_decode
  # Masked logits for group
  masked_logits = jnp.where(group_mask[:, None], logits, -1e9)
  max_logits = jnp.max(masked_logits, axis=0, keepdims=True)
  shifted_logits = masked_logits - max_logits
  exp_logits = jnp.exp(shifted_logits)
  sum_exp_logits = jnp.sum(exp_logits, axis=0, keepdims=True)
  num_in_group = jnp.sum(group_mask)
  avg_exp_logits = sum_exp_logits / num_in_group
  avg_logits = jnp.log(avg_exp_logits) + max_logits
  sampled_logits = avg_logits / temperature
  new_key, subkey = jax.random.split(key)
  token = jax.random.categorical(subkey, sampled_logits, axis=-1)[0]
  S_new = jnp.where(group_mask, token, S)
  return (new_key, S_new, model, tie_group_map, autoregressive_mask, temperature), None


"""Defines the single-pass autoregressive sampling step."""

from functools import partial
from typing import Literal

import jax.numpy as jnp
from jaxtyping import Float, PRNGKeyArray

from prxteinmpnn.model.decoding_signatures import RunAutoregressiveDecoderFn
from prxteinmpnn.utils.types import Logits, ProteinSequence

from .initialize import SamplingModelPassOutput

SampleModelPassFn = partial[SamplingModelPassOutput]
SamplingStepState = tuple[PRNGKeyArray, ProteinSequence, Logits]
SamplingStepFn = partial[SamplingStepState]


def temperature_sample(
  decoder: RunAutoregressiveDecoderFn,
  sample_model_pass_fn: SampleModelPassFn,
  temperature: Float | None,
  bias: Logits | None,
  prng_key: PRNGKeyArray,
) -> SamplingStepState:
  """Single autoregressive sampling step with temperature scaling."""
  (
    node_features,
    edge_features,
    neighbor_indices,
    mask,
    autoregressive_mask,
    decoding_key,
  ) = sample_model_pass_fn(prng_key=prng_key)

  output_sequence_one_hot, logits = decoder(
    decoding_key,
    node_features,
    edge_features,
    neighbor_indices,
    mask,
    autoregressive_mask,
    temperature,
    bias,
  )
  output_sequence = output_sequence_one_hot.argmax(axis=-1).astype(jnp.int8)
  return prng_key, output_sequence, logits


def preload_sampling_step_decoder(
  decoder: RunAutoregressiveDecoderFn,
  sample_model_pass_fn: SampleModelPassFn,
  sampling_strategy: Literal["temperature", "straight_through"],
  temperature: Float | None = None,
) -> SamplingStepFn:
  """Preloads the sampling step decoder for the specified strategy."""
  if sampling_strategy == "temperature":
    return partial(
      temperature_sample,
      decoder=decoder,
      sample_model_pass_fn=sample_model_pass_fn,
      temperature=temperature,
    )
  # No other sampling strategies are supported in this simplified version
  msg = f"Unsupported sampling strategy: {sampling_strategy}"
  raise NotImplementedError(msg)
