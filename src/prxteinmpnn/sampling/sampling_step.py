"""Defines the single-pass autoregressive sampling step."""

from functools import partial

import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from prxteinmpnn.model.decoder import RunAutoregressiveDecoderFn
from prxteinmpnn.utils.types import Logits, ProteinSequence

from .config import SamplingConfig, SamplingEnum
from .initialize import SamplingModelPassOutput

SampleModelPassFn = partial[SamplingModelPassOutput]
SamplingStepState = tuple[PRNGKeyArray, ProteinSequence, Logits]
SamplingStepFn = partial[SamplingStepState]


def temperature_sample(
  decoder: RunAutoregressiveDecoderFn,
  sample_model_pass_fn: SampleModelPassFn,
  temperature: float,
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
  )
  output_sequence = output_sequence_one_hot.argmax(axis=-1).astype(jnp.int8)
  return prng_key, output_sequence, logits


def preload_sampling_step_decoder(
  decoder: RunAutoregressiveDecoderFn,
  sample_model_pass_fn: SampleModelPassFn,
  sampling_config: SamplingConfig,
) -> SamplingStepFn:
  """Preloads the sampling step decoder for the specified strategy."""
  if sampling_config.sampling_strategy == SamplingEnum.TEMPERATURE:
    return partial(
      temperature_sample,
      decoder=decoder,
      sample_model_pass_fn=sample_model_pass_fn,
      temperature=sampling_config.temperature,
    )
  # No other sampling strategies are supported in this simplified version
  msg = f"Unsupported sampling strategy: {sampling_config.sampling_strategy}"
  raise NotImplementedError(msg)
