"""Conditional scoring kernel using InferenceBundle."""

from typing import Any, cast

import jax
from jaxtyping import PRNGKeyArray

from aminx.inference.decode.conditional import ConditionalDecode  # noqa: TID251
from aminx.inference.encode import make_encode_fn
from aminx.tiling.iterator import VmapIterator
from aminx.types.arrays import Logits
from aminx.types.bundles import EncoderOutput as BundleEncoderOutput
from aminx.types.bundles import InferenceBundle
from aminx.types.configs import InferenceConfig
from aminx.types.encodings import EncoderOutput
from aminx.types.protocols import ModelProtocol
from aminx.types.stages import StageSet  # noqa: TID251


def encode(
  model: ModelProtocol,
  prng_key: PRNGKeyArray,
  bundle: InferenceBundle,
  config: InferenceConfig,
) -> EncoderOutput:
  """Encode structure using vmap strategy (parallel over S conformational states).

  Args:
    model: Aminx model with encoder.
    prng_key: PRNG key.
    bundle: Inference bundle containing geometry and conditioning.
    config: Inference configuration.

  Returns:
    EncoderOutput with node_features, edge_features, neighbor_indices, mask
    stacked over S states.
  """
  encode_fn = make_encode_fn(model, use_rolling_state=False)
  return encode_fn(bundle, prng_key, config)


def score_from_encoding(
  model: ModelProtocol,
  prng_key: PRNGKeyArray,
  enc: Any,  # noqa: ANN401
  bundle: InferenceBundle,
  config: InferenceConfig,
  stage_set: StageSet,
) -> Logits:
  """Decode from a pre-computed encoding to produce logits.

  Args:
    model: Aminx model with decoder.
    prng_key: PRNG key.
    enc: Pre-computed encoder output (EncoderOutput from encode.py).
    bundle: Inference bundle containing conditioning.
    config: Inference configuration.
    stage_set: Stage set for logit fusion strategy.

  Returns:
    Logits (teacher-forced conditional logits).
  """
  # Use ConditionalDecode mode class for conditional decoding
  # (hardcoded Vmap iterator; STE uses this internally)
  decode_fn = ConditionalDecode(model=model, state_iterator=VmapIterator())
  return decode_fn(
    key=prng_key,
    enc=cast("BundleEncoderOutput", enc),
    bundle=bundle,
    config=config,
    stage_set=stage_set,
  )


def kernel(
  model: ModelProtocol,
  prng_key: PRNGKeyArray,
  bundle: InferenceBundle,
  config: InferenceConfig,
  stage_set: StageSet,
) -> Logits:
  """Compute teacher-forced conditional logits.

  Side-chain context (atom_37/atom_37_mask) is packaged onto the GeometryBundle by
  build_inference_bundle; it is not accepted as loose kernel kwargs (see #105).
  """
  k_enc, k_dec = jax.random.split(prng_key)

  # Encode and decode, byte-identical to original implementation
  enc = encode(model, k_enc, bundle, config)
  return score_from_encoding(model, k_dec, enc, bundle, config, stage_set)
