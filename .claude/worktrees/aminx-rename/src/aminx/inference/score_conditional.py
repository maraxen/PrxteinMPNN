"""Conditional scoring kernel using InferenceBundle."""

from jaxtyping import PRNGKeyArray

from aminx.inference.decode.conditional import ConditionalDecode
from aminx.inference.encode import make_encode_fn
from aminx.tiling.iterator import VmapIterator
from aminx.types.arrays import Logits
from aminx.types.bundles import InferenceBundle
from aminx.types.configs import InferenceConfig
from aminx.types.protocols import ModelProtocol
from aminx.types.stages import StageSet


def kernel(
  model: ModelProtocol,
  prng_key: PRNGKeyArray,
  bundle: InferenceBundle,
  config: InferenceConfig,
  stage_set: StageSet,
) -> Logits:
  """Compute teacher-forced conditional logits."""
  import jax

  k_enc, k_dec = jax.random.split(prng_key)

  # Encode using vmap strategy (parallel over S states)
  encode_fn = make_encode_fn(model, use_rolling_state=False)
  enc = encode_fn(bundle, k_enc, config)

  # Use ConditionalDecode mode class for conditional decoding
  # (hardcoded Vmap iterator; STE uses this internally)
  decode_fn = ConditionalDecode(model=model, state_iterator=VmapIterator())
  return decode_fn(
    key=k_dec,
    enc=enc,
    bundle=bundle,
    config=config,
    stage_set=stage_set,
  )
