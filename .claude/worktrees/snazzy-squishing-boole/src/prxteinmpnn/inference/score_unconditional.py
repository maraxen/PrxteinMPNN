"""Unconditional scoring kernel using InferenceBundle."""

from jaxtyping import PRNGKeyArray

from prxteinmpnn.inference.decode.factory import make_decode_fn
from prxteinmpnn.inference.decode.mode import UnconditionalMode
from prxteinmpnn.inference.encode import make_encode_fn
from prxteinmpnn.tiling.strategy import Vmap
from prxteinmpnn.types.arrays import Logits
from prxteinmpnn.types.bundles import InferenceBundle
from prxteinmpnn.types.configs import InferenceConfig
from prxteinmpnn.types.protocols import ModelProtocol
from prxteinmpnn.types.stages import StageSet


def kernel(
  model: ModelProtocol,
  prng_key: PRNGKeyArray,
  bundle: InferenceBundle,
  config: InferenceConfig,
  stage_set: StageSet,
) -> Logits:
  """Compute unconditional logits."""
  import jax

  k_enc, k_dec = jax.random.split(prng_key)

  # Encode using vmap strategy (parallel over S states)
  encode_fn = make_encode_fn(model, use_rolling_state=False)
  enc = encode_fn(bundle, k_enc, config)

  # Construct UnconditionalDecode with vmap strategy over state axis
  decode_fn = make_decode_fn(model, mode=UnconditionalMode(), strategy=Vmap())

  # Call the decode function to get logits
  return decode_fn(k_dec, enc, bundle, config, stage_set)
