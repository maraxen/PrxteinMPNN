"""Conditional scoring kernel using InferenceBundle."""

from jaxtyping import PRNGKeyArray

from prxteinmpnn.inference.encode import make_encode_fn
from prxteinmpnn.inference.driver import decode
from prxteinmpnn.types.bundles import InferenceBundle
from prxteinmpnn.types.configs import InferenceConfig
from prxteinmpnn.types.protocols import ModelProtocol
from prxteinmpnn.types.arrays import Logits
from prxteinmpnn.types.stages import StageSet

# TODO: by default, 1 - jnp.eye should be the autoregressive mask

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

    # Encode using make_encode_fn, which controls scan vs vmap over S states
    encode_fn = make_encode_fn(model, use_rolling_state=config.use_rolling_state)
    enc = encode_fn(bundle, k_enc, config)

    # Delegate to unified driver for conditional decoding
    return decode(model, k_dec, enc, bundle.conditioning, bundle.wave, config, stage_set)
