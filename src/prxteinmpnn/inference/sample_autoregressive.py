"""Autoregressive sampling kernel for PrxteinMPNN.

This kernel implements the core sampling loop, optimized for JIT and vmap.
It consumes a unified InferenceBundle and returns a structured SampleResult.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from jaxtyping import Array, Float, Int

if TYPE_CHECKING:
    from prxteinmpnn.model.mpnn import PrxteinMPNN
    from prxteinmpnn.types.bundles import InferenceBundle
    from prxteinmpnn.types.configs import InferenceConfig
    from prxteinmpnn.types.stages import StageSet
    from prxteinmpnn.types.arrays import PRNGKeyArray


@dataclass(frozen=True)
class SampleResult:
    """Result of an autoregressive sampling run."""
    sequence: Int[Array, "L"]
    logits: Float[Array, "L 21"]


def kernel(
    model: PrxteinMPNN,
    prng_key: PRNGKeyArray,
    bundle: InferenceBundle,
    config: InferenceConfig,
    stage_set: StageSet,
) -> SampleResult:
    """Autoregressive sampling kernel.

    Optimized to encode features once and then iterate through the decoding waves.
    Delegates to unified driver for decoding logic.
    """
    import jax
    from prxteinmpnn.inference.encode import make_encode_fn
    from prxteinmpnn.inference.driver import decode

    k_enc, k_dec = jax.random.split(prng_key)

    # Encode using make_encode_fn with rolling state disabled for AR
    encode_fn = make_encode_fn(model, use_rolling_state=False)
    enc = encode_fn(bundle, k_enc, config)

    # Delegate to unified driver for autoregressive sampling
    return decode(model, k_dec, enc, bundle.conditioning, bundle.wave, config, stage_set)
