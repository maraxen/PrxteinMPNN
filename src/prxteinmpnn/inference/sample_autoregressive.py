"""Autoregressive sampling kernel for PrxteinMPNN.

This kernel implements the core sampling loop, optimized for JIT and vmap.
It consumes a unified InferenceBundle and returns a structured SampleResult.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
from jaxtyping import Array, Float, Int

if TYPE_CHECKING:
  from prxteinmpnn.model.mpnn import PrxteinMPNN
  from prxteinmpnn.types.arrays import PRNGKeyArray
  from prxteinmpnn.types.bundles import InferenceBundle, PackerResult
  from prxteinmpnn.types.configs import InferenceConfig
  from prxteinmpnn.types.stages import StageSet


class SampleResult(eqx.Module):
  """Result of an autoregressive sampling run."""

  sequence: Int[Array, L]
  logits: Float[Array, "L 21"]
  packer_result: PackerResult | None = None


def kernel(
  model: PrxteinMPNN,
  prng_key: PRNGKeyArray,
  bundle: InferenceBundle,
  config: InferenceConfig,
  stage_set: StageSet,
) -> SampleResult:
  """Autoregressive sampling kernel.

  Optimized to encode features once and then iterate through the decoding waves.
  Delegates to unified driver for autoregressive decoding.
  """
  import jax

  from prxteinmpnn.inference.decode.factory import make_decode_fn
  from prxteinmpnn.inference.decode.mode import AutoregressiveMode
  from prxteinmpnn.inference.encode import make_encode_fn
  from prxteinmpnn.tiling.strategy import Vmap

  k_enc, k_dec = jax.random.split(prng_key)

  # Encode using make_encode_fn with rolling state disabled for AR
  encode_fn = make_encode_fn(model, use_rolling_state=False)
  enc = encode_fn(bundle, k_enc, config)

  # Resolve autoregressive decode function and execute
  decode_fn = make_decode_fn(
    model=model,
    mode=AutoregressiveMode(),
    strategy=Vmap(),
  )
  return decode_fn(k_dec, enc, bundle, config, stage_set)
