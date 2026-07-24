"""Autoregressive sampling kernel for Aminx.

This kernel implements the core sampling loop, optimized for JIT and vmap.
It consumes a unified InferenceBundle and returns a structured SampleResult.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
from jaxtyping import Array, Float, Int

if TYPE_CHECKING:
  from aminx.model.mpnn import Aminx
  from aminx.tiling.strategy import AxisStrategy
  from aminx.types.arrays import PRNGKeyArray
  from aminx.types.bundles import InferenceBundle, PackerResult
  from aminx.types.configs import InferenceConfig
  from aminx.types.stages import StageSet


class SampleResult(eqx.Module):
  """Result of an autoregressive sampling run."""

  sequence: Int[Array, L]
  logits: Float[Array, "L 21"]
  packer_result: PackerResult | None = None


def kernel(
  model: Aminx,
  prng_key: PRNGKeyArray,
  bundle: InferenceBundle,
  config: InferenceConfig,
  stage_set: StageSet,
  *,
  inference_only: bool = False,
  state_strategy: AxisStrategy | None = None,
) -> SampleResult:
  """Autoregressive sampling kernel.

  Optimized to encode features once and then iterate through the decoding waves.
  Delegates to unified driver for autoregressive decoding. Side-chain context
  (atom_37/atom_37_mask) is packaged onto the GeometryBundle by
  build_inference_bundle, not accepted as loose kernel kwargs (see #105).

  Args:
    inference_only: When True, the wave axis uses lax.while_loop instead of
      lax.scan (single XLA WhileOp -- much faster to compile, especially for
      long sequences / many wave counts). Not reverse-mode differentiable --
      never set True on a path that needs gradients through the AR loop
      (training never calls this kernel, so this is safe for any sampling use).
    state_strategy: State-axis (bundle.geometry.n_states) Vmap/SafeMap strategy.
      None (default) preserves prior behavior (Vmap -- correct for the
      num_states=1 single-structure campaign path in sampling/sample.py).
      Multi-state callers (sampling/multistate_poe.py) MUST pass a strategy
      resolved via BatchPlanner against the aminx.tiling.axes.N_STATES
      template (default_batch_size=1): an unconditional Vmap here batches
      ALL states through every decoder layer's MLPs simultaneously, and at
      production sample counts this produces a single fused GEMM XLA's
      autotuner cannot find a valid kernel config for (observed: sample_
      count=128 crashed after an 809s compile with "Autotuning failed for
      HLO: f32[128,12582912]{1,0} fusion(...)"; sample_count=512 failed
      differently, "9 out of 89 instructions" -- both against a
      hand-tuned linear activation_bytes_per_element estimate in
      multistate_poe.py that only ever fed a single-axis BatchPlanner.plan
      call for n_samples, with the state axis never resolved through
      BatchPlanner at all despite N_STATES already declaring the correct
      default_batch_size=1 SafeMap-per-state convention).

  """
  import jax

  from aminx.inference.decode.factory import make_decode_fn
  from aminx.inference.decode.mode import AutoregressiveConfig, AutoregressiveMode
  from aminx.inference.encode import make_encode_fn
  from aminx.tiling.strategy import Vmap

  k_enc, k_dec = jax.random.split(prng_key)

  # Encode using make_encode_fn with rolling state disabled for AR
  encode_fn = make_encode_fn(model, use_rolling_state=False)
  enc = encode_fn(bundle, k_enc, config)

  # Resolve autoregressive decode function and execute
  decode_fn = make_decode_fn(
    model=model,
    mode=AutoregressiveMode(),
    strategy=state_strategy if state_strategy is not None else Vmap(),
    autoregressive_config=AutoregressiveConfig(inference_only=inference_only),
  )
  return decode_fn(k_dec, enc, bundle, config, stage_set)
