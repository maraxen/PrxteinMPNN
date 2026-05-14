"""Factory for creating sequence sampling functions for PrxteinMPNN."""

from collections.abc import Callable
from functools import partial
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int, PRNGKeyArray

from prxteinmpnn.registry import SAMPLERS
from prxteinmpnn.types.protocols import ModelProtocol, SamplerFn
from prxteinmpnn.types.configs import InferenceConfig
from prxteinmpnn.types.stages import StageSet
from prxteinmpnn.inference import sample_autoregressive, optimize_ste
from prxteinmpnn.host.bundle_builder import build_inference_bundle
from prxteinmpnn.utils.decoding_order import DecodingOrderFn, random_decoding_order
from prxteinmpnn.utils.types import (
  AlphaCarbonMask,
  BackboneNoise,
  ChainIndex,
  DecodingOrder,
  InputBias,
  Logits,
  ProteinSequence,
  ResidueIndex,
  StructureAtomicCoordinates,
)


_AMINO_ACID_VOCAB = 21


_DEFAULT_DECODING_ORDER_FN = cast("DecodingOrderFn", random_decoding_order)


def make_sample_sequences(
  model: ModelProtocol,
  decoding_order_fn: DecodingOrderFn = _DEFAULT_DECODING_ORDER_FN,
  sampling_strategy: Literal["temperature", "straight_through"] = "temperature",
  _num_encoder_layers: int = 3,
  _num_decoder_layers: int = 3,
  use_concrete: bool = False,
  tau_start: float = 1.0,
  tau_end: float = 0.1,
) -> SamplerFn:
  """Create a function to sample sequences from a structure using PrxteinMPNN.

  Args:
    model: A PrxteinMPNN Equinox model instance.
    decoding_order_fn: Function to generate decoding order.
    sampling_strategy: "temperature" (autoregressive) or "straight_through" (iterative).
    use_concrete: Use Gumbel-Softmax for straight_through.
    tau_start: Start temperature for Gumbel-Softmax.
    tau_end: End temperature for Gumbel-Softmax.

  Returns:
    A function that samples sequences from structures.
  """
  del _num_encoder_layers, _num_decoder_layers

  if sampling_strategy == "straight_through":
    optimize_fn = optimize_ste.make_optimize_sequence_fn(
      model,
      decoding_order_fn,
      use_concrete=use_concrete,
      tau_start=tau_start,
      tau_end=tau_end,
    )

    @partial(jax.jit, static_argnames=("multi_state_strategy", "use_rolling_state"))
    def sample_sequences(
      prng_key: PRNGKeyArray,
      structure_coordinates: jax.Array,
      mask: jax.Array,
      residue_index: jax.Array,
      chain_index: jax.Array,
      bias: jax.Array | None = None,
      fixed_mask: jax.Array | None = None,
      fixed_tokens: jax.Array | None = None,
      backbone_noise: float | None = None,
      iterations: int = 100,
      learning_rate: float = 0.01,
      temperature: float = 1.0,
      tie_group_map: jax.Array | None = None,
      multi_state_strategy: Literal["arithmetic_mean", "geometric_mean", "product"] = "arithmetic_mean",
      multi_state_temperature: float = 1.0,
      state_weights: jax.Array | None = None,
      use_rolling_state: bool = False,
      y: jax.Array | None = None,
      y_t: jax.Array | None = None,
      y_m: jax.Array | None = None,
      **kwargs: Any,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
      
      L = structure_coordinates.shape[1] if structure_coordinates.ndim == 4 else structure_coordinates.shape[0]
      S = structure_coordinates.shape[0] if structure_coordinates.ndim == 4 else 1

      bundle, config, stage_set = build_inference_bundle(
          coords=structure_coordinates,
          mask=mask,
          residue_index=residue_index,
          chain_index=chain_index,
          backbone_noise=backbone_noise if backbone_noise is not None else 0.0,
          fixed_mask=fixed_mask,
          fixed_tokens=fixed_tokens,
          bias=bias,
          tie_group_map=tie_group_map,
          state_weights=state_weights,
          y=y, y_t=y_t, y_m=y_m,
          temperature=temperature,
          mode="score_conditional",
          strategy=multi_state_strategy,
          strategy_temperature=multi_state_temperature,
          use_rolling_state=use_rolling_state,
          inference=True
      )
      
      final_seq, final_logits, _ = optimize_fn(
          prng_key, 
          bundle, 
          config, 
          iterations, 
          learning_rate, 
          temperature,
          use_rolling_state=use_rolling_state,
          logit_combine_strategy=multi_state_strategy
      )
      return final_seq, final_logits, jnp.arange(L)

    return cast("SamplerFn", sample_sequences)

  if sampling_strategy == "temperature":

    @partial(jax.jit, static_argnames=("multi_state_strategy", "use_rolling_state"))
    def sample_sequences(
      prng_key: PRNGKeyArray,
      structure_coordinates: jax.Array,
      mask: jax.Array,
      residue_index: jax.Array,
      chain_index: jax.Array,
      bias: jax.Array | None = None,
      fixed_mask: jax.Array | None = None,
      fixed_tokens: jax.Array | None = None,
      backbone_noise: float | None = None,
      temperature: float = 1.0,
      tie_group_map: jax.Array | None = None,
      multi_state_strategy: Literal["arithmetic_mean", "geometric_mean", "product"] = "arithmetic_mean",
      multi_state_temperature: float = 1.0,
      state_weights: jax.Array | None = None,
      use_rolling_state: bool = False,
      y: jax.Array | None = None,
      y_t: jax.Array | None = None,
      y_m: jax.Array | None = None,
      **kwargs: Any,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
      
      L = structure_coordinates.shape[1] if structure_coordinates.ndim == 4 else structure_coordinates.shape[0]
      S = structure_coordinates.shape[0] if structure_coordinates.ndim == 4 else 1

      k_order, prng_key = jax.random.split(prng_key)
      decoding_order, _ = decoding_order_fn(k_order, L, None, None)

      from prxteinmpnn.utils.autoregression import generate_ar_mask
      ar_mask_single = generate_ar_mask(
          decoding_order if decoding_order is not None else jnp.arange(L),
          tie_group_map=tie_group_map[0] if tie_group_map is not None else None,
          num_groups=jnp.max(tie_group_map) + 1 if tie_group_map is not None else None
      )
      
      bundle, config, stage_set = build_inference_bundle(
          coords=structure_coordinates,
          mask=mask,
          residue_index=residue_index,
          chain_index=chain_index,
          backbone_noise=backbone_noise if backbone_noise is not None else 0.0,
          fixed_mask=fixed_mask,
          fixed_tokens=fixed_tokens,
          bias=bias,
          tie_group_map=tie_group_map,
          state_weights=state_weights,
          ar_mask=ar_mask_single,
          y=y, y_t=y_t, y_m=y_m,
          temperature=temperature,
          mode="sample_ar",
          strategy=multi_state_strategy,
          strategy_temperature=multi_state_temperature,
          use_rolling_state=use_rolling_state,
          inference=True
      )
      
      result = sample_autoregressive.kernel(model, prng_key, bundle, config, stage_set)
      
      return result.sequence.astype(jnp.int8), result.logits, decoding_order

    return cast("SamplerFn", sample_sequences)

  msg = f"Unknown sampling strategy: {sampling_strategy}"
  raise ValueError(msg)


SAMPLERS.register("make_sample_sequences")(make_sample_sequences)
