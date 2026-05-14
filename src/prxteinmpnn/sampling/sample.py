"""Factory for creating sequence sampling functions for PrxteinMPNN."""

from collections.abc import Callable
from functools import partial
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int, PRNGKeyArray

from prxteinmpnn.registry import SAMPLERS
from prxteinmpnn.types.bundles import (
    ConditioningBundle,
    GeometryBundle,
    InferenceBundle,
    LigandBundle as InferenceLigandBundle,
    WaveScheduleBundle,
)
from prxteinmpnn.types.configs import InferenceConfig
from prxteinmpnn.types.protocols import ModelProtocol, SamplerFn
from prxteinmpnn.types.stages import StageSet
from prxteinmpnn.inference.logits import LOGIT_STRATEGIES, BatchLogitFn
from prxteinmpnn.inference import sample_autoregressive, optimize_ste
from prxteinmpnn.registry import (
  combine_strategy_to_index,
)
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

      # Normalize inputs
      if structure_coordinates.ndim == 3:
          structure_coordinates = structure_coordinates[None, ...]
          mask = mask[None, ...]
          residue_index = residue_index[None, ...]
          chain_index = chain_index[None, ...]
          if tie_group_map is not None:
              tie_group_map = tie_group_map[None, ...]
          if y is not None:
              y = y[None, ...]
              y_t = y_t[None, ...]
              y_m = y_m[None, ...]

      geo = GeometryBundle(
          coords=structure_coordinates,
          mask=mask,
          residue_index=residue_index,
          chain_index=chain_index,
          state_flat_rows=jnp.zeros((S, L), dtype=jnp.int32),
          n_states=S,
          n_canonical=L,
          n_flat=L
      )
      
      if state_weights is None:
          state_weights = jnp.ones(S) / S
      
      cond = ConditioningBundle(
          fixed_mask=fixed_mask if fixed_mask is not None else jnp.zeros(L),
          fixed_tokens=fixed_tokens if fixed_tokens is not None else jnp.zeros(L, dtype=jnp.int32),
          bias=bias if bias is not None else jnp.zeros((L, 21)),
          tie_group_map=tie_group_map if tie_group_map is not None else jnp.broadcast_to(jnp.arange(L)[None, :], (S, L)),
          state_weights=state_weights,
          sequence_oh=jnp.zeros((L, 21)), # initial sequence not needed for STE
          ar_mask=jnp.zeros((S, L, L)) # will be updated in optimize_fn
      )
      
      lig = InferenceLigandBundle(
          y=y if y is not None else jnp.zeros((S, 0, 4, 3)),
          y_t=y_t if y_t is not None else jnp.zeros((S, 0, 4), dtype=jnp.int32),
          y_m=y_m if y_m is not None else jnp.zeros((S, 0, 4))
      )
      
      bundle = InferenceBundle(
          geometry=geo,
          conditioning=cond,
          ligand=lig,
          wave=WaveScheduleBundle.empty(L)
      )
      
      config = InferenceConfig(
          mode="score_conditional", # STE uses score_conditional in its loss
          temperature=temperature,
          logit_combine_strategy=combine_strategy_to_index(multi_state_strategy),
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
          logit_combine_strategy=config.logit_combine_strategy
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

      if structure_coordinates.ndim == 3:
          structure_coordinates = structure_coordinates[None, ...]
          mask = mask[None, ...]
          residue_index = residue_index[None, ...]
          chain_index = chain_index[None, ...]
          if tie_group_map is not None:
              tie_group_map = tie_group_map[None, ...]
          if y is not None:
              y = y[None, ...]
              y_t = y_t[None, ...]
              y_m = y_m[None, ...]

      geo = GeometryBundle(
          coords=structure_coordinates,
          mask=mask,
          residue_index=residue_index,
          chain_index=chain_index,
          state_flat_rows=jnp.zeros((S, L), dtype=jnp.int32),
          n_states=S,
          n_canonical=L,
          n_flat=L
      )
      
      if state_weights is None:
          state_weights = jnp.ones(S) / S
      
      if tie_group_map is None:
          tie_group_map = jnp.broadcast_to(jnp.arange(L)[None, :], (S, L))

      # For AR sampling, we need a decoding order to build the ar_mask stack
      # and wave schedule.
      k_order, prng_key = jax.random.split(prng_key)
      decoding_order, _ = decoding_order_fn(k_order, L, None, None) # simplified
      
      # For now, we use a sequential wave schedule for simplicity
      # but in a more advanced implementation, we'd use the decoding order.
      # WaveScheduleBundle.empty(L) gives a sequential 1-at-a-time schedule.
      
      from prxteinmpnn.utils.autoregression import generate_ar_mask
      ar_mask_single = generate_ar_mask(
          decoding_order if decoding_order is not None else jnp.arange(L),
          tie_group_map=tie_group_map[0] if tie_group_map is not None else None,
          num_groups=jnp.max(tie_group_map) + 1 if tie_group_map is not None else None
      )
      # Broadcast to S states
      ar_mask = jnp.broadcast_to(ar_mask_single[None, ...], (S, L, L))

      cond = ConditioningBundle(
          fixed_mask=fixed_mask if fixed_mask is not None else jnp.zeros(L),
          fixed_tokens=fixed_tokens if fixed_tokens is not None else jnp.zeros(L, dtype=jnp.int32),
          bias=bias if bias is not None else jnp.zeros((L, 21)),
          tie_group_map=tie_group_map if tie_group_map is not None else jnp.broadcast_to(jnp.arange(L)[None, :], (S, L)),
          state_weights=state_weights,
          sequence_oh=jnp.zeros((L, 21)), 
          ar_mask=ar_mask
      )
      
      lig = InferenceLigandBundle(
          y=y if y is not None else jnp.zeros((S, 0, 4, 3)),
          y_t=y_t if y_t is not None else jnp.zeros((S, 0, 4), dtype=jnp.int32),
          y_m=y_m if y_m is not None else jnp.zeros((S, 0, 4))
      )
      
      bundle = InferenceBundle(
          geometry=geo,
          conditioning=cond,
          ligand=lig,
          wave=WaveScheduleBundle.empty(L)
      )
      
      config = InferenceConfig(
          mode="sample_ar",
          temperature=temperature,
          logit_combine_strategy=combine_strategy_to_index(multi_state_strategy),
          use_rolling_state=use_rolling_state,
          inference=True
      )
      
      # Map combining strategy to the appropriate StageSet/LogitTransformFn
      strategy_cls = LOGIT_STRATEGIES.get(multi_state_strategy)
      if multi_state_strategy == "geometric_mean":
          logit_transform = strategy_cls(cond.state_weights, temperature=multi_state_temperature)
      else:
          logit_transform = strategy_cls(cond.state_weights)
          
      stage_set = StageSet(logit_transform=logit_transform)
      
      result_bundle = sample_autoregressive.kernel(model, prng_key, bundle, config, stage_set)
      sampled_seq_oh = result_bundle.conditioning.sequence_oh
      sampled_seq = jnp.argmax(sampled_seq_oh, axis=-1)
      
      # We return (sequence, logits, decoding_order)
      # For sampled_seq, logits are not immediately available from the kernel 
      # (it returns the final sequence). If we need logits, we'd have to re-score.
      # For now, return zero logits as a placeholder if not provided by kernel.
      return sampled_seq.astype(jnp.int8), jnp.zeros((L, 21)), decoding_order

    return cast("SamplerFn", sample_sequences)

  msg = f"Unknown sampling strategy: {sampling_strategy}"
  raise ValueError(msg)


SAMPLERS.register("make_sample_sequences")(make_sample_sequences)
