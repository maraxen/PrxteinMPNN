"""Factory for creating sequence sampling functions for PrxteinMPNN."""

from collections.abc import Callable
from functools import partial
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int, PRNGKeyArray

from prxteinmpnn.model import PrxteinLigandMPNN, PrxteinMPNN
from prxteinmpnn.payloads import LigandStack, MultistateStackPayload
from prxteinmpnn.protocols import SamplerFn
from prxteinmpnn.sampling.ste_optimize import make_optimize_sequence_fn
from prxteinmpnn.utils.autoregression import generate_ar_mask
from prxteinmpnn.utils.decoding_order import DecodingOrderFn, random_decoding_order

_DEFAULT_DECODING_ORDER_FN = cast("DecodingOrderFn", random_decoding_order)
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


def make_sample_sequences(
  model: PrxteinMPNN | PrxteinLigandMPNN,
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
    decoding_order_fn: Function to generate decoding order (default: random).
        Should accept (key, num_residues, tie_group_map, num_groups).
    sampling_strategy: Sampling strategy - "temperature" or "straight_through".
    _num_encoder_layers: Deprecated, ignored (kept for API compatibility).
    _num_decoder_layers: Deprecated, ignored (kept for API compatibility).

  Returns:
    A function that samples sequences from structures.

  Example:
    >>> from prxteinmpnn.io.weights import load_model
    >>> model = load_model()
    >>> sample_fn = make_sample_sequences(model, sampling_strategy="temperature")
    >>> seq, logits, order = sample_fn(key, coords, mask, res_idx, chain_idx)
    >>>
    >>> # With tied positions
    >>> tie_map = jnp.array([0, 0, 1, 1, 2])  # Positions 0-1 tied, 2-3 tied
    >>> seq, logits, order = sample_fn(
    ...     key, coords, mask, res_idx, chain_idx,
    ...     tie_group_map=tie_map, num_groups=3
    ... )
    >>>
    >>> # For optimization
    >>> optimize_fn = make_sample_sequences(model, sampling_strategy="straight_through")
    >>> seq, logits, order = optimize_fn(
    ...     key, coords, mask, res_idx, chain_idx,
    ...     iterations=100, learning_rate=0.01
    ... )

  """
  caps = model.capabilities
  supports_multi_state_temperature = caps.accepts_multi_state_temperature
  supports_state_weights = caps.accepts_state_weights
  supports_fixed_controls = caps.accepts_fixed_mask_and_tokens
  is_ligand_mpnn = caps.is_ligand_model

  if sampling_strategy == "straight_through":
    optimize_fn = make_optimize_sequence_fn(
      cast("PrxteinMPNN", model),
      decoding_order_fn,
      use_concrete=use_concrete,
      tau_start=tau_start,
      tau_end=tau_end,
    )

    @partial(
      jax.jit,
      static_argnames=(
        "num_groups",
        "multi_state_strategy",
        "multistate_mode",
        "max_group_size",
      ),
    )
    def sample_sequences(
      prng_key: PRNGKeyArray,
      structure_coordinates: StructureAtomicCoordinates,
      mask: AlphaCarbonMask,
      residue_index: ResidueIndex,
      chain_index: ChainIndex,
      bias: InputBias | None = None,
      fixed_positions: jnp.ndarray | None = None,
      fixed_mask: jnp.ndarray | None = None,
      fixed_tokens: jnp.ndarray | None = None,
      backbone_noise: BackboneNoise | None = None,
      iterations: Int | None = None,
      learning_rate: Float | None = None,
      temperature: Float | None = None,
      tie_group_map: jnp.ndarray | None = None,
      num_groups: int | None = None,
      multi_state_strategy: Literal[
        "arithmetic_mean",
        "geometric_mean",
        "product",
      ] = "arithmetic_mean",
      multistate_mode: Literal["flat", "state_vmap", "state_vmap_exact"] = "flat",
      structure_mapping: jax.Array | None = None,
      multi_state_temperature: Float = 1.0,
      max_group_size: int = 16,
      **kwargs: Any,
    ) -> tuple[ProteinSequence, Logits, DecodingOrder]:
      """Optimize a sequence using straight-through estimation.

      Args:
        prng_key: JAX random key.
        structure_coordinates: Atomic coordinates (N, 4, 3).
        mask: Alpha carbon mask indicating valid residues.
        residue_index: Residue indices.
        chain_index: Chain indices.
        bias: Not used in straight_through mode.
        fixed_positions: Not implemented yet.
        backbone_noise: Optional noise for backbone coordinates.
        iterations: Number of optimization steps (default: 100).
        learning_rate: Learning rate for optimization (default: 0.01).
        temperature: Temperature for STE sampling (default: 1.0).
        tie_group_map: Optional (N,) array mapping positions to group IDs for tied sampling.
        num_groups: Number of unique groups when using tied positions.
        multi_state_strategy: Unused in straight_through mode (kept for API compatibility).
        structure_mapping: Optional (N,) array mapping each residue to a structure ID.
                  When provided (multi-state mode), prevents cross-structure
                  neighbors to avoid information leakage between conformational states.
        multi_state_temperature: Unused in straight_through mode
          (kept for API compatibility).
        **kwargs: Additional arguments for LigandMPNN (Y, Y_t, Y_m) or weighting.

      Returns:
        Tuple of (optimized sequence, final logits, decoding order).

      """
      del bias, fixed_positions

      if iterations is None:
        iterations = jnp.array(100, dtype=jnp.int32)
      if learning_rate is None:
        learning_rate = jnp.array(0.01, dtype=jnp.float32)
      if temperature is None:
        temperature = jnp.array(1.0, dtype=jnp.float32)

      decoding_order, prng_key = decoding_order_fn(
        prng_key,
        structure_coordinates.shape[0],
        tie_group_map,
        num_groups,
      )

      optimized_sequence, final_logits, _ = optimize_fn(
        prng_key,
        structure_coordinates,
        mask,
        residue_index,
        chain_index,
        iterations,
        learning_rate,
        temperature,
        backbone_noise,
        tie_group_map,
        num_groups,
        structure_mapping,
        multi_state_strategy=multi_state_strategy,
        multistate_mode=multistate_mode,
        multi_state_temperature=multi_state_temperature,
        fixed_mask=fixed_mask,
        fixed_tokens=fixed_tokens,
        **kwargs,
      )

      return optimized_sequence, final_logits, decoding_order

    return cast("SamplerFn", sample_sequences)

  if sampling_strategy == "temperature":

    @partial(
      jax.jit,
      static_argnames=(
        "num_groups",
        "multi_state_strategy",
        "multistate_mode",
        "max_group_size",
        "state_vmap_n_canonical",
      ),
    )
    def sample_sequences(
      prng_key: PRNGKeyArray,
      structure_coordinates: StructureAtomicCoordinates,
      mask: AlphaCarbonMask,
      residue_index: ResidueIndex,
      chain_index: ChainIndex,
      bias: InputBias | None = None,
      fixed_positions: jnp.ndarray | None = None,
      fixed_mask: jnp.ndarray | None = None,
      fixed_tokens: jnp.ndarray | None = None,
      backbone_noise: BackboneNoise | None = None,
      iterations: Int | None = None,
      learning_rate: Float | None = None,
      temperature: Float | None = None,
      tie_group_map: jnp.ndarray | None = None,
      num_groups: int | None = None,
      multi_state_strategy: Literal[
        "arithmetic_mean",
        "geometric_mean",
        "product",
      ] = "arithmetic_mean",
      multistate_mode: Literal["flat", "state_vmap", "state_vmap_exact"] = "flat",
      structure_mapping: jax.Array | None = None,
      multi_state_temperature: Float = 1.0,
      max_group_size: int = 16,
      precomputed_node_features: jnp.ndarray | None = None,
      precomputed_edge_features: jnp.ndarray | None = None,
      precomputed_neighbor_indices: jnp.ndarray | None = None,
      state_vmap_n_canonical: int = 0,
      coords_stack: jnp.ndarray | None = None,
      mask_stack: jnp.ndarray | None = None,
      residue_index_stack: jnp.ndarray | None = None,
      chain_index_stack: jnp.ndarray | None = None,
      tie_group_map_stack: jnp.ndarray | None = None,
      fixed_mask_stack: jnp.ndarray | None = None,
      fixed_tokens_stack: jnp.ndarray | None = None,
      bias_stack: jnp.ndarray | None = None,
      state_flat_rows: jnp.ndarray | None = None,
      wave_group_ids_local: jnp.ndarray | None = None,
      wave_group_positions_local: jnp.ndarray | None = None,
      wave_group_valid_local: jnp.ndarray | None = None,
      wave_position_valid_local: jnp.ndarray | None = None,
      y_stack: jnp.ndarray | None = None,
      y_t_stack: jnp.ndarray | None = None,
      y_m_stack: jnp.ndarray | None = None,
      multistate_stack: MultistateStackPayload | None = None,
      ligand_stack: LigandStack | None = None,
      **kwargs: Any,
    ) -> tuple[ProteinSequence, Logits, DecodingOrder]:
      """Sample a sequence from a structure using the ProteinMPNN model.

      Args:
        prng_key: JAX random key.
        structure_coordinates: Atomic coordinates (N, 4, 3).
        mask: Alpha carbon mask indicating valid residues.
        residue_index: Residue indices.
        chain_index: Chain indices.
        bias: Optional bias to add to logits (N, 21).
        fixed_positions: Optional mask for positions to keep fixed (not implemented yet).
        backbone_noise: Optional noise for backbone coordinates.
        iterations: Unused in temperature mode (for API compatibility).
        learning_rate: Unused in temperature mode (for API compatibility).
        temperature: Temperature for sampling (default: 1.0).
        tie_group_map: Optional (N,) array mapping positions to group IDs for tied sampling.
        num_groups: Number of unique groups when using tied positions.
        multi_state_strategy: Strategy for combining logits across tied positions
          ("arithmetic_mean", "geometric_mean", "product").
        structure_mapping: Optional (N,) array mapping each residue to a structure ID.
                  When provided (multi-state mode), prevents cross-structure
                  neighbors to avoid information leakage between conformational states.
        multi_state_temperature: Temperature for geometric_mean multi-state combining.
        **kwargs: Additional arguments for LigandMPNN (Y, Y_t, Y_m) or weighting.


      Returns:
        Tuple of (sampled sequence, logits, decoding order).

      Example:
        >>> seq, logits, order = sample_sequences(
        ...     key, coords, mask, res_idx, chain_idx, temperature=0.1
        ... )

      """
      del iterations, learning_rate
      if fixed_mask is None and fixed_positions is not None:
        fixed_mask = fixed_positions

      if temperature is None:
        temperature = jnp.array(1.0, dtype=jnp.float32)

      decoding_order, prng_key = decoding_order_fn(
        prng_key,
        structure_coordinates.shape[0],
        tie_group_map,
        num_groups,
      )
      autoregressive_mask = cast("Callable[..., Any]", generate_ar_mask)(
        decoding_order, None, tie_group_map, num_groups,
      )

      if multistate_mode == "state_vmap_exact":
        if multistate_stack is not None and coords_stack is not None:
          msg = "state_vmap_exact: pass either multistate_stack= or coords_stack=..., not both"
          raise ValueError(msg)
        if multistate_stack is not None:
          coords_stack = multistate_stack.coords_stack
          mask_stack = multistate_stack.mask_stack
          residue_index_stack = multistate_stack.residue_index_stack
          chain_index_stack = multistate_stack.chain_index_stack
          tie_group_map_stack = multistate_stack.tie_group_map_stack
          fixed_mask_stack = multistate_stack.fixed_mask_stack
          fixed_tokens_stack = multistate_stack.fixed_tokens_stack
          state_flat_rows = multistate_stack.state_flat_rows
        if is_ligand_mpnn:
          if ligand_stack is not None:
            if y_stack is not None or y_t_stack is not None or y_m_stack is not None:
              msg = "state_vmap_exact: pass either ligand_stack= or y_stack= / y_t_stack= / y_m_stack=, not both"
              raise ValueError(msg)
            y_stack = ligand_stack.y_stack
            y_t_stack = ligand_stack.y_t_stack
            y_m_stack = ligand_stack.y_m_stack
          if y_stack is None or y_t_stack is None or y_m_stack is None:
            msg = (
              "state_vmap_exact for PrxteinLigandMPNN requires y_stack, y_t_stack, and y_m_stack "
              "(or ligand_stack=), stacked from flat ligand tensors via slice_flat_tensor_to_stack."
            )
            raise ValueError(msg)
        if (
          coords_stack is None
          or mask_stack is None
          or residue_index_stack is None
          or chain_index_stack is None
          or tie_group_map_stack is None
          or fixed_mask_stack is None
          or fixed_tokens_stack is None
          or state_flat_rows is None
          or wave_group_ids_local is None
          or wave_group_positions_local is None
          or wave_group_valid_local is None
          or wave_position_valid_local is None
        ):
          msg = (
            "state_vmap_exact requires coords_stack, mask_stack, residue_index_stack, "
            "chain_index_stack, tie_group_map_stack, fixed_mask_stack, fixed_tokens_stack, "
            "state_flat_rows, and local wave_group_* tensors (or multistate_stack= carrying them)."
          )
          raise ValueError(msg)

        def _ar_submatrix(rows: jnp.ndarray) -> jnp.ndarray:
          g = jnp.where(rows >= 0, rows, 0)
          sub = autoregressive_mask[g[:, None], g[None, :]]
          valid = rows >= 0
          vm = valid[:, None] & valid[None, :]
          return jnp.where(vm, sub, jnp.zeros_like(sub))

        autoregressive_mask_stack = jax.vmap(_ar_submatrix)(state_flat_rows)

        strategy_map = {"arithmetic_mean": 0, "geometric_mean": 1, "product": 2}
        multi_state_strategy_idx = jnp.asarray(strategy_map[multi_state_strategy], dtype=jnp.int32)

        bias_s = (
          bias_stack
          if bias_stack is not None
          else jnp.zeros(
            (coords_stack.shape[0], coords_stack.shape[1], 21),
            dtype=jnp.float32,
          )
        )
        state_weights = kwargs.get("state_weights")
        if state_weights is None:
          n_s = int(coords_stack.shape[0])
          state_weights = jnp.ones((n_s,), dtype=jnp.float32) / jnp.float32(n_s)

        if is_ligand_mpnn:
          ls = LigandStack(
            y_stack=cast("jax.Array", y_stack),
            y_t_stack=cast("jax.Array", y_t_stack),
            y_m_stack=cast("jax.Array", y_m_stack),
          )
          if multistate_stack is not None:
            sampled_sequence, logits = model.sample_autoregressive_state_vmap_exact_from_payload(
              prng_key,
              multistate_stack,
              ls,
              autoregressive_mask_stack,
              bias_s,
              temperature,
              multi_state_strategy_idx,
              jnp.asarray(multi_state_temperature, dtype=jnp.float32),
              state_weights,
              wave_group_ids_local,
              wave_group_positions_local,
              wave_group_valid_local,
              wave_position_valid_local,
            )
          else:
            sampled_sequence, logits = model.sample_autoregressive_state_vmap_exact(
              prng_key,
              coords_stack,
              mask_stack,
              residue_index_stack,
              chain_index_stack,
              autoregressive_mask_stack,
              tie_group_map_stack,
              bias_s,
              temperature,
              multi_state_strategy_idx,
              jnp.asarray(multi_state_temperature, dtype=jnp.float32),
              state_weights,
              fixed_mask_stack,
              fixed_tokens_stack,
              ls.y_stack,
              ls.y_t_stack,
              ls.y_m_stack,
              wave_group_ids_local,
              wave_group_positions_local,
              wave_group_valid_local,
              wave_position_valid_local,
            )
        elif multistate_stack is not None:
          sampled_sequence, logits = model.sample_autoregressive_state_vmap_exact_from_payload(
            prng_key,
            multistate_stack,
            autoregressive_mask_stack,
            bias_s,
            temperature,
            multi_state_strategy_idx,
            jnp.asarray(multi_state_temperature, dtype=jnp.float32),
            state_weights,
            wave_group_ids_local,
            wave_group_positions_local,
            wave_group_valid_local,
            wave_position_valid_local,
          )
        else:
          sampled_sequence, logits = model.sample_autoregressive_state_vmap_exact(
            prng_key,
            coords_stack,
            mask_stack,
            residue_index_stack,
            chain_index_stack,
            autoregressive_mask_stack,
            tie_group_map_stack,
            bias_s,
            temperature,
            multi_state_strategy_idx,
            jnp.asarray(multi_state_temperature, dtype=jnp.float32),
            state_weights,
            fixed_mask_stack,
            fixed_tokens_stack,
            wave_group_ids_local,
            wave_group_positions_local,
            wave_group_valid_local,
            wave_position_valid_local,
          )
        nc = state_vmap_n_canonical
        sampled_sequence = sampled_sequence[0, :nc, :]
        logits = logits[0, :nc, :]
        one_hot_ndim = 2
        if sampled_sequence.ndim == one_hot_ndim:
          sampled_sequence = sampled_sequence.argmax(axis=-1).astype(jnp.int8)
        return sampled_sequence, logits, decoding_order

      call_kwargs = {
        "decoding_approach": "autoregressive",
        "prng_key": prng_key,
        "ar_mask": autoregressive_mask,
        "temperature": temperature,
        "bias": bias,
        "backbone_noise": backbone_noise,
        "tie_group_map": tie_group_map,
        "num_groups": num_groups,
        "multi_state_strategy": multi_state_strategy,
        "multistate_mode": multistate_mode,
        "structure_mapping": structure_mapping,
        "group_indices_table": kwargs.get("group_indices_table"),
        "group_valid_table": kwargs.get("group_valid_table"),
        "wave_group_ids": kwargs.get("wave_group_ids"),
        "wave_group_positions": kwargs.get("wave_group_positions"),
        "wave_group_valid": kwargs.get("wave_group_valid"),
        "wave_position_valid": kwargs.get("wave_position_valid"),
      }

      if not is_ligand_mpnn:
        call_kwargs["precomputed_node_features"] = precomputed_node_features
        call_kwargs["precomputed_edge_features"] = precomputed_edge_features
        call_kwargs["precomputed_neighbor_indices"] = precomputed_neighbor_indices

      if supports_multi_state_temperature:
        call_kwargs["multi_state_temperature"] = multi_state_temperature

      if supports_state_weights:
        call_kwargs["state_weights"] = kwargs.get("state_weights")
        call_kwargs["state_mapping"] = kwargs.get("state_mapping")

      if supports_fixed_controls:
        call_kwargs["fixed_mask"] = fixed_mask
        call_kwargs["fixed_tokens"] = fixed_tokens

      if is_ligand_mpnn:
        call_kwargs["Y"] = kwargs.get("Y")
        call_kwargs["Y_t"] = kwargs.get("Y_t")
        call_kwargs["Y_m"] = kwargs.get("Y_m")
        call_kwargs["xyz_37"] = kwargs.get("xyz_37")
        call_kwargs["xyz_37_m"] = kwargs.get("xyz_37_m")
        call_kwargs["chain_mask"] = kwargs.get("chain_mask")

      sampled_sequence, logits = model(
        structure_coordinates,
        mask,
        residue_index,
        chain_index,
        **call_kwargs,
      )

      one_hot_ndim = 2
      if sampled_sequence.ndim == one_hot_ndim:
        sampled_sequence = sampled_sequence.argmax(axis=-1).astype(jnp.int8)

      return sampled_sequence, logits, decoding_order

    return cast("SamplerFn", sample_sequences)

  msg = f"Unknown sampling strategy: {sampling_strategy}"
  raise ValueError(msg)
