"""Straight-through estimator optimization for sequence design.

This module implements iterative sequence optimization using the straight-through
estimator (STE) to allow gradients through discrete sampling operations.

Rematerialization (jax.checkpoint) is used to reduce memory during backward passes
by recomputing activations on demand instead of storing them. Activate via the
JAX_ENABLE_REMAT environment variable to trade CPU speed for memory efficiency.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp
import optax

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
  from collections.abc import Callable

  from jaxtyping import Float, Int, PRNGKeyArray

  from prxteinmpnn.io.designs import DesignArrayRecordWriter, DesignMetadata
  from prxteinmpnn.model import PrxteinMPNN
  from prxteinmpnn.utils.types import (
    AlphaCarbonMask,
    AutoRegressiveMask,
    BackboneNoise,
    ChainIndex,
    Logits,
    ProteinSequence,
    ResidueIndex,
    StructureAtomicCoordinates,
  )

from prxteinmpnn.registry import assert_known_multistate_mode
from prxteinmpnn.utils.autoregression import generate_ar_mask
from prxteinmpnn.utils.decoding_order import DecodingOrderFn, random_decoding_order
from prxteinmpnn.utils.ste import gumbel_softmax, straight_through_estimator

_DEFAULT_DECODING_ORDER_FN = cast("DecodingOrderFn", random_decoding_order)


def make_optimize_sequence_fn(
  model: PrxteinMPNN,
  decoding_order_fn: DecodingOrderFn = _DEFAULT_DECODING_ORDER_FN,
  batch_size: int = 4,
  use_concrete: bool = False,
  tau_start: float = 1.0,
  tau_end: float = 0.1,
) -> Callable[..., tuple[ProteinSequence, Logits, Logits]]:
  """Create a function to optimize sequences using straight-through estimation.

  This matches the original implementation which:
  1. Initializes learnable logits to optimize
  2. For each iteration:
     - Apply STE to get discrete sequence from logits
     - Run autoregressive decoder with that sequence
     - Compute loss between decoder output and current logits
     - Update logits via gradient descent
  3. Return final optimized sequence

  The key insight: we're finding a sequence that is self-consistent with
  the autoregressive decoder's predictions under multiple decoding orders.

  When tied positions are provided, positions in the same group are constrained
  to have identical logits throughout optimization, ensuring they converge to
  the same amino acid.

  Args:
    model: A PrxteinMPNN Equinox model instance.
    decoding_order_fn: Function to generate decoding orders (default: random).
    batch_size: Number of decoding orders to use per optimization step.
    use_concrete: If True, use Gumbel-softmax (Concrete) relaxation instead of
        argmax straight-through. Prevents mode collapse by giving all AAs a
        gradient signal proportional to their probability at every step.
    tau_start: Initial Gumbel-softmax temperature (broad, exploratory). Ignored
        when use_concrete=False.
    tau_end: Final Gumbel-softmax temperature after annealing (focused, near-argmax).
        Ignored when use_concrete=False.

  Returns:
    A function that optimizes sequences for a given structure.

  Example:
    >>> from prxteinmpnn.io.weights import load_model
    >>> model = load_model()
    >>> optimize_fn = make_optimize_sequence_fn(model)
    >>> seq, logits = optimize_fn(
    ...     key, coords, mask, res_idx, chain_idx,
    ...     iterations=100, learning_rate=0.01, temperature=1.0
    ... )

  """

  @partial(jax.jit, static_argnames=("num_groups", "multi_state_strategy", "multistate_mode"))
  def optimize_sequence(
    prng_key: PRNGKeyArray,
    structure_coordinates: StructureAtomicCoordinates,
    mask: AlphaCarbonMask,
    residue_index: ResidueIndex,
    chain_index: ChainIndex,
    iterations: Int,
    learning_rate: Float,
    temperature: Float,
    backbone_noise: BackboneNoise | None = None,
    tie_group_map: jnp.ndarray | None = None,
    num_groups: int | None = None,
    structure_mapping: jax.Array | None = None,
    multi_state_strategy: str = "arithmetic_mean",
    multistate_mode: str = "flat",
    multi_state_temperature: Float = 1.0,
    fixed_mask: jnp.ndarray | None = None,
    fixed_tokens: jnp.ndarray | None = None,
    writer: DesignArrayRecordWriter | None = None,
    **kwargs: Any,
  ) -> tuple[ProteinSequence, Logits, Logits]:
    """Optimize a sequence by finding self-consistent logits via autoregressive decoder.

    This matches the original implementation where we optimize logits such that
    when discretized via STE and fed through the autoregressive decoder,
    the decoder's output matches our logits. This creates a self-consistent sequence.

    When tied positions are provided, the optimization ensures that positions in
    the same group maintain identical logits throughout the optimization process.

    Args:
      prng_key: JAX PRNG key for random operations.
      structure_coordinates: Atomic coordinates (N, 4, 3).
      mask: Alpha carbon mask indicating valid residues.
      residue_index: Residue indices.
      chain_index: Chain indices.
      iterations: Number of optimization steps to perform.
      learning_rate: Learning rate for the optimizer.
      temperature: Temperature for the STE softmax distribution.
      backbone_noise: Optional noise for backbone coordinates.
      tie_group_map: Optional (N,) array mapping each position to a group ID.
          When provided, positions in the same group are constrained to have
          identical logits during optimization.
      num_groups: Number of unique groups when using tied positions.
      structure_mapping: Optional (N,) array mapping each residue to a structure ID.
      multi_state_strategy: Strategy for combining logits across tied positions
          ("arithmetic_mean", "geometric_mean", "product").
      multi_state_temperature: Temperature for geometric_mean multi-state combining.
      **kwargs: Additional arguments for LigandMPNN (Y, Y_t, Y_m) or weighting.

    Environment Variables:
      JAX_ENABLE_REMAT: Enable rematerialization to reduce peak memory usage during
          backward passes. Accepted values (case-insensitive): "1", "true", "yes", "on",
          "enabled". Default is disabled (false).

          Trade-off: Rematerialization recomputes layer activations during gradient
          descent instead of storing them, trading memory for CPU speed. On CPU, expect
          ~3x slowdown. On GPU with memory pressure, net throughput gain is typical as
          the GPU avoids stalling on host-device memory transfers.

          Trace-time caveat: The setting is captured at function definition time when
          JAX traces the computation graph. Toggling the variable mid-process will not
          update already-compiled versions. To change the setting, restart the process.

          Example::

              # Via environment variable
              JAX_ENABLE_REMAT=1 uv run python3 optimize.py

              # Or in Python
              import os
              os.environ["JAX_ENABLE_REMAT"] = "1"
              # Then define/import the function to pick up the setting

    Returns:
      Tuple of (optimized sequence, final output logits, optimized logits).

    Example:
      >>> seq, output_logits, opt_logits = optimize_sequence(
      ...     key, coords, mask, res_idx, chain_idx,
      ...     iterations=100, learning_rate=0.01, temperature=1.0
      ... )

    """
    assert_known_multistate_mode(multistate_mode)
    num_residues = structure_coordinates.shape[0]
    num_classes = 21

    # Build clamping bias for fixed positions: large positive logit forces argmax
    # to the correct token, propagating the right embedding through optimization.
    CLAMP_LOGIT = 1e4
    if fixed_mask is not None and fixed_tokens is not None:
      fixed_mask_bool = fixed_mask.astype(jnp.bool_)
      fixed_one_hot = jax.nn.one_hot(fixed_tokens, num_classes, dtype=jnp.float32)
      fixed_bias = fixed_mask_bool[:, None] * (
        CLAMP_LOGIT * fixed_one_hot - CLAMP_LOGIT * (1.0 - fixed_one_hot)
      )
    else:
      fixed_bias = jnp.zeros((num_residues, num_classes), dtype=jnp.float32)
      fixed_mask_bool = jnp.zeros(num_residues, dtype=jnp.bool_)

    sequence_logits = jnp.zeros((num_residues, num_classes), dtype=jnp.float32)

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(sequence_logits)

    def update_step(
      _iteration: int,
      carry: tuple[Logits, optax.OptState, PRNGKeyArray],
    ) -> tuple[Logits, optax.OptState, PRNGKeyArray]:
      """Single optimization step matching original implementation.

      Args:
        _iteration: Current iteration (unused, required by fori_loop).
        carry: Tuple of (current_logits, optimizer_state, prng_key).

      Returns:
        Updated tuple of (logits, optimizer_state, prng_key).

      """
      current_logits, current_opt_state, current_key = carry

      if use_concrete:
        key_decoding_orders, gumbel_key, next_key = jax.random.split(current_key, 3)
        # Exponential annealing: tau_start → tau_end over all iterations.
        # Uses traced `iterations` and `_iteration` so no Python branching at step time.
        progress = jnp.float32(_iteration) / jnp.maximum(
          jnp.float32(iterations) - 1.0, 1.0,
        )
        tau = jnp.float32(tau_start) * (jnp.float32(tau_end / tau_start) ** progress)
      else:
        key_decoding_orders, next_key = jax.random.split(current_key)

      keys_for_decoding = jax.random.split(key_decoding_orders, batch_size)
      decoding_orders, _ = jax.vmap(decoding_order_fn, in_axes=(0, None, None, None))(
        keys_for_decoding,
        num_residues,
        tie_group_map,
        num_groups,
      )

      ar_masks = jax.vmap(generate_ar_mask, in_axes=(0, None))(decoding_orders, tie_group_map)

      def loss_fn(logits: Logits) -> Float:
        """Compute self-consistency loss.

        The loss measures how well the autoregressive decoder's predictions
        match our current logits when we feed in the STE-discretized sequence.

        Args:
          logits: Current logits to optimize.

        Returns:
          Scalar loss value.

        """
        if use_concrete:
          # Same Gumbel sample for model input and loss labels — ensures
          # the loss targets the exact sequence the model conditions on.
          seq_repr = gumbel_softmax(logits / temperature, tau, gumbel_key, hard=True)
          one_hot_sequence = seq_repr
        else:
          one_hot_sequence = straight_through_estimator(logits / temperature)

        def eval_with_mask(ar_mask: AutoRegressiveMask) -> Logits:
          call_kwargs = {}
          for k in ["Y", "Y_t", "Y_m", "xyz_37", "xyz_37_m", "chain_mask", "state_weights"]:
            if k in kwargs:
              call_kwargs[k] = kwargs[k]

          _, output_logits = model(
            structure_coordinates,
            mask,
            residue_index,
            chain_index,
            decoding_approach="conditional",
            one_hot_sequence=one_hot_sequence,
            ar_mask=ar_mask,
            backbone_noise=backbone_noise,
            structure_mapping=structure_mapping,
            tie_group_map=tie_group_map,
            multi_state_strategy=multi_state_strategy,
            multistate_mode=multistate_mode,
            **call_kwargs,
          )
          return output_logits

        # Optionally apply rematerialization (jax.checkpoint) to reduce memory usage.
        # This recomputes eval_with_mask during backward instead of storing activations.
        remat_value = os.getenv("JAX_ENABLE_REMAT", "0").lower()
        remat_truthy = {"1", "true", "yes", "on", "enabled"}
        if remat_value not in remat_truthy and remat_value != "0":
          logger.warning(
            f"Unrecognized JAX_ENABLE_REMAT='{remat_value}', treating as disabled. "
            f"Accepted values: {remat_truthy | {'0'}} (case-insensitive).",
          )
        enable_remat = remat_value in remat_truthy
        if enable_remat:
          logger.info("STE rematerialization enabled via JAX_ENABLE_REMAT=1")
          eval_with_mask_remat = jax.checkpoint(eval_with_mask)
          pred_logits_batch = jax.vmap(eval_with_mask_remat)(ar_masks)
        else:
          pred_logits_batch = jax.vmap(eval_with_mask)(ar_masks)

        output_logits = jnp.mean(pred_logits_batch, axis=0)

        labels = seq_repr if use_concrete else straight_through_estimator(logits)
        loss = optax.softmax_cross_entropy(
          logits=output_logits,
          labels=labels,
        )

        return (loss * mask).sum() / (mask.sum() + 1e-8)

      _, grads = jax.value_and_grad(loss_fn)(current_logits)

      updates, next_opt_state = optimizer.update(grads, current_opt_state)
      next_logits: Logits = optax.apply_updates(current_logits, updates)  # type: ignore[invalid-assignment]
      next_logits = jnp.where(fixed_mask_bool[:, None], fixed_bias, next_logits)

      if tie_group_map is not None and num_groups is not None:
        group_one_hot = jax.nn.one_hot(
          tie_group_map,
          num_groups,
          dtype=jnp.float32,
        )  # (N, num_groups)

        group_logit_sums = jnp.einsum("ng,na->ga", group_one_hot, next_logits)
        group_counts = group_one_hot.sum(axis=0)
        group_avg_logits = group_logit_sums / (group_counts[:, None] + 1e-8)
        next_logits = jnp.einsum("ng,ga->na", group_one_hot, group_avg_logits)

      return next_logits, next_opt_state, next_key

    # Run optimization loop
    final_logits, _, final_key = jax.lax.fori_loop(
      0,
      iterations,
      update_step,
      (sequence_logits, opt_state, prng_key),
    )

    # Clamp fixed positions one final time before deriving the output sequence
    final_logits = jnp.where(fixed_mask_bool[:, None], fixed_bias, final_logits)

    # Get final sequence from optimized logits
    final_one_hot = straight_through_estimator(final_logits / temperature)
    final_sequence = final_one_hot.argmax(axis=-1).astype(jnp.int8)

    # Save final logits via io_callback if writer is provided.
    # io_callback is non-differentiable — do not wrap this function in jax.grad.
    # (ste_optimize is already non-differentiable at the STE discretization step.)
    if writer is not None:
      def _save_logits(logits: jnp.ndarray) -> None:
        """Host-side callback to write final logits to the writer."""

        # Prepare sensible defaults for fields not naturally available from STE optimization.
        scores = jnp.array([0.0], dtype=jnp.float32)  # Default score; caller can override.
        # TODO: derive n_states from structure_mapping or caller's state_weights, not hardcoded
        n_states = (int(jnp.max(structure_mapping)) + 1) if structure_mapping is not None else 1
        state_weights = jnp.ones(n_states, dtype=jnp.float32)  # Default weights.
        metadata: DesignMetadata = {
          "pool_type": "BackboneOnly",
          "state_mapping": [],
          "weight_strategy": "uniform",
          "combination_algorithm": "ste_optimization",
          "structure_ids": [],
          "parent_structure_idx": 0,
        }

        payload = {
          "sequence": final_sequence,
          "logits": logits,
          "scores": scores,
          "state_weights": state_weights,
          "metadata": metadata,
        }
        writer.write(payload)

      jax.experimental.io_callback(_save_logits, None, final_logits, ordered=False)  # never ordered=True — syncs host per JAX semantics
      jax.effects_barrier()  # Drain host writes before returning.

    # Get final output logits by running through decoder one more time
    final_decoding_order, _ = decoding_order_fn(final_key, num_residues, tie_group_map, num_groups)
    final_ar_mask = cast("Callable", generate_ar_mask)(final_decoding_order, tie_group_map)

    final_call_kwargs = {}
    for k in ["Y", "Y_t", "Y_m", "xyz_37", "xyz_37_m", "chain_mask", "state_weights"]:
      if k in kwargs:
        final_call_kwargs[k] = kwargs[k]

    _, final_output_logits = model(
      structure_coordinates,
      mask,
      residue_index,
      chain_index,
      decoding_approach="conditional",
      one_hot_sequence=final_one_hot,
      ar_mask=final_ar_mask,
      backbone_noise=backbone_noise,
      structure_mapping=structure_mapping,
      tie_group_map=tie_group_map,
      multi_state_strategy=multi_state_strategy,
      **final_call_kwargs,
    )

    return final_sequence, final_output_logits, final_logits

  return cast("Callable", optimize_sequence)
