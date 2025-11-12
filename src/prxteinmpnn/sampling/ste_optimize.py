"""Straight-through estimator optimization for sequence design.

This module implements iterative sequence optimization using the straight-through
estimator (STE) to allow gradients through discrete sampling operations.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import optax

if TYPE_CHECKING:
  from collections.abc import Callable

  from jaxtyping import Float, Int, PRNGKeyArray

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

from prxteinmpnn.utils.autoregression import generate_ar_mask
from prxteinmpnn.utils.decoding_order import DecodingOrderFn, random_decoding_order
from prxteinmpnn.utils.ste import straight_through_estimator


def make_optimize_sequence_fn(
  model: PrxteinMPNN,
  decoding_order_fn: DecodingOrderFn = random_decoding_order,
  batch_size: int = 4,
) -> Callable[
  [
    PRNGKeyArray,
    StructureAtomicCoordinates,
    AlphaCarbonMask,
    ResidueIndex,
    ChainIndex,
    Int,
    Float,
    Float,
    BackboneNoise | None,
    jnp.ndarray | None,
    int | None,
    jax.Array | None,
  ],
  tuple[ProteinSequence, Logits, Logits],
]:
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

  @partial(jax.jit, static_argnames=("num_groups",))
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

    Returns:
      Tuple of (optimized sequence, final output logits, optimized logits).

    Example:
      >>> seq, output_logits, opt_logits = optimize_sequence(
      ...     key, coords, mask, res_idx, chain_idx,
      ...     iterations=100, learning_rate=0.01, temperature=1.0
      ... )

    """
    num_residues = structure_coordinates.shape[0]
    num_classes = 21

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
        one_hot_sequence = straight_through_estimator(logits / temperature)

        def eval_with_mask(ar_mask: AutoRegressiveMask) -> Logits:
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
          )
          return output_logits

        pred_logits_batch = jax.vmap(eval_with_mask)(ar_masks)
        output_logits = jnp.mean(pred_logits_batch, axis=0)

        loss = optax.softmax_cross_entropy(
          logits=output_logits,
          labels=straight_through_estimator(logits),
        )

        return (loss * mask).sum() / (mask.sum() + 1e-8)

      _, grads = jax.value_and_grad(loss_fn)(current_logits)

      updates, next_opt_state = optimizer.update(grads, current_opt_state)
      next_logits: Logits = optax.apply_updates(current_logits, updates)  # type: ignore[assignment]

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

    # Get final sequence from optimized logits
    final_one_hot = straight_through_estimator(final_logits / temperature)
    final_sequence = final_one_hot.argmax(axis=-1).astype(jnp.int8)

    # Get final output logits by running through decoder one more time
    final_decoding_order, _ = decoding_order_fn(final_key, num_residues, tie_group_map, num_groups)
    final_ar_mask = generate_ar_mask(final_decoding_order, tie_group_map)

    _, final_output_logits = model(
      structure_coordinates,
      mask,
      residue_index,
      chain_index,
      decoding_approach="conditional",
      one_hot_sequence=final_one_hot,
      ar_mask=final_ar_mask,
      backbone_noise=backbone_noise,
    )

    return final_sequence, final_output_logits, final_logits

  return optimize_sequence
