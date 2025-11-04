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

from prxteinmpnn.sampling.unconditional_logits import make_unconditional_logits_fn
from prxteinmpnn.utils.autoregression import generate_ar_mask
from prxteinmpnn.utils.decoding_order import DecodingOrderFn, random_decoding_order
from prxteinmpnn.utils.ste import straight_through_estimator


def make_optimize_sequence_fn(
  model: PrxteinMPNN,
  decoding_order_fn: DecodingOrderFn = random_decoding_order,
  batch_size: int = 4,
) -> callable:
  """Create a function to optimize sequences using straight-through estimation.

  The optimization works by:
  1. Getting target (unconditional) logits from the model
  2. Initializing learnable logits to optimize
  3. For each iteration:
     - Apply STE to get discrete sequence from logits
     - Compute loss between current logits and target logits
     - Update logits via gradient descent
  4. Return final optimized sequence

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
  # Create unconditional logits function for target
  unconditional_fn = make_unconditional_logits_fn(model)

  @partial(jax.jit)
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
  ) -> tuple[ProteinSequence, Logits]:
    """Optimize a sequence by finding self-consistent logits.

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

    Returns:
      Tuple of (optimized sequence, final logits).

    Example:
      >>> seq, logits = optimize_sequence(
      ...     key, coords, mask, res_idx, chain_idx,
      ...     iterations=100, learning_rate=0.01, temperature=1.0
      ... )

    """
    num_residues = structure_coordinates.shape[0]
    num_classes = 21

    # Get target logits (unconditional prediction)
    # Use a dummy key since unconditional doesn't need randomness
    target_key = jax.random.key(0)
    target_logits = unconditional_fn(
      target_key,
      structure_coordinates,
      mask,
      residue_index,
      chain_index,
      ar_mask=None,
      backbone_noise=backbone_noise,
    )

    # Initialize sequence logits to optimize
    sequence_logits = jnp.zeros((num_residues, num_classes), dtype=jnp.float32)

    # Set up optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(sequence_logits)

    def update_step(
      _iteration: int,
      carry: tuple[Logits, optax.OptState, PRNGKeyArray],
    ) -> tuple[Logits, optax.OptState, PRNGKeyArray]:
      """Single optimization step.

      Args:
        _iteration: Current iteration (unused, required by fori_loop).
        carry: Tuple of (current_logits, optimizer_state, prng_key).

      Returns:
        Updated tuple of (logits, optimizer_state, prng_key).

      """
      current_logits, current_opt_state, current_key = carry
      key, next_key = jax.random.split(current_key)

      # Generate multiple decoding orders for robust optimization
      keys_for_decoding = jax.random.split(key, batch_size)

      # Generate decoding orders and autoregressive masks
      decoding_orders, _ = jax.vmap(decoding_order_fn, in_axes=(0, None))(
        keys_for_decoding,
        num_residues,
      )
      ar_masks = jax.vmap(generate_ar_mask)(decoding_orders)

      def loss_fn(logits: Logits) -> Float:
        """Compute loss for current logits.

        Loss is the cross-entropy between the STE-discretized sequence
        and the target (unconditional) logits, averaged over multiple
        decoding orders.

        Args:
          logits: Current logits to optimize.

        Returns:
          Scalar loss value.

        """
        # Apply straight-through estimator with temperature
        one_hot_sequence = straight_through_estimator(logits / temperature)

        # Get predictions for this sequence under different AR masks
        def eval_with_mask(ar_mask: AutoRegressiveMask) -> Logits:
          # Run conditional mode with the sequence
          _, pred_logits = model(
            structure_coordinates,
            mask,
            residue_index,
            chain_index,
            decoding_approach="conditional",
            one_hot_sequence=one_hot_sequence,
            ar_mask=ar_mask,
            backbone_noise=backbone_noise,
          )
          return pred_logits

        # Average predictions over multiple decoding orders
        pred_logits_batch = jax.vmap(eval_with_mask)(ar_masks)
        pred_logits = jnp.mean(pred_logits_batch, axis=0)

        # Compute cross-entropy loss between predictions and target
        loss = optax.softmax_cross_entropy(
          logits=pred_logits,
          labels=jax.nn.softmax(target_logits),
        )

        # Mask invalid positions and return mean loss
        return (loss * mask).sum() / (mask.sum() + 1e-8)

      # Compute loss and gradients
      _, grads = jax.value_and_grad(loss_fn)(current_logits)

      # Update logits
      updates, next_opt_state = optimizer.update(grads, current_opt_state)
      next_logits = optax.apply_updates(current_logits, updates)

      return next_logits, next_opt_state, next_key

    # Run optimization loop
    final_logits, _, _ = jax.lax.fori_loop(
      0,
      iterations,
      update_step,
      (sequence_logits, opt_state, prng_key),
    )

    # Get final sequence from optimized logits
    final_one_hot = straight_through_estimator(final_logits / temperature)
    final_sequence = final_one_hot.argmax(axis=-1).astype(jnp.int8)

    return final_sequence, final_logits

  return optimize_sequence

