"""Implements sequence optimization by guiding a differentiable autoregressive decoder."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Float, Int, PRNGKeyArray
from optax import Params

from prxteinmpnn.model.decoding_signatures import (
  RunConditionalDecoderFn,
)
from prxteinmpnn.model.projection import final_projection
from prxteinmpnn.model.ste import straight_through_estimator
from prxteinmpnn.utils.autoregression import generate_ar_mask
from prxteinmpnn.utils.decoding_order import DecodingOrderFn
from prxteinmpnn.utils.types import (
  AlphaCarbonMask,
  AtomMask,
  EdgeFeatures,
  Logits,
  ModelParameters,
  NeighborIndices,
  NodeFeatures,
  OneHotProteinSequence,
)


def make_optimize_sequence_fn(
  decoder: RunConditionalDecoderFn,
  decoding_order_fn: DecodingOrderFn,
  model_parameters: ModelParameters,
) -> Callable[
  [
    PRNGKeyArray,
    NodeFeatures,
    EdgeFeatures,
    NeighborIndices,
    AtomMask,
    Int,
    Float,
    Float,
  ],
  tuple[OneHotProteinSequence, Logits],
]:
  """Create a function to optimize a sequence using the STE decoder.

  Args:
    decoder: The conditional decoder.
    decoding_order_fn: Function to generate decoding orders.
    model_parameters: Model parameters for the decoder.

  Returns:
    A function that takes a PRNG key, node features, edge features, neighbor indices,
    atom mask, number of optimization steps, and learning rate, and returns the optimized sequence
    and its logits.

  """

  def optimize_sequence(
    prng_key: PRNGKeyArray,
    node_features: NodeFeatures,
    edge_features: EdgeFeatures,
    neighbor_indices: NeighborIndices,
    mask: AlphaCarbonMask,
    iterations: Int,
    learning_rate: Float,
    temperature: Float,
    batch_size: int = 4,
  ) -> tuple[OneHotProteinSequence, Logits]:
    """Optimize a sequence by finding a self-consistent sequence for the conditional decoder.

    Args:
      prng_key: JAX PRNG key for random operations.
      node_features: Node features for the structure.
      edge_features: Edge features for the structure.
      neighbor_indices: Indices of neighboring nodes.
      mask: Atom mask indicating valid positions.
      iterations: Number of optimization steps to perform.
      learning_rate: Learning rate for the optimizer.
      temperature: Temperature for the softmax distribution.
      batch_size: The batch size for decoding.

    Returns:
      A tuple containing the optimized sequence as a one-hot encoded array and the logits.

    """
    num_residues, _ = node_features.shape
    num_classes = 21

    sequence_logits = jnp.zeros((num_residues, num_classes))

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(sequence_logits)

    node_features_batched = jnp.repeat(
      node_features[jnp.newaxis, ...],
      batch_size,
      axis=0,
    )
    edge_features_batched = jnp.repeat(
      edge_features[jnp.newaxis, ...],
      batch_size,
      axis=0,
    )
    neighbor_indices_batched = jnp.repeat(
      neighbor_indices[jnp.newaxis, ...],
      batch_size,
      axis=0,
    )

    mask_batched = jnp.repeat(mask[jnp.newaxis, ...], batch_size, axis=0)

    @jax.jit
    def update_step(
      carry: tuple[Logits, optax.OptState],
      key: PRNGKeyArray,
    ) -> tuple[tuple[Params, optax.OptState], jnp.ndarray]:
      current_sequence_logits, current_opt_state = carry
      keys_for_decoding = jax.random.split(key, batch_size)

      decoding_order, _ = jax.vmap(decoding_order_fn, in_axes=(0, None))(
        keys_for_decoding,
        num_residues,
      )
      ar_mask = jax.vmap(generate_ar_mask)(decoding_order)

      def loss_fn(logits: Logits) -> jnp.ndarray:
        one_hot_sequence = straight_through_estimator(logits / temperature)
        one_hot_sequence_batched = jnp.repeat(
          one_hot_sequence[jnp.newaxis, ...],
          batch_size,
          axis=0,
        )

        decoded_features = decoder(
          node_features_batched,
          edge_features_batched,
          neighbor_indices_batched,
          mask_batched,
          ar_mask,
          one_hot_sequence_batched,
        )
        output_logits = final_projection(model_parameters, decoded_features)

        loss = optax.softmax_cross_entropy(logits=output_logits, labels=one_hot_sequence)

        return (loss * mask).sum() / (mask.sum() + 1e-8)

      loss_value, grads = jax.value_and_grad(loss_fn)(current_sequence_logits)
      updates, next_opt_state = optimizer.update(grads, current_opt_state)
      next_sequence_logits = optax.apply_updates(current_sequence_logits, updates)
      return (next_sequence_logits, next_opt_state), loss_value

    keys = jax.random.split(prng_key, iterations)
    (final_sequence_logits, _), _ = jax.lax.scan(
      update_step,
      (sequence_logits, opt_state),
      keys,
    )

    final_one_hot = straight_through_estimator(final_sequence_logits)

    final_key, _ = jax.random.split(prng_key)
    final_decoding_order, _ = decoding_order_fn(final_key, num_residues)
    final_ar_mask = generate_ar_mask(final_decoding_order)

    final_decoded_features = decoder(
      node_features,
      edge_features,
      neighbor_indices,
      mask,
      final_ar_mask,
      final_one_hot,
    )
    final_logits = final_projection(model_parameters, final_decoded_features)

    return final_one_hot, final_logits

  return optimize_sequence
