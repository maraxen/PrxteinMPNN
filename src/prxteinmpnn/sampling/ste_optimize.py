"""Implements sequence optimization by guiding a differentiable autoregressive decoder."""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import optax
from jaxtyping import PRNGKeyArray

from prxteinmpnn.model.decoding_signatures import (
  RunAutoregressiveDecoderFn,
  RunConditionalDecoderFn,
)
from prxteinmpnn.model.projection import final_projection
from prxteinmpnn.utils.autoregression import generate_ar_mask
from prxteinmpnn.utils.decoding_order import DecodingOrderFn
from prxteinmpnn.utils.types import (
  AtomMask,
  EdgeFeatures,
  Logits,
  ModelParameters,
  NeighborIndices,
  NodeFeatures,
  OneHotProteinSequence,
)


@partial(jax.jit, static_argnames=("conditional_decoder"))
def _score_sequence_loss(
  generated_one_hot: jax.Array,
  conditional_decoder: RunConditionalDecoderFn,
  model_parameters: ModelParameters,
  node_features: NodeFeatures,
  edge_features: EdgeFeatures,
  autoregressive_mask: jax.Array,
  neighbor_indices: NeighborIndices,
  mask: AtomMask,
) -> jnp.ndarray:
  """Calculate the negative log-likelihood of a sequence using the conditional decoder."""
  scored_features = conditional_decoder(
    node_features,
    edge_features,
    neighbor_indices,
    mask,
    autoregressive_mask,
    generated_one_hot,
  )
  scored_logits = final_projection(model_parameters, scored_features)

  log_probs = jax.nn.log_softmax(scored_logits, axis=-1)
  loss_per_position = -jnp.sum(generated_one_hot * log_probs, axis=-1)

  return (loss_per_position * mask).sum() / (mask.sum() + 1e-8)


def make_optimize_sequence_fn(
  autoregressive_decoder: RunAutoregressiveDecoderFn,
  conditional_decoder: RunConditionalDecoderFn,
  decoding_order_fn: DecodingOrderFn,
  model_parameters: ModelParameters,
) -> Callable[
  [PRNGKeyArray, NodeFeatures, EdgeFeatures, NeighborIndices, AtomMask, int, float, float],
  tuple[OneHotProteinSequence, Logits],
]:
  """Create a function to optimize a sequence using the STE autoregressive decoder.

  Args:
    autoregressive_decoder: The autoregressive decoder function with STE.
    conditional_decoder: The conditional decoder function to score sequences.
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
    mask: AtomMask,
    num_steps: int,
    learning_rate: float,
    temperature: float,
  ) -> tuple[OneHotProteinSequence, Logits]:
    """Optimize a sequence by guiding the STE autoregressive decoder.

    Args:
      prng_key: JAX PRNG key for random operations.
      node_features: Node features for the structure.
      edge_features: Edge features for the structure.
      neighbor_indices: Indices of neighboring nodes.
      mask: Atom mask indicating valid positions.
      num_steps: Number of optimization steps to perform.
      learning_rate: Learning rate for the optimizer.

    Returns:
      A tuple containing the optimized sequence as a one-hot encoded array and the logits.

    """
    num_residues, num_classes = node_features.shape[0], 21

    guiding_logits = jnp.zeros((num_residues, num_classes))

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(guiding_logits)

    @jax.jit
    def update_step(
      carry: tuple[Logits, optax.OptState],
      key: PRNGKeyArray,
    ) -> tuple[tuple[Logits, optax.OptState], jnp.ndarray]:
      current_guiding_logits, current_opt_state = carry

      sampler_decoding_order, next_key = decoding_order_fn(key, num_residues)
      score_decoding_order, next_key = decoding_order_fn(next_key, num_residues)
      ar_mask = generate_ar_mask(sampler_decoding_order)
      score_ar_mask = generate_ar_mask(score_decoding_order)

      def loss_fn(guides: Logits) -> jnp.ndarray:
        generated_sequence_one_hot, _ = autoregressive_decoder(
          next_key,
          node_features,
          edge_features,
          neighbor_indices,
          mask,
          ar_mask,
          temperature,
          guides,
        )

        return _score_sequence_loss(
          generated_sequence_one_hot,
          conditional_decoder,
          model_parameters,
          node_features,
          edge_features,
          score_ar_mask,
          neighbor_indices,
          mask,
        )

      loss_value, grads = jax.value_and_grad(loss_fn)(current_guiding_logits)
      updates, next_opt_state = optimizer.update(grads, current_opt_state)
      next_guiding_logits = optax.apply_updates(current_guiding_logits, updates)

      return (next_guiding_logits, next_opt_state), loss_value  # type: ignore[return-value]

    keys = jax.random.split(prng_key, num_steps)
    (final_guiding_logits, _), losses = jax.lax.scan(
      update_step,
      (guiding_logits, opt_state),
      keys,
    )

    final_key, prng_key = jax.random.split(prng_key)
    final_decoding_order, _ = decoding_order_fn(final_key, num_residues)
    final_ar_mask = generate_ar_mask(final_decoding_order)

    final_sequence_one_hot, final_logits = autoregressive_decoder(
      prng_key,
      node_features,
      edge_features,
      neighbor_indices,
      mask,
      final_ar_mask,
      temperature,
      final_guiding_logits,
    )

    return final_sequence_one_hot, final_logits

  return optimize_sequence
