"""Implements sequence optimization using a conditional decoder and STE."""

import jax
import jax.numpy as jnp
import optax
from jaxtyping import PRNGKeyArray

from prxteinmpnn.model.decoder import RunConditionalDecoderFn
from prxteinmpnn.model.projection import final_projection
from prxteinmpnn.model.ste import straight_through_estimator
from prxteinmpnn.utils.types import (
  AtomMask,
  EdgeFeatures,
  Logits,
  ModelParameters,
  NeighborIndices,
  NodeFeatures,
  ProteinSequence,
)


def _cross_entropy_loss(
  generated_one_hot: jax.Array,
  scored_logits: Logits,
  mask: AtomMask,
) -> jnp.ndarray:
  """Calculate the cross-entropy between a generated sequence and the model's scoring."""
  log_probs = jax.nn.log_softmax(scored_logits, axis=-1)
  loss_per_position = -jnp.sum(generated_one_hot * log_probs, axis=-1)
  return (loss_per_position * mask).sum() / (mask.sum() + 1e-8)


def optimize_sequence(
  prng_key: PRNGKeyArray,
  conditional_decoder: RunConditionalDecoderFn,
  model_parameters: ModelParameters,
  node_features: NodeFeatures,
  edge_features: EdgeFeatures,
  neighbor_indices: NeighborIndices,
  mask: AtomMask,
  num_steps: int,
  learning_rate: float,
) -> ProteinSequence:
  """Optimizes a sequence to maximize its conditional log-likelihood using STE."""
  num_residues, num_classes = node_features.shape[0], 21
  key, logit_key = jax.random.split(prng_key)
  latent_logits = jax.random.normal(logit_key, (num_residues, num_classes))

  optimizer = optax.adam(learning_rate)
  opt_state = optimizer.init(latent_logits)

  # Use a zeroed-out autoregressive mask for full-context scoring
  autoregressive_mask = jnp.zeros((num_residues, num_residues), dtype=jnp.int32)

  @jax.jit
  def update_step(carry, _):
    current_logits, current_opt_state = carry

    def loss_fn(logits_to_optimize: Logits):
      # Convert continuous logits to a discrete sequence via STE
      generated_sequence_one_hot = straight_through_estimator(logits_to_optimize)

      # Score the complete sequence with the conditional decoder
      scored_features = conditional_decoder(
        node_features,
        edge_features,
        neighbor_indices,
        mask,
        autoregressive_mask,
        generated_sequence_one_hot,
      )
      # Project features to get scoring logits
      scored_logits = final_projection(model_parameters, scored_features)

      # Calculate loss
      return _cross_entropy_loss(generated_sequence_one_hot, scored_logits, mask)

    loss_value, grads = jax.value_and_grad(loss_fn)(current_logits)
    updates, next_opt_state = optimizer.update(grads, current_opt_state)
    next_logits = optax.apply_updates(current_logits, updates)

    return (next_logits, next_opt_state), loss_value

  (final_logits, _), _ = jax.lax.scan(
    update_step,
    (latent_logits, opt_state),
    None,
    length=num_steps,
  )

  return final_logits.argmax(axis=-1).astype(jnp.int8)
