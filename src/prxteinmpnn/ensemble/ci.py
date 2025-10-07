"""Conformational-inference from ProteinMPNN logits."""

from typing import Literal

import jax
import jax.numpy as jnp
from jax.scipy.special import entr

from prxteinmpnn.ensemble.dbscan import (
  ConformationalStates,
  GMMClusteringResult,
  compute_component_distances,
  dbscan_cluster,
)
from prxteinmpnn.ensemble.em_fit import GMM, Axis, log_likelihood
from prxteinmpnn.utils.entropy import posterior_mean_std
from prxteinmpnn.utils.types import EdgeFeatures, Logits, NodeFeatures

ConformationalInferenceStrategy = Literal["logits", "node_features", "edge_features"]
"""Determines what features to use for conformational inference.

Options are "logits" (output conditional logits), "node_features" (decoded node features),
and "edge_features" (edge features from the encoder).
"""


@jax.jit
def predict_probability(gmm: GMM, data: jax.Array) -> jax.Array:
  """Predict the probability of each sample belonging to each component.

  Args:
    gmm: Fitted GMM object.
    data: Input data, shape (num_samples, num_features).

  Returns:
  probabilities : jax.Array
      Predicted probabilities

  """
  log_prob = log_likelihood(data, gmm.means, gmm.covariances)
  log_prob_norm = jax.scipy.special.logsumexp(
    log_prob,
    axis=Axis.components,
    keepdims=True,
  )
  return jnp.exp(log_prob - log_prob_norm)


def infer_states(
  gmm: GMM,
  features: Logits | NodeFeatures | EdgeFeatures,
  eps_std_scale: float = 1.0,
  min_cluster_weight: float = 0.01,
) -> tuple[ConformationalStates, GMMClusteringResult, GMM]:
  """Infer residue or global states by clustering a GMM fit on input features.

  Args:
    gmm: Fitted GMM object.
    features: Input features (logits or message), shape compatible with predict_probability.
    eps_std_scale: Scaling factor for DBSCAN epsilon.
    min_cluster_weight: Minimum cluster weight threshold.

  Returns:
    ResidueConformationalStates containing clustering results and statistics.

  """
  distance_matrix = compute_component_distances(gmm.means)
  component_weights = gmm.weights
  responsibility_matrix = predict_probability(gmm, features)
  triu_indices = jnp.triu_indices_from(distance_matrix, k=1)
  eps = 1.0 - eps_std_scale * jnp.std(distance_matrix[triu_indices])

  cluster_result = dbscan_cluster(
    distance_matrix,
    component_weights,
    responsibility_matrix,
    eps,
    min_cluster_weight,
  )

  state_responsibilities = responsibility_matrix @ cluster_result.coarse_graining_matrix
  state_trajectory = jnp.argmax(state_responsibilities, axis=1)

  states, counts = jnp.unique(
    state_trajectory,
    size=gmm.n_components,
    fill_value=-1,
    return_counts=True,
  )
  n_states = jnp.sum(states != -1)
  mle_entropy = entr(counts[counts > 0] / counts.sum()).sum()
  _, mle_entropy_se = posterior_mean_std(counts[counts > 0].astype(jnp.float32))

  return (
    ConformationalStates(
      n_states=n_states,
      mle_entropy=mle_entropy,
      mle_entropy_se=mle_entropy_se,
      state_trajectory=state_trajectory,
      state_counts=counts,
      cluster_entropy=cluster_result.plug_in_entropy,
      cluster_probabilities=cluster_result.state_probabilities,
      dbscan_eps=eps,
      min_cluster_weight=min_cluster_weight,
      coarse_graining_matrix=cluster_result.coarse_graining_matrix,
    ),
    cluster_result,
    gmm,
  )
