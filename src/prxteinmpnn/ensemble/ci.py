"""Conformational-inference from ProteinMPNN logits."""

from typing import Literal

import jax.numpy as jnp
from gmmx import GaussianMixtureModelJax
from jax.scipy.special import entr

from prxteinmpnn.ensemble.dbscan import (
  ConformationalStates,
  compute_component_distances,
  dbscan_cluster,
)
from prxteinmpnn.utils.entropy import posterior_mean_std
from prxteinmpnn.utils.types import EdgeFeatures, Logits, NodeFeatures, StructureAtomicCoordinates

ConformationalInferenceStrategy = Literal["logits", "node_features", "edge_features"]
"""Determines what features to use for conformational inference.

Options are "logits" (output conditional logits), "node_features" (decoded node features),
and "edge_features" (edge features from the encoder).
"""


def infer_states(
  gmm: GaussianMixtureModelJax,
  features: Logits | NodeFeatures | EdgeFeatures | StructureAtomicCoordinates,
  eps_std_scale: float = 1.0,
  min_cluster_weight: float = 0.01,
) -> ConformationalStates:
  """Infer residue or global states by clustering a GMM fit on input features.

  Args:
    gmm: Fitted GaussianMixtureModelJax object.
    features: Input features (logits or message), shape compatible with gmm.predict_proba.
    eps_std_scale: Scaling factor for DBSCAN epsilon.
    min_cluster_weight: Minimum cluster weight threshold.

  Returns:
    ResidueConformationalStates containing clustering results and statistics.

  """
  distance_matrix = compute_component_distances(gmm.means)
  component_weights = gmm.weights
  responsibility_matrix = jnp.squeeze(gmm.predict_proba(features), axis=(2, 3))
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

  return ConformationalStates(
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
  )
