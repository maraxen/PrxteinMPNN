"""Vectorized DBSCAN for inferring conformational states from GMM-fitted logits.

This module implements a JAX-based, vectorized version of the DBSCAN algorithm
to cluster components of a Gaussian Mixture Model (GMM). The GMM is
fit on residue-level logits from a ProteinMPNN, with each GMM component
representing a potential conformational microstate. DBSCAN is then used to
coarse-grain these GMM components into a smaller set of meaningful conformational
states.

The core of the algorithm, `perform_dbscan_clustering`, is adapted from the
matrix-based approach in github.com/justktln2/ciMIST.
"""

from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from flax import struct
from gmmx import GaussianMixtureModelJax
from jax import Array, vmap
from jax.scipy.special import entr
from jaxtyping import ArrayLike, Float

from prxteinmpnn.utils.entropy import posterior_mean_std, von_neumman
from prxteinmpnn.utils.types import Logits


@struct.dataclass
class ConformationalStates:
  """Represents the final inferred conformational states for a residue or protein.

  Attributes:
    n_states: The total number of conformational states found (excluding noise).
    mle_entropy: "Plug-in" entropy calculated from the hard-assigned state counts.
    mle_entropy_se: Standard error of the plug-in entropy estimate.
    state_trajectory: An array showing the assigned state for each frame in the
      input trajectory.
    state_counts: The number of times each state is observed in the trajectory.
    cluster_entropy: Entropy calculated from the soft probabilities (weights) of
      the final clusters.
    cluster_probabilities: The soft probability (total weight) of each cluster.
    dbscan_eps: The `eps` radius parameter used for DBSCAN.
    min_cluster_weight: The minimum probability mass used for a point to be
      considered a core point in DBSCAN.
    coarse_graining_matrix: The operator matrix that maps GMM components to
      final states.

  """

  n_states: ArrayLike
  mle_entropy: ArrayLike
  mle_entropy_se: ArrayLike
  state_trajectory: Array
  state_counts: Array
  cluster_entropy: ArrayLike
  cluster_probabilities: ArrayLike
  dbscan_eps: ArrayLike
  min_cluster_weight: float
  coarse_graining_matrix: Array | None = None


@struct.dataclass
class GMMClusteringResult:
  """Represents the direct output of clustering GMM components with DBSCAN.

  Attributes:
    coarse_graining_matrix: Operator that maps original GMM components to
      final states (clusters).
    core_component_connectivity: Adjacency matrix for connections between
      core GMM components.
    non_noise_connectivity: Connectivity matrix including both core and
      border components.
    state_probabilities: The probability (summed weight) of each final state.
    plug_in_entropy: The "plug-in" entropy estimated from state_probabilities.
    von_neumann_entropy: The von Neumann entropy of the state density matrix.
    posterior_mean_entropy: Posterior mean of the entropy.
    posterior_entropy_std_err: Posterior standard error of the entropy.
    dbscan_eps: The `eps` parameter used for DBSCAN.
    min_cluster_weight: The minimum weight for a component to be a core point.
    state_density_matrix: The coarse-grained density overlap matrix between states.

  """

  coarse_graining_matrix: Array
  core_component_connectivity: Array
  non_noise_connectivity: Array
  state_probabilities: Array
  plug_in_entropy: ArrayLike
  von_neumann_entropy: ArrayLike
  posterior_mean_entropy: ArrayLike
  posterior_entropy_std_err: ArrayLike
  dbscan_eps: ArrayLike
  min_cluster_weight: ArrayLike
  state_density_matrix: Array


@struct.dataclass
class EntropyTrace:
  """Stores entropy metrics calculated over a range of DBSCAN `eps` values."""

  plug_in_entropy: Array
  von_neumann_entropy: Array
  posterior_mean_entropy: Array
  posterior_entropy_std_err: Array
  z_score_sq: Array
  eps_values: Array


@partial(jax.jit, static_argnames=("distance_metric",))
def compute_component_distances(
  means: Float,
  distance_metric: Literal["euclidean", "cosine"] = "euclidean",
) -> Float:
  """Compute the pairwise Euclidean distance between GMM component means.

  This distance metric treats each GMM component's mean vector as a point in
  logit space and calculates the distances between them.

  ||a - b||^2 = ||a||^2 - 2a.b + ||b||^2

  Clip at the end to avoid negative numbers.

  Args:
    means: The mean vectors of the GMM components, with shape
      (n_components, n_features).
    distance_metric: The distance metric to use, either "euclidean" or "cosine".

  Returns:
    A symmetric distance matrix of shape (n_components, n_components).

  """
  if distance_metric == "cosine":
    norms = jnp.linalg.norm(means, axis=1, keepdims=True)
    normalized_means = means / norms
    cosine_similarity = jnp.dot(normalized_means, normalized_means.T)
    return 1.0 - cosine_similarity
  squared_norms = jnp.sum(means**2, axis=1)
  squared_distances = squared_norms[:, None] - 2 * jnp.dot(means, means.T) + squared_norms[None, :]
  return jnp.sqrt(jnp.maximum(squared_distances, 0))


def _get_neighborhood(distance_matrix: Array, eps: ArrayLike) -> Array:
  """Compute a binary matrix indicating which components are within the ε-neighborhood.

    distance_matrix (Array): A matrix of pairwise distances between components.
    eps (ArrayLike): The ε threshold(s) for neighborhood inclusion. Can be a scalar or array.

  Returns:
    Array: A binary matrix of the same shape as `distance_matrix`, where each entry is 1 if the
      corresponding distance is less than or equal to `eps`, and 0 otherwise.

  Notes:
    This function uses a Heaviside step function to determine neighborhood membership.

  """
  return jnp.heaviside(eps - distance_matrix, 0)


@partial(jax.jit, static_argnames=("connectivity_method",))
def dbscan_cluster(
  distance_matrix: Array,
  component_weights: Array,
  responsibility_matrix: Array,
  eps: ArrayLike,
  min_cluster_weight: ArrayLike,
  connectivity_method: Literal["expm", "power"] = "expm",
) -> GMMClusteringResult:
  """Perform vectorized DBSCAN on GMM components.

  Algorithm process.
    1. Identify core components (_get_neighborhood).
      Sum weights of all neighbors for each component
        is_in_neighborhood @ component_weights
    2. Identify connected components among core points.
      Adjacency matrix for core components: 1 if two core points are neighbors
        core_point_pairs * is_in_neighborhood
      Use matrix exponentiation or power to find all reachable nodes
    3. Assign border components to the cluster of their nearest core component
    4. Construct the final coarse-graining matrix
    5. Calculate properties of the new coarse-grained states

  Args:
    distance_matrix: A precomputed (n_components, n_components) distance matrix.
    component_weights: The weight of each GMM component (from gmm.weights_).
    responsibility_matrix: The probability of each data point belonging to each
      GMM component. Shape (n_observations, n_components).
    eps: The epsilon radius for DBSCAN neighborhood search.
    min_cluster_weight: The minimum total weight for a point to be a core point.
    connectivity_method: Method to find connected components ('expm' or 'power').
      'expm' (matrix exponential) is generally more stable and preferred.

  Returns:
      A GMMClusteringResult object containing the detailed results of the clustering.

  """
  is_in_neighborhood = _get_neighborhood(distance_matrix, eps)
  neighborhood_weights = is_in_neighborhood @ component_weights
  is_core_component = jnp.heaviside(neighborhood_weights - min_cluster_weight, 1)
  core_point_pairs = jnp.outer(is_core_component, is_core_component)
  core_adjacency_matrix = core_point_pairs * is_in_neighborhood
  core_adjacency_matrix = jnp.fill_diagonal(core_adjacency_matrix, 0, inplace=False)
  if connectivity_method == "power":
    # Use a fixed maximum power to avoid JAX tracing issues
    max_power = distance_matrix.shape[0] - 1
    power_result = core_adjacency_matrix
    # Iteratively compute matrix powers up to max_power
    for _ in range(max_power):
      power_result = power_result + (power_result @ core_adjacency_matrix)
    core_connectivity_matrix = jnp.heaviside(power_result, 0)
  elif connectivity_method == "expm":
    core_connectivity_matrix = jnp.heaviside(
      jax.scipy.linalg.expm(core_adjacency_matrix, max_squarings=48),
      0,
    )

  is_border_component = 1 - is_core_component
  border_core_link_mask = jnp.outer(is_core_component, is_border_component)
  border_core_adjacency = border_core_link_mask * is_in_neighborhood
  min_dist_to_core = jnp.min(
    border_core_adjacency * distance_matrix + (1 - border_core_adjacency) * jnp.inf,
    axis=0,
  )
  border_component_assignments = (border_core_adjacency * distance_matrix) == min_dist_to_core

  non_noise_connectivity_matrix = (
    core_connectivity_matrix * is_core_component
    + core_connectivity_matrix @ (border_component_assignments * is_border_component)
  )

  is_noise_component = 1 - jnp.heaviside(non_noise_connectivity_matrix.sum(axis=1), 0)
  coarse_graining_matrix = non_noise_connectivity_matrix + jnp.diag(is_noise_component)

  coarse_graining_matrix = jnp.unique(
    coarse_graining_matrix,
    axis=0,
    size=coarse_graining_matrix.shape[0],
    fill_value=0,
  )

  state_responsibilities = responsibility_matrix @ coarse_graining_matrix.T
  state_density_matrix = jnp.sqrt(state_responsibilities).T @ jnp.sqrt(state_responsibilities)
  state_density_matrix /= jnp.trace(state_density_matrix)

  state_probabilities = state_responsibilities.mean(axis=0)

  is_noise_cluster = coarse_graining_matrix @ is_noise_component
  sort_order = jnp.argsort(state_probabilities * (1 - 2 * is_noise_cluster))
  coarse_graining_matrix = coarse_graining_matrix[sort_order, :]
  state_probabilities = state_probabilities[sort_order]

  plug_in_entropy = entr(state_probabilities).sum()
  von_neumann_entropy = von_neumman(state_density_matrix)
  posterior_mean, posterior_std = posterior_mean_std(
    state_probabilities * responsibility_matrix.shape[0],
  )

  return GMMClusteringResult(
    coarse_graining_matrix=coarse_graining_matrix,
    core_component_connectivity=core_connectivity_matrix,
    non_noise_connectivity=non_noise_connectivity_matrix,
    state_probabilities=state_probabilities,
    plug_in_entropy=plug_in_entropy,
    von_neumann_entropy=von_neumann_entropy,
    posterior_mean_entropy=posterior_mean,
    posterior_entropy_std_err=posterior_std,
    dbscan_eps=eps,
    min_cluster_weight=min_cluster_weight,
    state_density_matrix=state_density_matrix,
  )


def trace_entropy_across_eps(
  gmm: GaussianMixtureModelJax,
  logits: Logits,
  eps_values: Array | None = None,
  min_cluster_weight: float = 0.01,
  vmap_chunk_size: int | None = None,
  distance_metric: Literal["euclidean", "cosine"] = "euclidean",
) -> EntropyTrace:
  """Calculate entropy metrics across a range of `eps` values for analysis.

  Args:
    gmm: A fitted `GaussianMixtureModelJax` object.
    logits: The original trajectory data.
    eps_values: An array of `eps` values to test. If None, a default range
      from 0.01 to 0.99 is used.
    min_cluster_weight: The minimum weight for a component to be a core point.
    vmap_chunk_size: If specified, breaks the calculation into chunks to
      manage memory usage with very large `eps_values` arrays.
    distance_metric: The distance metric to use for computing component distances.

  Returns:
    An `EntropyTrace` object containing the calculated metrics for each `eps`.

  """
  if eps_values is None:
    eps_values = jnp.linspace(0.01, 0.99, 99)

  distance_matrix = compute_component_distances(gmm.means_, distance_metric=distance_metric)
  component_weights = gmm.weights_
  responsibility_matrix = gmm.predict_proba(logits)

  def _calculate_for_single_eps(eps: Float) -> tuple:
    result = dbscan_cluster(
      distance_matrix,
      component_weights,
      responsibility_matrix,
      eps,
      min_cluster_weight,
    )
    z_score_sq = (
      (result.von_neumann_entropy - result.plug_in_entropy)
      / (result.posterior_entropy_std_err + 1e-9)
    ) ** 2
    return (
      result.plug_in_entropy,
      result.von_neumann_entropy,
      result.posterior_mean_entropy,
      result.posterior_entropy_std_err,
      z_score_sq,
    )

  if vmap_chunk_size and len(eps_values) > vmap_chunk_size:
    s_plug_in, s_vn, s_post_mean, s_post_se, z_sq = jax.lax.map(
      _calculate_for_single_eps,
      eps_values,
      batch_size=vmap_chunk_size,
    )
  else:
    s_plug_in, s_vn, s_post_mean, s_post_se, z_sq = vmap(_calculate_for_single_eps)(eps_values)

  return EntropyTrace(
    plug_in_entropy=s_plug_in.flatten(),
    von_neumann_entropy=s_vn.flatten(),
    posterior_mean_entropy=s_post_mean.flatten(),
    posterior_entropy_std_err=s_post_se.flatten(),
    z_score_sq=z_sq.flatten(),
    eps_values=eps_values,
  )
