"""Expectation-Maximization algorithm for Gaussian Mixture Models.

Adapted from gmmx (https://github.com/google/gmmx)
with modifications to support fitting from data generators, such as
HDF5 datasets.
"""

import logging
from enum import Enum
from functools import partial
from typing import Literal, NamedTuple

import jax
from flax.struct import dataclass
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Int

logger = logging.getLogger(__name__)

Data = Float[Array, "n_samples n_features"] | Float[Array, "n_batches n_samples n_features"]
DataMask = Bool[Array, "n_samples"]
Means = Float[Array, "n_components n_features"]
Covariances = Float[Array, "n_components n_features n_features"]
Weights = Float[Array, "n_components"]
Responsibilities = Float[Array, "n_samples n_components"]
Converged = Bool[Array, ""]
LogLikelihood = Float[Array, ""]
ComponentCounts = Int[Array, "n_components"]


@dataclass
class _EStepState:
  """State for accumulating statistics during the E-step."""

  component_counts: ComponentCounts
  weighted_data: Data
  weighted_squared_data: Data
  log_likelihood_total: LogLikelihood


@dataclass
class GMM:
  """Dataclass to hold GMM parameters."""

  means: Means
  covariances: Covariances
  weights: Weights
  responsibilities: Responsibilities
  n_components: int
  n_features: int


class Axis(int, Enum):
  """Internal axis order."""

  batch = 0
  components = 1
  features = 2
  features_covar = 3


def precisions(covariances: Covariances) -> Covariances:
  """Compute precision matrices."""
  cholesky_covariance = jax.scipy.linalg.cholesky(covariances, lower=True)

  identity = jnp.expand_dims(
    jnp.eye(covariances.shape[Axis.features]),
    axis=(Axis.batch, Axis.components),
  )
  b = jnp.repeat(identity, covariances.shape[Axis.components], axis=Axis.components)
  cholesky_precisions = jax.scipy.linalg.solve_triangular(cholesky_covariance, b, lower=True)
  return cholesky_precisions.mT


DIAG_NDIM, FULL_NDIM = 2, 3


def log_likelihood_diag(data: Data, means: Means, covariances: Covariances) -> LogLikelihood:
  """Compute log likelihood for diagonal covariance GMM."""
  _, n_features = data.shape
  diff = data[:, None, :] - means[None, :, :]
  log_det_cov = jnp.sum(jnp.log(covariances), axis=1)
  precision = 1.0 / covariances
  mahalanobis_dist = jnp.sum((diff**2) * precision[None, :, :], axis=2)
  return -0.5 * (n_features * jnp.log(2 * jnp.pi) + mahalanobis_dist + log_det_cov[None, :])


def log_likelihood(data: Data, means: Means, covariances: Covariances) -> LogLikelihood:
  """Compute log likelihood from the covariance for a given feature vector.

  Dispatches to the correct implementation based on covariance matrix shape.
  """
  if covariances.ndim == DIAG_NDIM:
    return log_likelihood_diag(data, means, covariances)
  if covariances.ndim == FULL_NDIM:
    cholesky_precisions = precisions(covariances)
    y = jnp.matmul(data.mT, cholesky_precisions) - jnp.matmul(
      means.mT,
      cholesky_precisions,
    )
    return jnp.sum(
      jnp.square(y),
      axis=(Axis.features, Axis.features_covar),
      keepdims=True,
    )

  msg = f"Unsupported covariance shape: {covariances.shape}"
  raise ValueError(msg)


class _EMLoopState(NamedTuple):
  """State for the in-memory EM loop."""

  gmm: GMM
  n_iter: int
  log_likelihood: LogLikelihood
  log_likelihood_diff: LogLikelihood


@dataclass
class EMFitterResult:
  """Result of the Expectation-Maximization fitting process.

  Attributes
  ----------
  gmm : GMM
      The final fitted Gaussian mixture model.
  n_iter : jax.Array
      The total number of iterations performed.
  log_likelihood : jax.Array
      The log-likelihood of the data under the final model.
  converged : jax.Array
      A boolean indicating if the algorithm converged within the max iterations.

  """

  gmm: GMM
  n_iter: Int
  log_likelihood: LogLikelihood
  log_likelihood_diff: LogLikelihood
  converged: Converged
  features: Data | None = None


def _e_step_diag(data: Data, gmm: "GMM") -> tuple[Means, Responsibilities]:
  """E-step for diagonal covariance GMM."""
  _, n_features = data.shape
  diff = data[:, None, :] - gmm.means[None, :, :]
  log_det_cov = jnp.sum(jnp.log(gmm.covariances), axis=1)
  precision = 1.0 / gmm.covariances
  mahalanobis_dist = jnp.sum((diff**2) * precision[None, :, :], axis=2)
  log_prob = -0.5 * (n_features * jnp.log(2 * jnp.pi) + mahalanobis_dist + log_det_cov[None, :])
  weighted_log_prob = log_prob + jnp.log(gmm.weights)
  log_prob_norm = jax.scipy.special.logsumexp(weighted_log_prob, axis=1)
  log_resp = weighted_log_prob - log_prob_norm[:, None]
  mean_log_prob_norm = jnp.mean(log_prob_norm)
  return mean_log_prob_norm, log_resp


def _e_step_full(data: Data, gmm: "GMM") -> tuple[Means, Responsibilities]:
  """E-step for full covariance GMM."""
  log_prob = log_likelihood(data, gmm.means, gmm.covariances)
  log_prob_norm = jax.scipy.special.logsumexp(
    log_prob,
    axis=Axis.components,
    keepdims=True,
  )
  log_resp = log_prob - log_prob_norm
  return jnp.mean(log_prob_norm), log_resp


def _e_step(
  data: Float,
  gmm: "GMM",
  covariance_type: str,
) -> tuple[Means, Responsibilities]:
  """Dispatcher for the E-step based on covariance type."""
  if covariance_type == "diag":
    return _e_step_diag(data, gmm)
  if covariance_type == "full":
    return _e_step_full(data, gmm)
  msg = f"Unknown covariance type: {covariance_type}"
  raise ValueError(msg)


@partial(jax.jit, static_argnames=("covariance_regularization", "covariance_type"))
def _m_step_from_responsibilities(
  data: Data,
  means: Means,
  covariances: Covariances,
  responsibilities: Responsibilities,
  covariance_regularization: float,
  covariance_type: Literal["full", "diag"] = "full",
) -> tuple[Weights, Means, Covariances]:
  """Maximization (M) step for in-memory data using responsibilities."""
  component_counts = jnp.sum(responsibilities, axis=Axis.batch)

  safe_component_counts = jnp.where(component_counts == 0, 1.0, component_counts)

  updated_means = jnp.einsum("ij,ik->jk", responsibilities, data) / safe_component_counts[..., None]
  means = jnp.asarray(jnp.where(component_counts[..., None] > 0, updated_means, means))

  if covariance_type == "full":
    updated_covariances = (
      jnp.einsum("ij,ik,il->jkl", responsibilities, data, data)
      / safe_component_counts[..., None, None]
    )
    updated_covariances = updated_covariances - jnp.einsum("...i,...j->...ij", means, means)
    updated_covariances += covariance_regularization * jnp.eye(means.shape[-1])

    covariances_final = jnp.where(
      component_counts[..., None, None] > 0,
      updated_covariances,
      covariances,
    )
  elif covariance_type == "diag":
    updated_vars = (
      jnp.einsum("ij,ik->jk", responsibilities, data**2) / safe_component_counts[..., None]
      - means**2
    )
    updated_vars += covariance_regularization
    covariances_final = jnp.where(component_counts[..., None] > 0, updated_vars, covariances)

  weights = component_counts / data.shape[Axis.batch]

  return weights, means, covariances_final


@partial(
  jax.jit,
  static_argnames=("n_total_samples", "covariance_regularization", "covariance_type"),
)
def _m_step_from_stats(
  gmm: GMM,
  component_counts: ComponentCounts,
  weighted_data: Data,
  weighted_squared_data: Data,
  n_total_samples: int,
  covariance_regularization: float,
  covariance_type: Literal["full", "diag"],
) -> GMM:
  """Update GMM parameters from accumulated sufficient statistics for batch processing."""
  if n_total_samples == 0:
    return GMM(
      means=gmm.means,
      covariances=gmm.covariances,
      weights=jnp.zeros_like(component_counts),
      responsibilities=jnp.zeros_like(component_counts),
      n_components=gmm.n_components,
      n_features=gmm.n_features,
    )

  weights = component_counts / n_total_samples

  # Add safeguard for components with no assigned data points
  safe_component_counts = jnp.where(component_counts == 0, 1.0, component_counts)

  updated_means = weighted_data / safe_component_counts[..., None]
  means = jnp.array(
    jnp.where(component_counts[..., None] > 0, updated_means, gmm.means),
  )
  if covariance_type == "full":
    updated_covariances = weighted_squared_data / safe_component_counts[..., None, None]
    updated_covariances = updated_covariances - jnp.einsum("...i,...j->...ij", means, means)
    updated_covariances += covariance_regularization * jnp.eye(means.shape[-1])

    covariances_final = jnp.where(
      component_counts[..., None, None] > 0,
      updated_covariances,
      gmm.covariances,
    )
  elif covariance_type == "diag":
    updated_vars = weighted_squared_data / safe_component_counts[..., None] - means**2
    updated_vars += covariance_regularization

    covariances_final = jnp.where(component_counts[..., None] > 0, updated_vars, gmm.covariances)

  return GMM(
    means=means,
    covariances=covariances_final,
    weights=weights,
    responsibilities=gmm.responsibilities,
    n_components=gmm.n_components,
    n_features=gmm.n_features,
  )


def fit_gmm_states(
  data: Data,
  gmm: GMM,
  covariance_type: Literal["full", "diag"] = "full",
  max_iter: int = 100,
  min_iter: int = 10,
  tol: float = 1e-3,
  covariance_regularization: float = 1e-3,
) -> EMFitterResult:
  """Fit a GMM to in-memory data using the EM algorithm."""

  @jax.jit
  def em_step_fn(state: _EMLoopState) -> _EMLoopState:
    """Run a single EM step."""
    log_likelihood, log_resp = _e_step(data, state.gmm, covariance_type)
    weights, means, covariances = _m_step_from_responsibilities(
      data,
      state.gmm.means,
      state.gmm.covariances,
      jnp.exp(log_resp),
      covariance_regularization,
      covariance_type,
    )
    return _EMLoopState(
      gmm=GMM(
        means=means,
        covariances=covariances,
        weights=weights,
        responsibilities=log_resp,
        n_components=state.gmm.n_components,
        n_features=state.gmm.n_features,
      ),
      n_iter=state.n_iter + 1,
      log_likelihood=log_likelihood,
      log_likelihood_diff=jnp.abs(log_likelihood - state.log_likelihood),
    )

  def em_cond_fn(state: _EMLoopState) -> Converged:
    """Check if EM should continue iterating."""
    result = state
    jax.debug.print(
      "EM iteration {}/{}: log-likelihood = {}, diff = {}",
      result.n_iter,
      max_iter,
      result.log_likelihood,
      result.log_likelihood_diff,
    )
    has_converged = jnp.abs(result.log_likelihood_diff) < tol
    keep_going = (~has_converged) | (result.n_iter < min_iter)
    return (result.n_iter < max_iter) & keep_going

  initial_state = _EMLoopState(
    gmm=gmm,
    n_iter=0,
    log_likelihood=jnp.asarray(-jnp.inf),
    log_likelihood_diff=jnp.asarray(jnp.inf),
  )

  final_state = jax.lax.while_loop(em_cond_fn, em_step_fn, initial_state)
  final_state = jax.block_until_ready(final_state)

  return EMFitterResult(
    gmm=final_state.gmm,
    n_iter=final_state.n_iter,
    log_likelihood=final_state.log_likelihood,
    log_likelihood_diff=final_state.log_likelihood_diff,
    converged=final_state.log_likelihood_diff < tol,
    features=data,
  )
