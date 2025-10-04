"""Expectation-Maximization algorithm for Gaussian Mixture Models.

Adapted from gmmx (https://github.com/google/gmmx)
with modifications to support fitting from data generators, such as
HDF5 datasets.
"""

import logging
from collections.abc import Generator
from enum import Enum
from functools import partial
from typing import Literal, NamedTuple

import jax
from flax.struct import dataclass
from jax import numpy as jnp

logger = logging.getLogger(__name__)


@dataclass
class GMM:
  """Dataclass to hold GMM parameters."""

  means: jax.Array
  covariances: jax.Array
  weights: jax.Array
  responsibilities: jax.Array
  n_components: int
  n_features: int


class Axis(int, Enum):
  """Internal axis order."""

  batch = 0
  components = 1
  features = 2
  features_covar = 3


def precisions(covariances: jax.Array) -> jax.Array:
  """Compute precision matrices."""
  cholesky_covariance = jax.scipy.linalg.cholesky(covariances, lower=True)

  identity = jnp.expand_dims(
    jnp.eye(covariances.shape[Axis.features]),
    axis=(Axis.batch, Axis.components),
  )
  b = jnp.repeat(identity, covariances.shape[Axis.components], axis=Axis.components)
  cholesky_precisions = jax.scipy.linalg.solve_triangular(cholesky_covariance, b, lower=True)
  return cholesky_precisions.mT


def log_likelihood(data: jax.Array, means: jax.Array, covariances: jax.Array) -> jax.Array:
  """Compute log likelihood from the covariance for a given feature vector.

  Parameters
  ----------
  data : jax.array
      Feature vectors
  means : jax.array
      Means of the components
  covariances : jax.array
      Covariances of the components

  Returns
  -------
  log_prob : jax.array
      Log likelihood

  """
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


class _EMLoopState(NamedTuple):
  """State for the in-memory EM loop."""

  gmm: GMM
  n_iter: int
  log_likelihood: jax.Array
  log_likelihood_diff: jax.Array


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
  n_iter: jax.Array
  log_likelihood: jax.Array
  log_likelihood_diff: jax.Array
  converged: jax.Array


def _e_step_diag(data: jax.Array, gmm: "GMM") -> tuple[jax.Array, jax.Array]:
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


def _e_step_full(data: jax.Array, gmm: "GMM") -> tuple[jax.Array, jax.Array]:
  """E-step for full covariance GMM."""
  log_prob = log_likelihood(data, gmm.means, gmm.covariances)
  log_prob_norm = jax.scipy.special.logsumexp(
    log_prob,
    axis=Axis.components,
    keepdims=True,
  )
  log_resp = log_prob - log_prob_norm
  return jnp.mean(log_prob_norm), log_resp


# --- Modified _e_step to act as a dispatcher ---
def _e_step(
  data: jax.Array,
  gmm: "GMM",
  covariance_type: str,
) -> tuple[jax.Array, jax.Array]:
  """Dispatcher for the E-step based on covariance type."""
  if covariance_type == "diag":
    return _e_step_diag(data, gmm)
  if covariance_type == "full":
    return _e_step_full(data, gmm)
  msg = f"Unknown covariance type: {covariance_type}"
  raise ValueError(msg)


@partial(jax.jit, static_argnames=("covariance_regularization", "covariance_type"))
def _m_step_from_responsibilities(
  data: jax.Array,
  means: jax.Array,
  covariances: jax.Array,
  responsibilities: jax.Array,
  covariance_regularization: float,
  covariance_type: Literal["full", "diag"] = "full",
) -> tuple[jax.Array, jax.Array, jax.Array]:
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

    original_covariances_squeezed = jnp.squeeze(covariances)
    covariances_3d = jnp.where(
      component_counts[..., None, None] > 0,
      updated_covariances,
      original_covariances_squeezed,
    )
    covariances_final = covariances_3d[None, ...]
  elif covariance_type == "diag":
    updated_vars = (
      jnp.einsum("ij,ik->jk", responsibilities, data**2) / safe_component_counts[..., None]
      - means**2
    )
    updated_vars += covariance_regularization

    original_vars_squeezed = jnp.squeeze(covariances)
    variances_2d = jnp.where(component_counts[..., None] > 0, updated_vars, original_vars_squeezed)
    covariances_final = variances_2d[None, ..., None]

  weights = component_counts / data.shape[Axis.batch]

  return weights, means, covariances_final


@partial(
  jax.jit,
  static_argnames=("n_total_samples", "covariance_regularization", "covariance_type"),
)
def _m_step_from_stats(
  gmm: GMM,
  component_counts: jax.Array,
  weighted_data: jax.Array,
  weighted_squared_data: jax.Array,
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

    original_covariances_squeezed = jnp.squeeze(gmm.covariances)
    covariances_3d = jnp.where(
      component_counts[..., None, None] > 0,
      updated_covariances,
      original_covariances_squeezed,
    )
    covariances_final = covariances_3d[None, ...]
  elif covariance_type == "diag":
    updated_vars = weighted_squared_data / safe_component_counts[..., None] - means**2
    updated_vars += covariance_regularization

    original_vars_squeezed = jnp.squeeze(gmm.covariances)
    variances_2d = jnp.where(component_counts[..., None] > 0, updated_vars, original_vars_squeezed)
    covariances_final = variances_2d[None, ..., None]

  return GMM(
    means=means,
    covariances=covariances_final,
    weights=weights,
    responsibilities=gmm.responsibilities,
    n_components=gmm.n_components,
    n_features=gmm.n_features,
  )


def fit_gmm_in_memory(
  data: jax.Array,
  gmm: GMM,
  covariance_type: Literal["full", "diag"] = "full",
  max_iter: int = 100,
  tol: float = 1e-3,
  covariance_regularization: float = 1e-6,
) -> EMFitterResult:
  """Fit a GMM to in-memory data using the EM algorithm."""

  @jax.jit
  def em_step_fn(state: _EMLoopState) -> _EMLoopState:
    """Run a single EM step."""
    log_likelihood, log_resp = _e_step(data, state.gmm, covariance_type)
    means, covariances, weights = _m_step_from_responsibilities(
      data,
      state.gmm.means,
      state.gmm.covariances,
      state.gmm.weights,
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

  def em_cond_fn(state: _EMLoopState) -> jax.Array:
    """Stop condition for the EM loop."""
    return (state.n_iter < max_iter) & (state.log_likelihood_diff >= tol)

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
  )


def fit_gmm_generator(
  data_generator: Generator[jax.Array, None, None],
  gmm: GMM,
  n_total_samples: int,
  max_iter: int = 100,
  tol: float = 1e-3,
  covariance_regularization: float = 1e-6,
  covariance_type: str = "full",
) -> EMFitterResult:
  """Fit a GMM to data from a generator (for large, out-of-memory datasets)."""
  log_likelihood = jnp.asarray(-jnp.inf)

  data = list(data_generator)

  if not data or n_total_samples == 0:
    return EMFitterResult(
      gmm=gmm,
      n_iter=jnp.asarray(0),
      log_likelihood=jnp.asarray(-jnp.inf),
      log_likelihood_diff=jnp.asarray(jnp.inf),
      converged=jnp.asarray([False]),
    )

  component_counts_init = jnp.zeros(gmm.n_components)
  weighted_data_init = jnp.zeros((gmm.n_components, gmm.n_features))
  if covariance_type == "full":
    weighted_squared_data_init = jnp.zeros((gmm.n_components, gmm.n_features, gmm.n_features))
  elif covariance_type == "diag":
    weighted_squared_data_init = jnp.zeros((gmm.n_components, gmm.n_features))
  else:
    msg = f"Unsupported covariance type: {covariance_type}"
    raise ValueError(msg)
  log_likelihood_total_init = jnp.asarray(0.0)

  def body_fn(
    i: jax.Array,
    val: tuple[EMFitterResult, jax.Array],
  ) -> tuple[EMFitterResult, jax.Array]:
    n_iter = i + 1

    def accum_step(
      state: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
      data_batch: jax.Array,
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array], None]:
      component_counts, weighted_data, weighted_squared_data, log_likelihood_total = state
      batch_ll, log_resp = _e_step(data_batch, gmm, covariance_type)
      resp = jnp.exp(log_resp)

      log_likelihood_total += batch_ll * data_batch.shape[0]
      component_counts += jnp.sum(resp, axis=Axis.batch)
      weighted_data += jnp.einsum("ij,ik->jk", resp, data_batch)
      if covariance_type == "full":
        weighted_squared_data += jnp.einsum("ij,ik,il->jkl", resp, data_batch, data_batch)
      elif covariance_type == "diag":
        weighted_squared_data += jnp.einsum("ij,ik->jk", resp, data_batch**2)
      return (component_counts, weighted_data, weighted_squared_data, log_likelihood_total), None

    (component_counts, weighted_data, weighted_squared_data, log_likelihood_total), _ = (
      jax.lax.scan(
        accum_step,
        (
          component_counts_init,
          weighted_data_init,
          weighted_squared_data_init,
          log_likelihood_total_init,
        ),
        jnp.array(data),
      )
    )
    updated_gmm = _m_step_from_stats(
      previous_result=val[0].gmm,
      component_counts=component_counts,
      weighted_data_sum=weighted_data,
      weighted_squared_data_sum=weighted_squared_data,
      n_total_samples=n_total_samples,
      covariance_regularization=covariance_regularization,
      covariance_type=covariance_type,
    )
    log_likelihood = jnp.asarray(log_likelihood_total / n_total_samples)
    log_likelihood_diff = jnp.abs(log_likelihood - val[1])

    return EMFitterResult(
      gmm=updated_gmm,
      n_iter=n_iter,
      log_likelihood=log_likelihood,
      log_likelihood_diff=log_likelihood_diff,
      converged=log_likelihood_diff < tol,
    ), log_likelihood

  def converged(state: tuple[EMFitterResult, jax.Array]) -> jax.Array:
    return (state[0].n_iter < max_iter) & (state[0].log_likelihood_diff >= tol)

  def dispatch(
    i: jax.Array,
    state: tuple[EMFitterResult, jax.Array],
  ) -> tuple[EMFitterResult, jax.Array]:
    return jax.lax.cond(
      converged(state),
      body_fn(i, state),
      state,
      operand=None,
    )

  return jax.lax.fori_loop(
    0,
    max_iter,
    dispatch,
    (
      EMFitterResult(
        gmm=gmm,
        n_iter=jnp.asarray(0),
        log_likelihood=jnp.asarray(-jnp.inf),
        log_likelihood_diff=jnp.asarray(jnp.inf),
        converged=jnp.asarray([False]),
      ),
      log_likelihood,
    ),
  )
