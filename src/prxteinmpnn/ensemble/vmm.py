"""Utilities for von Mises mixture models."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np  # Used for loading `.npz` files
from flax.struct import dataclass, field
from jax import Array, jit, vmap
from jax.lax import cond, scan
from jax.nn import softmax
from jax.scipy.optimize import OptimizeResults, minimize
from jax.scipy.special import i0, logsumexp
from jaxtyping import Float, PRNGKeyArray


@dataclass
class MixtureFitState:
  """State for the von Mises mixture model fitting process.

  This dataclass represents the carry of the `jax.lax.scan` loop during the EM
  fitting process. It holds all the necessary parameters and metadata that
  change over iterations.

  Attributes:
      mu: Mean direction of each mixture component. Shape (n_components, n_features).
      kappa: Concentration parameter of each mixture component. Shape (n_components, n_features).
      log_weights: Log weights of the mixture components. Shape (n_components,).
      log_likelihood: Log likelihood of the model over iterations. Shape (max_iter,).
      statuses: Convergence statuses from the optimizer for each M-step. Shape (max_iter,).
      n_jacobian_evals: Number of Jacobian evaluations from the optimizer. Shape (max_iter,).
      n_func_evals: Number of function evaluations from the optimizer. Shape (max_iter,).
      n_iterations: Number of iterations from the optimizer. Shape (max_iter,).
      successes: Success statuses from the optimizer. Shape (max_iter,).

  """

  mu: Array = field(metadata={"help": "Mean direction of the mixture component."})
  kappa: Array = field(metadata={"help": "Concentration parameter of the mixture component."})
  log_weights: Array = field(metadata={"help": "Log weights of the mixture components."})
  log_likelihood: Array = field(metadata={"help": "Log likelihood of the mixture model."})
  statuses: Array = field(metadata={"help": "Convergence statuses of each component."})
  n_jacobian_evals: Array = field(metadata={"help": "Number of Jacobian evaluations."})
  n_func_evals: Array = field(metadata={"help": "Number of function evaluations."})
  n_iterations: Array = field(metadata={"help": "Number of iterations."})
  successes: Array = field(metadata={"help": "Success statuses of each component."})


def sort_by_weight(
  state: MixtureFitState,
  responsibilities: Array,
) -> tuple[MixtureFitState, Array]:
  """Sort the mixture components by their weights in descending order.

  Args:
      state: The current state of the mixture model.
      responsibilities: Responsibilities of each mixture component.

  Returns:
      A tuple containing the new state and the sorted responsibilities.

  """
  ix = jnp.argsort(-state.log_weights)
  new_state = state.replace(  # type: ignore[attr-access]
    mu=state.mu[ix],
    kappa=state.kappa[ix],
    log_weights=state.log_weights[ix],
  )
  sorted_responsibilities = responsibilities[:, ix]
  return new_state, sorted_responsibilities


def von_mises_mixture_log_pdf(
  x: Array,
  mu: Array,
  kappa: Array,
  log_weights: Array,
  mask: Array,
) -> Array:
  """Calculate the log-pdf of the von Mises-Fisher mixture distribution.

  Args:
      x: Data points. Shape (n_data, n_features).
      mu: Mean direction of each component. Shape (n_components, n_features).
      kappa: Concentration parameter of each component. Shape (n_components, n_features).
      log_weights: Log weights of the mixture components. Shape (n_components,).
      mask: Mask indicating valid features. Shape (n_features,).

  Returns:
      The log-pdf for each data point. Shape (n_data,).

  """
  component_log_probs = vmap(
    lambda m, k, lw: von_mises_component_log_prob(x, m, k, lw, mask),
    in_axes=(0, 0, 0),
  )
  return logsumexp(component_log_probs(mu, kappa, log_weights), axis=0)


def von_mises_component_log_prob(
  x: Array,
  mu: Array,
  kappa: Array,
  logw: Array,
  mask: Array,
) -> Array:
  """Calculate the log-probability of an individual mixture component.

  This includes the component's weight.

  Args:
      x: Data points. Shape (n_data, n_features).
      mu: Mean direction of the component. Shape (n_features,).
      kappa: Concentration parameter of the component. Shape (n_features,).
      logw: Log weight of the component.
      mask: Mask indicating valid features. Shape (n_features,).

  Returns:
      The log-probability for each data point. Shape (n_data,).

  """
  return von_mises_log_pdf(x, mu, kappa, mask) + logw


def von_mises_log_pdf(x: Array, mu: Array, kappa: Array, mask: Array) -> Array:
  """Calculate the log-pdf of a single von Mises distribution.

  Args:
      x: Data points. Shape (n_data, n_features).
      mu: Mean direction of the component. Shape (n_features,).
      kappa: Concentration parameter of the component. Shape (n_features,).
      mask: Mask indicating valid features. Shape (n_features,).

  Returns:
      The log-pdf for each data point. Shape (n_data,).

  """
  summands = jnp.where(mask, kappa * jnp.cos(x - mu) - jnp.log(2 * jnp.pi * i0(kappa)), 0.0)
  return jnp.sum(summands, axis=-1)


def e_step(x: Array, mu: Array, kappa: Array, log_weights: Array, mask: Array) -> Array:
  """Perform the E-step of the EM algorithm.

  Calculates the responsibilities of each component for each data point.

  Args:
      x: Data points. Shape (n_data, n_features).
      mu: Mean direction of each component. Shape (n_components, n_features).
      kappa: Concentration parameter of each component. Shape (n_components, n_features).
      log_weights: Log weights of the mixture components. Shape (n_components,).
      mask: Mask indicating valid features. Shape (n_features,).

  Returns:
      The responsibilities array. Shape (n_data, n_components).

  """
  component_log_likelihoods = vmap(
    lambda m, k: von_mises_log_pdf(x, m, k, mask),
    in_axes=(0, 0),
    out_axes=1,
  )
  return softmax(component_log_likelihoods(mu, kappa) + log_weights, axis=1)


def mu_mle(theta: Array, responsibilities: Array) -> Array:
  """Calculate the maximum likelihood estimate for the mean angles."""
  complex_exp = jnp.exp(1j * theta)
  num = responsibilities.T @ complex_exp
  denom = jnp.abs(num)
  return jnp.where(denom > 0, jnp.angle(num), 0.0)


def m_step(
  theta: Array,
  kappa: Array,
  responsibilities: Array,
  mask: Array,
  gtol: float = 1e-3,
  maxiter: int = 100,
  line_search_maxiter: int = 10,
  min_kappa: float = 10.0,
  max_kappa: float = 600.0,
) -> tuple[Array, Array, Array, OptimizeResults]:
  """Perform the M-step of the EM algorithm.

  Args:
      theta: Data points. Shape (n_data, n_features).
      mu: Current mean directions. Shape (n_components, n_features).
      kappa: Current concentration parameters. Shape (n_components, n_features).
      responsibilities: Responsibilities array. Shape (n_data, n_components).
      mask: Mask indicating valid features. Shape (n_features,).
      gtol: Gradient tolerance for the optimizer.
      maxiter: Maximum iterations for the optimizer.
      line_search_maxiter: Maximum line search iterations for the optimizer.
      min_kappa: Minimum allowed concentration parameter.
      max_kappa: Maximum allowed concentration parameter.

  Returns:
      A tuple containing the updated mean directions, concentration parameters,
      log weights, and the optimization result.

  """

  def scale_kappa(k: Array) -> Array:
    return jnp.where(mask, jnp.minimum(jnp.maximum(min_kappa, k), max_kappa), 0)

  weights = responsibilities.sum(axis=0)

  new_mu = mu_mle(theta, responsibilities)
  new_log_weights = jnp.log(weights / weights.sum())

  def negative_ell(p: Array, m: Array, k: Array, r_val: Array, mask_val: Array) -> Float:
    p_reshaped = p.reshape(k.shape)
    kappas = scale_kappa(p_reshaped)

    component_log_pdfs = vmap(
      von_mises_log_pdf,
      in_axes=(None, 0, 0, None),
      out_axes=1,
    )(m, new_mu, kappas, mask_val)

    return -jnp.sum(r_val * component_log_pdfs)

  solver_options = {"gtol": gtol, "maxiter": maxiter, "line_search_maxiter": line_search_maxiter}
  result = minimize(
    fun=lambda p: negative_ell(p, theta, kappa, responsibilities, mask),
    x0=kappa.flatten(),
    method="bfgs",
    options=solver_options,
  )
  new_kappa = result.x.reshape(kappa.shape)

  return (
    new_mu,
    jnp.where(jnp.isfinite(new_kappa), new_kappa, kappa),
    new_log_weights,
    result,
  )


@jit
def em_step(
  carry: tuple[MixtureFitState, Array, Array, float, float, int],
  iteration_and_converged_flag: tuple[int, bool],
) -> tuple[MixtureFitState, Array, Array, float, float, int]:
  """Perform a single step of the EM algorithm.

  This function is designed to be the body of `jax.lax.scan`.

  Args:
      carry: A tuple containing the current state and other static parameters.
      iteration_and_converged_flag: A tuple containing the current iteration and
                                    a boolean flag indicating convergence.

  Returns:
      A tuple containing the updated state and a placeholder for the scan output.

  """
  state, theta, mask, atol, gtol, gmaxiter = carry
  i, converged = iteration_and_converged_flag

  def update_state(current_state: MixtureFitState) -> MixtureFitState:
    responsibilities = e_step(
      x=theta,
      mu=current_state.mu,
      kappa=current_state.kappa,
      log_weights=current_state.log_weights,
      mask=mask,
    )
    mu, kappa, log_weights, result = m_step(
      theta=theta,
      kappa=current_state.kappa,
      responsibilities=responsibilities,
      mask=mask,
      gtol=gtol,
      maxiter=gmaxiter,
    )
    ll = jnp.mean(von_mises_mixture_log_pdf(theta, mu, kappa, log_weights, mask))

    # Update state fields
    log_likelihood = current_state.log_likelihood.at[i].set(ll)
    statuses = current_state.statuses.at[i].set(result.status)
    njevs = current_state.n_jacobian_evals.at[i].set(result.njev)
    nfevs = current_state.n_func_evals.at[i].set(result.nfev)
    nits = current_state.n_iterations.at[i].set(result.nit)
    successes = current_state.successes.at[i].set(result.success)

    return MixtureFitState(
      mu=mu,
      kappa=kappa,
      log_weights=log_weights,
      log_likelihood=log_likelihood,
      statuses=statuses,
      n_jacobian_evals=njevs,
      n_func_evals=nfevs,
      n_iterations=nits,
      successes=successes,
    )

  def do_nothing(current_state: MixtureFitState) -> MixtureFitState:
    return current_state

  next_state = cond(
    converged,
    do_nothing,
    update_state,
    state,
  )

  next_converged = cond(
    i > 0,
    lambda: jnp.abs(next_state.log_likelihood[i] - next_state.log_likelihood[i - 1]) < atol,
    lambda: False,
  )

  return (next_state, theta, mask, atol, gtol, gmaxiter), next_converged  # type: ignore[return-type]


@partial(jax.jit, static_argnames=("max_iter", "atol", "gtol", "gmaxiter"))
def fit(
  theta: Array,
  init_state: MixtureFitState,
  mask: Array,
  max_iter: int = 100,
  atol: float = 1e-2,
  gtol: float = 1e-3,
  gmaxiter: int = 500,
) -> tuple[MixtureFitState, Array]:
  """Fits a von Mises mixture model using the EM algorithm.

  Args:
      theta: Data points. Shape (n_data, n_features).
      init_state: Initial state of the mixture model.
      mask: Mask indicating valid features. Shape (n_features,).
      max_iter: Maximum number of EM iterations.
      atol: Absolute tolerance for convergence.
      gtol: Gradient tolerance for the optimizer.
      gmaxiter: Maximum iterations for the optimizer.

  Returns:
      A tuple containing the final state and the responsibilities after fitting.

  """
  init_carry = (init_state, theta, mask, atol, gtol, gmaxiter)
  final_carry, _ = scan(
    em_step,
    init_carry,
    (jnp.arange(max_iter), jnp.zeros(max_iter, dtype=bool)),
  )
  final_state, _, _, _, _, _ = final_carry
  responsibilities = e_step(
    theta,
    final_state.mu,
    final_state.kappa,
    final_state.log_weights,
    mask,
  )
  return sort_by_weight(final_state, responsibilities)


_MIN_DISTANCE_THRESHOLD = 1e-8


def spherical_kmeans_plus_plus(
  theta: Array,
  n_clusters: int,
  key: PRNGKeyArray,
) -> Array:
  """Adapts the k-means++ initialization to the unit sphere.

  Args:
      theta: Data points. Shape (n_data, n_features).
      n_clusters: The number of clusters to find.
      key: A JAX PRNG key.

  Returns:
      Initial cluster centers. Shape (n_clusters, n_features).

  """
  keys = jax.random.split(key, n_clusters)
  ix = jax.random.randint(keys[0], (1,), 0, theta.shape[0])
  inv_norm = 1 / jnp.linalg.norm(theta, axis=1)
  theta_hat = theta * inv_norm[:, jnp.newaxis]
  normalized_cluster_centers = theta_hat[ix].T
  ixs = [ix]
  for k in keys[1:]:
    cosine_distances = 1 - theta_hat @ normalized_cluster_centers
    min_distance = jnp.min(cosine_distances, axis=1)
    log_p = jnp.where(
      min_distance > _MIN_DISTANCE_THRESHOLD,
      2 * jnp.log(min_distance),
      -jnp.inf,
    )
    ix = jax.random.categorical(k, log_p, shape=(1,))
    new_center = theta_hat[ix].T
    normalized_cluster_centers = jnp.hstack((normalized_cluster_centers, new_center)).squeeze()
    ixs.append(ix)
  ixs = jnp.concatenate(ixs)
  return theta[ixs]


def _init_params(
  mu: Array,
  cloud_mask: Array,
  k_components: int,
  key: PRNGKeyArray,
) -> dict[str, Array]:
  """Assumes mu has been generated with kmeans++.

  Initializes random concentration parameters from a mask and an integer that
  sets the number of components.

  Args:
      mu: Initial mean directions from k-means++.
      cloud_mask: Mask indicating valid features.
      k_components: Number of mixture components.
      key: A JAX PRNG key.

  Returns:
      A dictionary containing the initialized parameters.

  """
  w = jnp.ones(k_components) / k_components
  kappa_bond_key, kappa_torsion_key = jax.random.split(key, 2)

  kappa_bond = (
    jax.random.uniform(kappa_bond_key, minval=100, maxval=500, shape=(k_components, 15))
    * cloud_mask[:15]
  )
  kappa_torsion = (
    jax.random.uniform(kappa_torsion_key, minval=25, maxval=50, shape=(k_components, 13))
    * cloud_mask[15:]
  )
  kappa = jnp.concatenate([kappa_bond, kappa_torsion], axis=-1)
  return {
    "mu": mu,
    "kappa": kappa,
    "log_weights": jax.nn.log_softmax(jnp.log(w)),
  }


@jit
def analytical_kl(mu: Array, kappa: Array, mask: Array) -> Float:
  """Calculate KL-divergence based on analytical expression from Kitagawa and Rowley.

  Args:
      mu: Mean directions of the two distributions. Shape (2, n_features).
      kappa: Concentration parameters of the two distributions. Shape (2, n_features).
      mask: Mask indicating valid features. Shape (n_features,).

  Returns:
      The analytical KL divergence.

  """
  i1 = vmap(jax.grad(i0))
  rv = i1(kappa[0]) / i0(kappa[0])
  cos_mu_diff = jnp.cos(mu[0] - mu[1])
  mu_terms = jnp.sum(rv * (kappa[0] - kappa[1] * cos_mu_diff))
  kappa_terms = jnp.sum(jnp.where(mask, -jnp.log(i0(kappa[0]) / i0(kappa[1])), 0.0))
  return mu_terms + kappa_terms


def load_mixture_state(f: str) -> MixtureFitState:
  """Load a saved mixture state from an npz file.

  Args:
      f: Path to the .npz file.

  Returns:
      A MixtureFitState dataclass initialized with the loaded arrays.

  """
  states_loaded = np.load(f)
  return MixtureFitState(**{k: jnp.array(states_loaded[k]) for k in states_loaded.files})
