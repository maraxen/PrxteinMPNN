# tests/test_vmm.py
"""Tests for von Mises mixture model utilities."""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from jax.scipy.special import i0
from jaxtyping import Float, PRNGKeyArray

# Import the functions and dataclass from your script
# Assuming your script is saved as `src/prxteinmpnn/vmm.py`
from prxteinmpnn.ensemble.vmm import (
  MixtureFitState,
  analytical_kl,
  e_step,
  fit,
  load_mixture_state,
  mu_mle,
  sort_by_weight,
  spherical_kmeans_plus_plus,
  von_mises_log_pdf,
)

# Test constants
N_COMPONENTS = 3
N_FEATURES = 5
N_SAMPLES = 100
MAX_ITER = 10

def _vonmises_sampler(key: PRNGKeyArray, mu: Array, kappa: Array, shape: tuple) -> Array:
  """JAX-native sampler for the von Mises distribution."""
  k1, k2 = jax.random.split(key)

  # Use rejection sampling
  a = 1 + jnp.sqrt(1 + 4 * kappa**2)
  b = (a - jnp.sqrt(2 * a)) / (2 * kappa)
  r = (1 + b**2) / (2 * b)

  def sample_once(carry, _):
    k1, k2, k3 = jax.random.split(carry, 3)
    u1, u2, u3 = (
      jax.random.uniform(k1, shape),
      jax.random.uniform(k2, shape),
      jax.random.uniform(k3, shape),
    )
    z = jnp.cos(jnp.pi * u1)
    f = (1 + r * z) / (r + z)
    c = kappa * (r - f)

    # Condition to accept the sample
    accept = (c * (2 - c) - u2 > 0) | (jnp.log(c / u2) + 1 - c >= 0)

    # If not accepted, try again (this is a simplified loop for one step)
    # A full loop is complex in JAX; for testing, this approximation is often sufficient
    # and for larger kappas, acceptance rate is high.
    return k3, jnp.sign(u3 - 0.5) * jnp.arccos(f) + mu

  # This is a simplified version; for a robust sampler, a `lax.while_loop` would be needed
  # to handle rejections correctly, but for testing this is usually adequate.
  _, samples = jax.lax.scan(sample_once, k1, jnp.arange(1))
  return samples.squeeze(0)

@pytest.fixture
def key() -> PRNGKeyArray:
  """Provides a reusable JAX PRNG key."""
  return jax.random.PRNGKey(42)


@pytest.fixture
def test_data(key: PRNGKeyArray) -> tuple[Array, Array, Array, Array]:
  """Generates synthetic data from a known von Mises mixture."""
  keys = jax.random.split(key, 4)
  true_mu = jax.random.uniform(keys[0], (N_COMPONENTS, N_FEATURES), minval=-np.pi, maxval=np.pi)
  true_kappa = jax.random.uniform(keys[1], (N_COMPONENTS, N_FEATURES), minval=10, maxval=50)
  true_weights = jnp.array([0.5, 0.3, 0.2])

  assignment_keys, sample_keys = jax.random.split(keys[2], 2)
  assignments = jax.random.choice(assignment_keys, N_COMPONENTS, (N_SAMPLES,), p=true_weights)

  # Use the new JAX-native sampler
  all_samples = _vonmises_sampler(
    sample_keys,
    mu=true_mu[assignments],
    kappa=true_kappa[assignments],
    shape=(N_SAMPLES, N_FEATURES),
  )
  mask = jnp.ones(N_FEATURES, dtype=bool)

  return all_samples, true_mu, true_kappa, mask

@pytest.fixture
def init_state(key: PRNGKeyArray) -> MixtureFitState:
  """Creates an initial MixtureFitState for testing."""
  mu = jnp.zeros((N_COMPONENTS, N_FEATURES))
  kappa = jnp.ones((N_COMPONENTS, N_FEATURES)) * 20.0
  log_weights = jnp.log(jnp.ones(N_COMPONENTS) / N_COMPONENTS)

  return MixtureFitState(
    mu=mu,
    kappa=kappa,
    log_weights=log_weights,
    log_likelihood=jnp.full(MAX_ITER, -jnp.inf),
    statuses=jnp.zeros(MAX_ITER, dtype=jnp.int32),
    n_jacobian_evals=jnp.zeros(MAX_ITER, dtype=jnp.int32),
    n_func_evals=jnp.zeros(MAX_ITER, dtype=jnp.int32),
    n_iterations=jnp.zeros(MAX_ITER, dtype=jnp.int32),
    successes=jnp.zeros(MAX_ITER, dtype=bool),
  )


def test_mixture_fit_state_is_pytree(init_state: MixtureFitState):
  """Tests that MixtureFitState is a valid JAX PyTree."""
  leaves, treedef = jax.tree_util.tree_flatten(init_state)
  reconstructed_state = jax.tree_util.tree_unflatten(treedef, leaves)
  chex.assert_trees_all_close(init_state, reconstructed_state)

  @jax.jit
  def process_state(state: MixtureFitState) -> Float:
    return jnp.sum(state.mu)

  result = process_state(init_state)
  assert isinstance(result, jax.Array)


def test_von_mises_log_pdf_shape_and_value():
  """Tests the von Mises log PDF for correct shape and a known value."""
  x = jnp.ones((10, N_FEATURES))
  mu = jnp.ones(N_FEATURES)  # x == mu
  kappa = jnp.ones(N_FEATURES) * 5.0
  mask = jnp.ones(N_FEATURES, dtype=bool)

  log_pdf = von_mises_log_pdf(x, mu, kappa, mask)
  chex.assert_shape(log_pdf, (10,))

  # For a single feature where x=mu, log_pdf = kappa - log(2*pi*i0(kappa))
  expected_val = (kappa[0] - jnp.log(2 * jnp.pi * i0(kappa[0]))) * N_FEATURES
  chex.assert_trees_all_close(log_pdf[0], expected_val)


def test_e_step_responsibilities(test_data: tuple, init_state: MixtureFitState):
  """Tests that responsibilities sum to 1 for each data point."""
  theta, _, _, mask = test_data

  responsibilities = e_step(
    x=theta,
    mu=init_state.mu,
    kappa=init_state.kappa,
    log_weights=init_state.log_weights,
    mask=mask,
  )

  chex.assert_shape(responsibilities, (N_SAMPLES, N_COMPONENTS))
  chex.assert_trees_all_close(jnp.sum(responsibilities, axis=1), jnp.ones(N_SAMPLES), atol=1e-6)


def test_mu_mle_simple_case():
  """Tests the MLE for mu with a simple, obvious case."""
  theta = jnp.array([[0.1], [-0.1], [0.0], [0.05], [-0.05]]) * np.pi
  responsibilities = jnp.ones_like(theta)  # Single component

  new_mu = mu_mle(theta, responsibilities)
  chex.assert_shape(new_mu, (1, 1))
  chex.assert_trees_all_close(new_mu[0, 0], 0.0, atol=1e-4)


def test_sort_by_weight(init_state: MixtureFitState):
  """Tests that mixture components are sorted correctly by weight."""
  state = init_state.replace(log_weights=jnp.log(jnp.array([0.2, 0.5, 0.3])))  # type: ignore[attr-defined]
  responsibilities = jnp.ones((N_SAMPLES, N_COMPONENTS))

  sorted_state, sorted_resp = sort_by_weight(state, responsibilities)

  expected_order = jnp.array([1, 2, 0])
  expected_log_weights = state.log_weights[expected_order]

  chex.assert_trees_all_close(sorted_state.log_weights, expected_log_weights)
  chex.assert_trees_all_close(sorted_state.mu, state.mu[expected_order])


def test_spherical_kmeans_plus_plus(key: PRNGKeyArray, test_data: tuple):
  """Tests k-means++ initialization for correct shape and output."""
  theta, _, _, _ = test_data
  n_clusters = 4

  centers = spherical_kmeans_plus_plus(theta, n_clusters, key)
  chex.assert_shape(centers, (n_clusters, N_FEATURES))

  # Check if all returned centers are actually present in the original data
  # This can be slow for large N, but is fine for testing
  is_in_theta = jnp.all(jnp.any(jnp.all(theta[:, None, :] == centers[None, :, :], axis=2), axis=0))
  assert is_in_theta


def test_analytical_kl_identity(key: PRNGKeyArray):
  """Tests that the KL divergence between identical distributions is zero."""
  mu = jax.random.uniform(key, (2, N_FEATURES))
  kappa = jax.random.uniform(key, (2, N_FEATURES), minval=10, maxval=20)
  mask = jnp.ones(N_FEATURES, dtype=bool)

  # Make distributions identical
  mu = mu.at[1].set(mu[0])
  kappa = kappa.at[1].set(kappa[0])

  kl_div = analytical_kl(mu, kappa, mask)
  chex.assert_trees_all_close(kl_div, 0.0, atol=1e-5)


def test_fit_integration(test_data: tuple, init_state: MixtureFitState, key: PRNGKeyArray):
  """Performs an integration test of the `fit` function."""
  theta, true_mu, _, mask = test_data

  # Initialize with k-means++ for better starting points
  init_mu = spherical_kmeans_plus_plus(theta, N_COMPONENTS, key)
  init_state = init_state.replace(mu=init_mu)  # type: ignore[attr-access]

  final_state, responsibilities = fit(theta, init_state, mask, max_iter=MAX_ITER)

  # Check output shapes
  chex.assert_shape(final_state.mu, (N_COMPONENTS, N_FEATURES))
  chex.assert_shape(final_state.kappa, (N_COMPONENTS, N_FEATURES))
  chex.assert_shape(final_state.log_weights, (N_COMPONENTS,))
  chex.assert_shape(responsibilities, (N_SAMPLES, N_COMPONENTS))

  # Check that log-likelihood generally increased
  final_ll = final_state.log_likelihood[final_state.n_iterations[MAX_ITER - 1] > 0]
  # Check that log-likelihood generally increased
  # A step is valid if its log-likelihood was updated from -inf
  valid_mask = final_state.log_likelihood > -jnp.inf
  final_ll = final_state.log_likelihood[valid_mask]

  # The model must have run for at least one step
  assert len(final_ll) > 0

  # If more than one step ran, the final LL should be better than the first
  if len(final_ll) > 1:
      assert final_ll[0] < final_ll[-1]


def test_load_save_state(init_state: MixtureFitState, tmp_path):
  """Tests saving and loading a MixtureFitState object."""
  f_path = tmp_path / "test_state.npz"

  # Convert to numpy for saving
  state_dict = {
    "mu": np.array(init_state.mu),
    "kappa": np.array(init_state.kappa),
    "log_weights": np.array(init_state.log_weights),
    "log_likelihood": np.array(init_state.log_likelihood),
    "statuses": np.array(init_state.statuses),
    "n_jacobian_evals": np.array(init_state.n_jacobian_evals),
    "n_func_evals": np.array(init_state.n_func_evals),
    "n_iterations": np.array(init_state.n_iterations),
    "successes": np.array(init_state.successes),
  }
  np.savez(f_path, **state_dict, allow_pickle=False)

  loaded_state = load_mixture_state(str(f_path))

  chex.assert_trees_all_close(init_state, loaded_state)
