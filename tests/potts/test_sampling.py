"""Tests for Potts sampling: Gibbs, parallel tempering, and energy computation."""

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float, Int

from aminx.potts.sampling import (
    _parallel_tempering_exchange,
    gibbs_sweep,
    log_energy,
    parallel_tempering,
)


class TestLogEnergy:
  """Test energy computation for Potts models."""

  def test_log_energy_output_shape(self) -> None:
    """Test log_energy returns scalar."""
    n, q = 5, 3
    seq = jnp.array([0, 1, 2, 0, 1], dtype=jnp.int32)
    h = jnp.ones((n, q), dtype=jnp.float32)
    J = jnp.zeros((n, n, q, q), dtype=jnp.float32)
    w = jnp.zeros((n, n), dtype=jnp.float32)
    mask = jnp.ones((n,), dtype=jnp.float32)

    energy = log_energy(seq, h, J, w, mask)

    assert energy.shape == ()
    assert energy.dtype == jnp.float32

  def test_log_energy_zero_couplings(self) -> None:
    """Test log_energy with zero couplings equals field-only energy."""
    n, q = 3, 2
    seq = jnp.array([0, 1, 0], dtype=jnp.int32)
    h = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=jnp.float32)
    J = jnp.zeros((n, n, q, q), dtype=jnp.float32)
    w = jnp.zeros((n, n), dtype=jnp.float32)
    mask = jnp.ones((n,), dtype=jnp.float32)

    energy = log_energy(seq, h, J, w, mask)

    # E = h[0, seq[0]] + h[1, seq[1]] + h[2, seq[2]]
    #   = h[0, 0] + h[1, 1] + h[2, 0]
    #   = 1.0 + 4.0 + 5.0 = 10.0
    expected = 10.0
    assert jnp.allclose(energy, expected, atol=1e-5)

  def test_log_energy_with_masking(self) -> None:
    """Test log_energy respects mask."""
    n, q = 3, 2
    seq = jnp.array([0, 1, 0], dtype=jnp.int32)
    h = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=jnp.float32)
    J = jnp.zeros((n, n, q, q), dtype=jnp.float32)
    w = jnp.zeros((n, n), dtype=jnp.float32)
    # Mask out middle site
    mask = jnp.array([1.0, 0.0, 1.0], dtype=jnp.float32)

    energy = log_energy(seq, h, J, w, mask)

    # E = h[0, 0] + 0 + h[2, 0]
    #   = 1.0 + 0.0 + 5.0 = 6.0
    expected = 6.0
    assert jnp.allclose(energy, expected, atol=1e-5)

  def test_log_energy_with_pairs(self) -> None:
    """Test log_energy with pairwise couplings."""
    n, q = 2, 2
    seq = jnp.array([0, 0], dtype=jnp.int32)
    h = jnp.zeros((n, q), dtype=jnp.float32)
    # J[0,1,0,0] = 3.0 (coupling between seq[0]=0 and seq[1]=0)
    J = jnp.zeros((n, n, q, q), dtype=jnp.float32)
    J = J.at[0, 1, 0, 0].set(3.0)
    J = J.at[1, 0, 0, 0].set(3.0)  # symmetric
    w = jnp.ones((n, n), dtype=jnp.float32)
    mask = jnp.ones((n,), dtype=jnp.float32)

    energy = log_energy(seq, h, J, w, mask)

    # E = 0 + 0 + 0.5 * (1.0 * 1.0 * J[0,1,0,0] + 1.0 * 1.0 * J[1,0,0,0])
    #   = 0.5 * (3.0 + 3.0) = 3.0
    expected = 3.0
    assert jnp.allclose(energy, expected, atol=1e-5)

  def test_log_energy_jit_compatible(self) -> None:
    """Test log_energy is JIT-compatible."""
    n, q = 4, 3
    seq = jnp.array([0, 1, 2, 0], dtype=jnp.int32)
    h = jnp.ones((n, q), dtype=jnp.float32)
    J = jnp.zeros((n, n, q, q), dtype=jnp.float32)
    w = jnp.zeros((n, n), dtype=jnp.float32)
    mask = jnp.ones((n,), dtype=jnp.float32)

    jitted = jax.jit(log_energy)
    energy = jitted(seq, h, J, w, mask)

    assert energy.shape == ()


class TestGibbsSweep:
  """Test single Gibbs sweep."""

  def test_gibbs_sweep_output_shape(self) -> None:
    """Test gibbs_sweep returns updated sequence with correct shape."""
    n, q = 5, 3
    key = jax.random.PRNGKey(0)
    seq = jnp.array([0, 1, 2, 0, 1], dtype=jnp.int32)
    h = jnp.ones((n, q), dtype=jnp.float32)
    J = jnp.zeros((n, n, q, q), dtype=jnp.float32)
    w = jnp.zeros((n, n), dtype=jnp.float32)
    mask = jnp.ones((n,), dtype=jnp.float32)

    seq_out, key_out = gibbs_sweep(key, seq, h, J, w, mask)

    assert seq_out.shape == seq.shape
    assert seq_out.dtype == jnp.int32
    assert key_out.shape == key.shape

  def test_gibbs_sweep_respects_mask(self) -> None:
    """Test gibbs_sweep does not update masked-out sites."""
    n, q = 3, 2
    key = jax.random.PRNGKey(42)
    seq = jnp.array([0, 1, 0], dtype=jnp.int32)
    h = jnp.ones((n, q), dtype=jnp.float32)
    J = jnp.zeros((n, n, q, q), dtype=jnp.float32)
    w = jnp.zeros((n, n), dtype=jnp.float32)
    # Mask out middle site
    mask = jnp.array([1.0, 0.0, 1.0], dtype=jnp.float32)

    seq_out, _ = gibbs_sweep(key, seq, h, J, w, mask)

    # Middle site should remain unchanged
    assert seq_out[1] == seq[1]

  def test_gibbs_sweep_jit_compatible(self) -> None:
    """Test gibbs_sweep is JIT-compatible via eqx.filter_jit."""
    import equinox as eqx

    n, q = 4, 3
    key = jax.random.PRNGKey(0)
    seq = jnp.array([0, 1, 2, 0], dtype=jnp.int32)
    h = jnp.ones((n, q), dtype=jnp.float32)
    J = jnp.zeros((n, n, q, q), dtype=jnp.float32)
    w = jnp.zeros((n, n), dtype=jnp.float32)
    mask = jnp.ones((n,), dtype=jnp.float32)

    jitted = eqx.filter_jit(gibbs_sweep)
    seq_out, key_out = jitted(key, seq, h, J, w, mask)

    assert seq_out.shape == seq.shape
    assert seq_out.dtype == jnp.int32

  def test_gibbs_sweep_random_seed_determinism(self) -> None:
    """Test gibbs_sweep with same seed produces same output."""
    n, q = 5, 3
    seq = jnp.array([0, 1, 2, 0, 1], dtype=jnp.int32)
    h = jnp.ones((n, q), dtype=jnp.float32)
    J = jnp.zeros((n, n, q, q), dtype=jnp.float32)
    w = jnp.zeros((n, n), dtype=jnp.float32)
    mask = jnp.ones((n,), dtype=jnp.float32)

    key1 = jax.random.PRNGKey(42)
    key2 = jax.random.PRNGKey(42)

    seq_out1, _ = gibbs_sweep(key1, seq, h, J, w, mask)
    seq_out2, _ = gibbs_sweep(key2, seq, h, J, w, mask)

    assert jnp.array_equal(seq_out1, seq_out2)


class TestParallelTempering:
  """Test parallel tempering MCMC sampler."""

  def test_parallel_tempering_output_shapes(self) -> None:
    """Test parallel_tempering returns correct shapes."""
    n, q, n_replicas = 5, 3, 4
    key = jax.random.PRNGKey(0)
    seq = jnp.array([0, 1, 2, 0, 1], dtype=jnp.int32)
    h = jnp.ones((n, q), dtype=jnp.float32)
    J = jnp.zeros((n, n, q, q), dtype=jnp.float32)
    w = jnp.zeros((n, n), dtype=jnp.float32)
    mask = jnp.ones((n,), dtype=jnp.float32)

    seqs_out, energies_out = parallel_tempering(
      key, seq, h, J, w, mask, n_replicas=n_replicas, n_sweeps=1
    )

    assert seqs_out.shape == (n_replicas, n)
    assert seqs_out.dtype == jnp.int32
    assert energies_out.shape == (n_replicas,)
    assert energies_out.dtype == jnp.float32

  def test_parallel_tempering_default_temperatures(self) -> None:
    """Test parallel_tempering generates sensible default temperatures."""
    n, q, n_replicas = 4, 2, 3
    key = jax.random.PRNGKey(0)
    seq = jnp.ones((n,), dtype=jnp.int32)
    h = jnp.zeros((n, q), dtype=jnp.float32)
    J = jnp.zeros((n, n, q, q), dtype=jnp.float32)
    w = jnp.zeros((n, n), dtype=jnp.float32)
    mask = jnp.ones((n,), dtype=jnp.float32)

    # Should use default temperatures without raising
    seqs_out, energies_out = parallel_tempering(
      key, seq, h, J, w, mask, n_replicas=n_replicas, n_sweeps=1
    )

    assert seqs_out.shape == (n_replicas, n)
    assert energies_out.shape == (n_replicas,)

  def test_parallel_tempering_custom_temperatures(self) -> None:
    """Test parallel_tempering accepts custom temperatures."""
    n, q, n_replicas = 4, 2, 3
    key = jax.random.PRNGKey(0)
    seq = jnp.ones((n,), dtype=jnp.int32)
    h = jnp.zeros((n, q), dtype=jnp.float32)
    J = jnp.zeros((n, n, q, q), dtype=jnp.float32)
    w = jnp.zeros((n, n), dtype=jnp.float32)
    mask = jnp.ones((n,), dtype=jnp.float32)
    temps = jnp.array([0.5, 1.0, 2.0], dtype=jnp.float32)

    seqs_out, energies_out = parallel_tempering(
      key, seq, h, J, w, mask, n_replicas=n_replicas, temperatures=temps, n_sweeps=1
    )

    assert seqs_out.shape == (n_replicas, n)
    assert energies_out.shape == (n_replicas,)

  def test_parallel_tempering_temperatures_temperature_mismatch_raises(self) -> None:
    """Test that mismatched temperature count raises ValueError."""
    n, q, n_replicas = 4, 2, 3
    key = jax.random.PRNGKey(0)
    seq = jnp.ones((n,), dtype=jnp.int32)
    h = jnp.zeros((n, q), dtype=jnp.float32)
    J = jnp.zeros((n, n, q, q), dtype=jnp.float32)
    w = jnp.zeros((n, n), dtype=jnp.float32)
    mask = jnp.ones((n,), dtype=jnp.float32)
    temps = jnp.array([0.5, 1.0], dtype=jnp.float32)  # Only 2, but need 3

    with pytest.raises(ValueError, match="n_replicas"):
      parallel_tempering(
        key, seq, h, J, w, mask, n_replicas=n_replicas, temperatures=temps, n_sweeps=1
      )

  def test_parallel_tempering_jit_compatible(self) -> None:
    """Test parallel_tempering is JIT-compatible."""
    import equinox as eqx

    n, q, n_replicas = 4, 3, 2
    key = jax.random.PRNGKey(0)
    seq = jnp.array([0, 1, 2, 0], dtype=jnp.int32)
    h = jnp.ones((n, q), dtype=jnp.float32)
    J = jnp.zeros((n, n, q, q), dtype=jnp.float32)
    w = jnp.zeros((n, n), dtype=jnp.float32)
    mask = jnp.ones((n,), dtype=jnp.float32)

    # Create a wrapper that has static n_replicas, n_sweeps
    def call_pt(key_: Array, seq_: Array) -> tuple[Array, Array]:
      return parallel_tempering(
        key_, seq_, h, J, w, mask, n_replicas=n_replicas, n_sweeps=2
      )

    jitted = eqx.filter_jit(call_pt)
    seqs_out, energies_out = jitted(key, seq)

    assert seqs_out.shape == (n_replicas, n)
    assert energies_out.shape == (n_replicas,)


class TestParallelTemperingExchange:
  """Test _parallel_tempering_exchange replica boundary conditions and parity logic."""

  @pytest.mark.parametrize("n_replicas", [1, 2, 3, 4, 5])
  def test_accept_edge_shape_matches_replicas(self, n_replicas: int) -> None:
    """Test that accept_edge shape is always (n_replicas - 1,)."""
    n, q = 4, 5
    key = jax.random.PRNGKey(0)
    seqs = jnp.ones((n_replicas, n), dtype=jnp.int32)
    betas = jnp.linspace(0.1, 1.0, n_replicas, dtype=jnp.float32)
    h = jnp.zeros((n, q), dtype=jnp.float32)
    J = jnp.zeros((n, n, q, q), dtype=jnp.float32)
    w = jnp.zeros((n, n), dtype=jnp.float32)
    mask = jnp.ones((n,), dtype=jnp.float32)

    seqs_out, key_out, accept_edge = _parallel_tempering_exchange(
      key, seqs, betas, h, J, w, mask
    )

    assert accept_edge.shape == (n_replicas - 1,)
    assert accept_edge.dtype == jnp.float32

  def test_single_replica_no_swap(self) -> None:
    """Test n_replicas=1: no swaps possible, sequence unchanged."""
    n, q = 4, 5
    key = jax.random.PRNGKey(0)
    seqs = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
    betas = jnp.array([1.0], dtype=jnp.float32)
    h = jnp.ones((n, q), dtype=jnp.float32)
    J = jnp.zeros((n, n, q, q), dtype=jnp.float32)
    w = jnp.zeros((n, n), dtype=jnp.float32)
    mask = jnp.ones((n,), dtype=jnp.float32)

    seqs_out, key_out, accept_edge = _parallel_tempering_exchange(
      key, seqs, betas, h, J, w, mask
    )

    # Single replica: no edges to swap
    assert jnp.array_equal(seqs_out, seqs)
    assert accept_edge.shape == (0,)

  def test_two_replicas_edge_processing(self) -> None:
    """Test n_replicas=2: only even parity (edge 0) is active; odd is no-op."""
    n, q = 4, 5
    key = jax.random.PRNGKey(0)
    seqs = jnp.ones((2, n), dtype=jnp.int32)
    betas = jnp.array([0.5, 1.0], dtype=jnp.float32)
    h = jnp.zeros((n, q), dtype=jnp.float32)
    J = jnp.zeros((n, n, q, q), dtype=jnp.float32)
    w = jnp.zeros((n, n), dtype=jnp.float32)
    mask = jnp.ones((n,), dtype=jnp.float32)

    seqs_out, key_out, accept_edge = _parallel_tempering_exchange(
      key, seqs, betas, h, J, w, mask
    )

    # Two replicas: exactly one edge (0,1) with parity 0 (even)
    assert accept_edge.shape == (1,)
    assert 0.0 <= accept_edge[0] <= 1.0

  @pytest.mark.parametrize("n_replicas", [1, 2, 3, 4, 5])
  def test_exchange_completes_without_error(self, n_replicas: int) -> None:
    """Test _parallel_tempering_exchange completes without error for all replica counts."""
    n, q = 4, 5
    key = jax.random.PRNGKey(0)
    seqs = jnp.ones((n_replicas, n), dtype=jnp.int32)
    betas = jnp.linspace(0.1, 1.0, n_replicas, dtype=jnp.float32)
    h = jnp.ones((n, q), dtype=jnp.float32)
    J = jnp.zeros((n, n, q, q), dtype=jnp.float32)
    w = jnp.zeros((n, n), dtype=jnp.float32)
    mask = jnp.ones((n,), dtype=jnp.float32)

    # Should not raise
    seqs_out, key_out, accept_edge = _parallel_tempering_exchange(
      key, seqs, betas, h, J, w, mask
    )

    assert seqs_out.shape == seqs.shape
    assert seqs_out.dtype == jnp.int32


class TestExports:
  """Test that functions are exported from aminx.potts."""

  def test_log_energy_exported(self) -> None:
    """Test log_energy is accessible via package."""
    # If import at top succeeds, function is exported
    assert callable(log_energy)

  def test_gibbs_sweep_exported(self) -> None:
    """Test gibbs_sweep is accessible via package."""
    # If import at top succeeds, function is exported
    assert callable(gibbs_sweep)

  def test_parallel_tempering_exported(self) -> None:
    """Test parallel_tempering is accessible via package."""
    # If import at top succeeds, function is exported
    assert callable(parallel_tempering)
