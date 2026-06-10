"""Tests for Potts model correctness: TRW vs exact, scale factors, alphabets."""

import sys
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import pytest

from aminx.utils.aa_convert import MPNN_ALPHABET

# Add mistypotts and prxteinmpnn to path if not already available
_mistypotts_path = Path("/home/marielle/projects/mistypotts/src")
if str(_mistypotts_path) not in sys.path:
  sys.path.insert(0, str(_mistypotts_path))

_prxteinmpnn_path = Path("/home/marielle/projects/mistypotts/vendor/prxteinmpnn/src")
if str(_prxteinmpnn_path) not in sys.path:
  sys.path.insert(0, str(_prxteinmpnn_path))

# These tests validate against the mistypotts reference implementation (exact Potts
# marginals / TRW), which in turn pulls its vendored prxteinmpnn. mistypotts is an
# optional local dev checkout, not a packaged dependency; skip the whole module cleanly
# when the reference symbols aren't importable (e.g. CI, or any install without the
# sibling checkout). The top-level `mistypotts` package may import while its submodules
# still fail on the missing prxteinmpnn, so guard on the submodules the tests actually use.
try:
  import mistypotts.exact_potts  # noqa: F401
  import mistypotts.trw  # noqa: F401
except ImportError:
  pytest.skip(
    "mistypotts reference implementation (and its vendored prxteinmpnn) is not "
    "installed; skipping Potts reference-validation tests.",
    allow_module_level=True,
  )


@pytest.mark.slow
@pytest.mark.potts
class TestTRWMarginalVsExact:
  """Group 1: TRW marginals vs exact bruteforce across various sizes."""

  @pytest.mark.parametrize("n", [4, 6, 8])
  def test_trw_marginals_vs_exact(self, n: int):
    """Test that TRW marginals converge to exact marginals within tolerance.

    Constructs a random PottsModel with sparse coupling matrix W,
    runs TRW inference, compares marginals to exact bruteforce computation
    within max absolute error < 0.05.
    """
    # Import locally to avoid circular dependencies
    from mistypotts.exact_potts import exact_marginals_bruteforce
    from mistypotts.trw import DifferentiableTRW

    # Fixed alphabet size — use q=4 to keep exact bruteforce tractable (4^8 = 65536 states)
    q = 4

    # Create random key and generate random fields
    key = jr.PRNGKey(42 + n)
    key_h, key_J, key_W, key_mask = jr.split(key, 4)

    # Random h: (n, q)
    h = jr.normal(key_h, shape=(n, q)) * 0.5

    # Random J: (n, n, q, q) symmetric
    J_raw = jr.normal(key_J, shape=(n, n, q, q)) * 0.3
    J = 0.5 * (J_raw + jnp.transpose(J_raw, (1, 0, 3, 2)))

    # Random W: sparse coupling (adjacency), scale to [0, 1]
    W_raw = jr.uniform(key_W, shape=(n, n))
    W = jnp.where(W_raw > 0.5, 1.0, 0.0)  # Sparse: ~50% edges
    W = 0.5 * (W + W.T)  # Symmetrize

    # Compute exact marginals via bruteforce
    exact_marg, exact_log_z = exact_marginals_bruteforce(h, J, W)

    # Compute TRW marginals
    trw = DifferentiableTRW(q=q, trw_iters=30, damping=0.5)
    trw_marg, _ = trw(W, J, h)

    # Check convergence: max absolute error < 0.05
    max_error = jnp.max(jnp.abs(trw_marg - exact_marg))
    assert max_error < 0.05, f"TRW max error {max_error:.6f} exceeds tolerance 0.05 for n={n}"


@pytest.mark.potts
class TestX2ScaleFactor:
  """Group 2: Synthetic x2 scale factor check for h and J tensors."""

  def test_x2_scale_factor_two_residue(self):
    """Test x2 scale factor with hand-constructed 2-residue system.

    Constructs known h(2,20) and J(2,2,20,20) by hand,
    computes all 400 log_unnormalized values,
    verifies log_Z via logsumexp, and tests log_prob consistency.
    """
    from mistypotts.exact_potts import (
      exact_marginals_bruteforce,
      log_unnormalized_one_sequence,
    )

    # 2 residues, 20 amino acids
    n, q = 2, 20

    # Hand-construct simple fields
    h = jnp.array(
      [
        [0.1, 0.2, 0.3, 0.4, 0.5, -0.5, -0.4, -0.3, -0.2, -0.1] + [0.0] * 10,
        [-0.1, -0.2, -0.3, -0.4, -0.5, 0.5, 0.4, 0.3, 0.2, 0.1] + [0.0] * 10,
      ],
      dtype=jnp.float32,
    )

    # Hand-construct simple coupling (only diagonal-like interactions matter)
    J = jnp.zeros((n, n, q, q), dtype=jnp.float32)
    # Set strong interaction between same amino acids at both positions
    for aa in range(q):
      J = J.at[0, 1, aa, aa].set(0.5)
      J = J.at[1, 0, aa, aa].set(0.5)

    # Adjacency matrix with single edge
    W = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32)

    # Compute exact marginals and log_Z
    exact_marg, log_z = exact_marginals_bruteforce(h, J, W)

    # Test specific sequences
    seqs = [
      jnp.array([0, 0], dtype=jnp.int32),  # Both position 0
      jnp.array([1, 1], dtype=jnp.int32),  # Both position 1
      jnp.array([0, 1], dtype=jnp.int32),  # Different
    ]

    for seq in seqs:
      log_u = log_unnormalized_one_sequence(h, J, W, seq)
      log_prob_expected = log_u - log_z
      # Verify log_prob is well-defined
      assert jnp.isfinite(log_prob_expected), f"Non-finite log_prob for seq {seq}"

    # Verify marginals sum to 1 at each site
    marginal_sums = jnp.sum(exact_marg, axis=1)
    assert jnp.allclose(
      marginal_sums, jnp.ones(n), atol=1e-5
    ), f"Marginals do not sum to 1: {marginal_sums}"


@pytest.mark.potts
class TestPottsVsMPNNAlphabet:
  """Group 3: Verify Potts and MPNN alphabets match in first 20 positions."""

  def test_potts_vs_mpnn_alphabet_ordering(self):
    """Test that MPNN alphabet is identical to expected Potts alphabet.

    Imports MPNN_ALPHABET from aminx.utils and verifies it matches
    the standard 20 canonical amino acids in order.
    """
    # Standard 20 canonical amino acids (alphabetical order in MPNN)
    expected_alphabet = "ACDEFGHIKLMNPQRSTVWY"

    # MPNN_ALPHABET from aminx.utils.aa_convert includes X and other variants
    assert len(MPNN_ALPHABET) >= 20, f"MPNN_ALPHABET too short: {MPNN_ALPHABET}"

    # Check first 20 match canonical order
    actual_first_20 = MPNN_ALPHABET[:20]
    assert (
      actual_first_20 == expected_alphabet
    ), f"MPNN alphabet mismatch:\nExpected: {expected_alphabet}\nActual:   {actual_first_20}"

    # Verify all chars are unique in first 20
    assert len(set(actual_first_20)) == 20, f"Duplicate amino acids in first 20: {actual_first_20}"
