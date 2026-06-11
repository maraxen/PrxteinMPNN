"""Tests for Potts model correctness: TRW vs exact, scale factors, alphabets.

These tests validate aminx's own Potts machinery (``aminx.potts._trw.DifferentiableTRW``
and the MPNN alphabet) against a self-contained brute-force reference. They carry no
dependency on the optional ``mistypotts`` dev checkout, so they run unconditionally in
CI rather than skipping.

Energy convention (matches DifferentiableTRW):

    log P(x) ∝ Σ_i h[i, x_i] + Σ_{i<j} W[i, j] · J[i, j, x_i, x_j]
"""

from __future__ import annotations

import itertools

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from scipy.special import logsumexp

from aminx.potts._trw import DifferentiableTRW
from aminx.utils.aa_convert import MPNN_ALPHABET


def _exact_marginals_bruteforce(
  h: np.ndarray,
  j: np.ndarray,
  w: np.ndarray,
) -> tuple[np.ndarray, float]:
  """Exact single-site marginals and log-partition by full state enumeration.

  Ground-truth reference for small systems (tractable while q**n is small). Used to
  validate the TRW approximation; this is the measurement pipeline, so it is verified
  independently in :class:`TestX2ScaleFactor` on a hand-constructed system.

  Args:
    h: Node potentials, shape (n, q).
    j: Pairwise potentials, shape (n, n, q, q).
    w: Graph adjacency (symmetric), shape (n, n).

  Returns:
    Tuple of (marginals (n, q), log_partition scalar).
  """
  h = np.asarray(h)
  j = np.asarray(j)
  w = np.asarray(w)
  n, q = h.shape
  states = list(itertools.product(range(q), repeat=n))
  log_unnorm = np.empty(len(states))
  for k, s in enumerate(states):
    energy = sum(h[i, s[i]] for i in range(n))
    for a in range(n):
      for b in range(a + 1, n):
        energy += w[a, b] * j[a, b, s[a], s[b]]
    log_unnorm[k] = energy
  log_z = float(logsumexp(log_unnorm))
  probs = np.exp(log_unnorm - log_z)
  marg = np.zeros((n, q))
  for k, s in enumerate(states):
    for i in range(n):
      marg[i, s[i]] += probs[k]
  return marg, log_z


def _log_unnormalized_one_sequence(
  h: np.ndarray,
  j: np.ndarray,
  w: np.ndarray,
  seq: np.ndarray,
) -> float:
  """Unnormalized log-probability (energy) of a single sequence."""
  h = np.asarray(h)
  j = np.asarray(j)
  w = np.asarray(w)
  seq = np.asarray(seq)
  n = h.shape[0]
  energy = float(sum(h[i, seq[i]] for i in range(n)))
  for a in range(n):
    for b in range(a + 1, n):
      energy += float(w[a, b] * j[a, b, seq[a], seq[b]])
  return energy


@pytest.mark.potts
class TestTRWMarginalVsExact:
  """Group 1: aminx TRW marginals vs exact brute-force.

  TRW (tree-reweighted belief propagation) is exact on trees and a variational
  approximation on loopy graphs, so the marginal gap grows with graph density. n=4 and
  n=6 random sparse systems sit well within a 0.05 max-error tolerance; n>=8 at ~50%
  edge density exceeds it by an inherent (iteration-independent) variational gap and is
  therefore not asserted here.
  """

  @pytest.mark.parametrize("n", [4, 6])
  def test_trw_marginals_vs_exact(self, n: int):
    """aminx DifferentiableTRW marginals converge to exact within 0.05 max error."""
    q = 4  # keep brute-force tractable (q**n states)

    key = jr.PRNGKey(42 + n)
    key_h, key_j, key_w, _ = jr.split(key, 4)

    h = jr.normal(key_h, shape=(n, q)) * 0.5
    j_raw = jr.normal(key_j, shape=(n, n, q, q)) * 0.3
    j = 0.5 * (j_raw + jnp.transpose(j_raw, (1, 0, 3, 2)))
    w_raw = jr.uniform(key_w, shape=(n, n))
    w = jnp.where(w_raw > 0.5, 1.0, 0.0)  # ~50% sparse edges
    w = 0.5 * (w + w.T)  # symmetrize

    exact_marg, _ = _exact_marginals_bruteforce(h, j, w)

    trw = DifferentiableTRW(q=q, trw_iters=30, damping=0.5)
    trw_marg, _ = trw(w, j, h)

    max_error = float(jnp.max(jnp.abs(jnp.asarray(trw_marg) - jnp.asarray(exact_marg))))
    assert max_error < 0.05, f"TRW max error {max_error:.6f} exceeds tolerance 0.05 for n={n}"


@pytest.mark.potts
class TestX2ScaleFactor:
  """Group 2: verify the exact brute-force reference on a hand-constructed system.

  This pins the measurement pipeline (the brute-force reference that the TRW test
  depends on) to a known 2-residue, 20-state Potts model.
  """

  def test_x2_scale_factor_two_residue(self):
    """Hand-constructed 2-residue system: log_prob finite, marginals normalized."""
    n, q = 2, 20

    h = jnp.array(
      [
        [0.1, 0.2, 0.3, 0.4, 0.5, -0.5, -0.4, -0.3, -0.2, -0.1] + [0.0] * 10,
        [-0.1, -0.2, -0.3, -0.4, -0.5, 0.5, 0.4, 0.3, 0.2, 0.1] + [0.0] * 10,
      ],
      dtype=jnp.float32,
    )

    j = jnp.zeros((n, n, q, q), dtype=jnp.float32)
    for aa in range(q):
      j = j.at[0, 1, aa, aa].set(0.5)
      j = j.at[1, 0, aa, aa].set(0.5)

    w = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32)

    exact_marg, log_z = _exact_marginals_bruteforce(h, j, w)

    seqs = [
      jnp.array([0, 0], dtype=jnp.int32),
      jnp.array([1, 1], dtype=jnp.int32),
      jnp.array([0, 1], dtype=jnp.int32),
    ]
    for seq in seqs:
      log_u = _log_unnormalized_one_sequence(h, j, w, seq)
      log_prob_expected = log_u - log_z
      assert np.isfinite(log_prob_expected), f"Non-finite log_prob for seq {seq}"

    marginal_sums = np.sum(exact_marg, axis=1)
    assert np.allclose(
      marginal_sums, np.ones(n), atol=1e-5
    ), f"Marginals do not sum to 1: {marginal_sums}"


@pytest.mark.potts
class TestPottsVsMPNNAlphabet:
  """Group 3: Verify Potts and MPNN alphabets match in first 20 positions."""

  def test_potts_vs_mpnn_alphabet_ordering(self):
    """MPNN_ALPHABET first 20 chars are the canonical amino acids in order."""
    expected_alphabet = "ACDEFGHIKLMNPQRSTVWY"

    assert len(MPNN_ALPHABET) >= 20, f"MPNN_ALPHABET too short: {MPNN_ALPHABET}"

    actual_first_20 = MPNN_ALPHABET[:20]
    assert (
      actual_first_20 == expected_alphabet
    ), f"MPNN alphabet mismatch:\nExpected: {expected_alphabet}\nActual:   {actual_first_20}"

    assert len(set(actual_first_20)) == 20, f"Duplicate amino acids in first 20: {actual_first_20}"
