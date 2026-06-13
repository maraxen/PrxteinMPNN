"""Test TRW marginals correctness against brute-force exact computation.

Tests DifferentiableTRW.marginals against exact enumeration over q^n configurations.
For small systems (n <= 12), brute-force enumeration is tractable (~16M configs for q=4, n=12).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Float, Array

from aminx.potts._trw import DifferentiableTRW


def exact_marginals(
    h: Float[Array, "n q"],
    J: Float[Array, "n n q q"],
    mask: Float[Array, " n"],
    q: int = 4,
) -> Float[Array, "n q"]:
    """Compute exact site marginals by enumeration over all q^n valid sequences.

    Enumerates all q^n configurations, computes unnormalized probability
    exp(-E(seq)) where E = -sum_i h[i,s[i]] - sum_{i<j} J[i,j,s[i],s[j]],
    and returns normalized site marginals p_i(a) = sum_{seq: seq[i]=a} p(seq).

    Args:
        h: Unary potentials, shape (N, q)
        J: Pairwise potentials, shape (N, N, q, q)
        mask: Residue mask, shape (N,), indicating which positions are valid
        q: Alphabet size (default 4 for toy tests)

    Returns:
        Site marginals, shape (N, q)

    Notes:
        - Complexity: O(q^n) where n = N (all positions)
        - For q=4, n=6: ~4096 configs
        - For q=4, n=12: ~16M configs (slow, ~5s)
        - Masked positions (mask=0) contribute uniform marginals
    """
    n = h.shape[0]
    h = jnp.asarray(h, dtype=jnp.float32)
    J = jnp.asarray(J, dtype=jnp.float32)
    mask = jnp.asarray(mask, dtype=jnp.float32)

    # Generate all q^n sequence indices as (n,) arrays
    # Enumerate sequences in base-q: idx = sum_i seq[i] * q^i
    total_seqs = q ** n

    # Pre-compute all sequences in vectorized form
    # sequences[s, i] = i-th position of sequence s (in base-q)
    seq_indices = jnp.arange(total_seqs, dtype=jnp.int32)

    def decode_seq(idx: Array) -> Array:
        """Decode sequence index (base-q) to sequence [s0, s1, ..., s_{n-1}]."""
        seq = jnp.zeros((n,), dtype=jnp.int32)
        for i in range(n):
            seq = seq.at[i].set((idx // (q ** i)) % q)
        return seq

    # Compute log-probability for each sequence via vmap
    def compute_log_prob(seq: Array) -> float:
        """Compute log P(seq) under the Potts model.

        The Potts model uses h and J as log-factors:
        log P(seq) = sum_i h[i,s[i]] + sum_{i<j} J[i,j,s[i],s[j]] + const
        """
        # Unary log-prob: sum_i h[i, s[i]]
        h_seq = jnp.take_along_axis(h, seq[..., None], axis=-1).squeeze(-1)  # (n,)
        h_masked = h_seq * mask
        unary_logp = jnp.sum(h_masked)

        # Pairwise log-prob: sum_{i,j} J[i,j,s[i],s[j]]
        # Reshape J to (n*n, q*q), then use advanced indexing
        j_flat = J.reshape(n * n, q * q)  # (n*n, q*q)

        # For each (i,j) pair, get index = seq[i] * q + seq[j]
        seq_ij_idx = seq[..., None] * q + seq[None, ...]  # (n, n)
        seq_ij_idx_flat = seq_ij_idx.reshape(-1)  # (n*n,)

        # Index into J_flat
        pair_indices = jnp.arange(n * n)  # (n*n,)
        j_seq_flat = j_flat[pair_indices, seq_ij_idx_flat]  # (n*n,)
        j_seq = j_seq_flat.reshape(n, n)  # (n, n)

        # Mask pairwise
        mask_pair = jnp.outer(mask, mask)
        j_masked = j_seq * mask_pair

        # Sum all pairwise (counts each edge twice, so divide by 2)
        pairwise_logp = 0.5 * jnp.sum(j_masked)

        # Total log-probability
        return unary_logp + pairwise_logp

    # Vectorize decode and log-prob computation over all sequences
    sequences = jax.vmap(decode_seq)(seq_indices)  # (q^n, n)
    log_probs = jax.vmap(compute_log_prob)(sequences)  # (q^n,)

    # Normalize via log-sum-exp
    max_log_prob = jnp.max(log_probs)
    log_probs_shifted = log_probs - max_log_prob
    unnorm_probs = jnp.exp(log_probs_shifted)
    probs = unnorm_probs / jnp.sum(unnorm_probs)  # (q^n,)

    # Extract marginals: for each position i, state a, sum probs[s] where seq[s,i]=a
    # marginals[i, a] = sum over s: probs[s] * (sequences[s, i] == a)
    marginals = jnp.zeros((n, q), dtype=jnp.float32)

    for a in range(q):
        # Check where sequences[:, :] == a for each position
        # sequences: (q^n, n), shape -> (q^n, n) boolean
        match = sequences == a  # (q^n, n)

        # For each position i: marginals[i, a] = sum_s probs[s] * match[s, i]
        for i in range(n):
            if mask[i] > 0:
                marginals = marginals.at[i, a].set(jnp.sum(probs * match[:, i]))
            else:
                # Masked: uniform contribution (skip, set to 0 for now)
                marginals = marginals.at[i, a].set(0.0)

    # For masked positions, set to uniform
    uniform_q = 1.0 / q
    marginals = jnp.where(
        mask[:, None] > 0, marginals, uniform_q
    )

    return marginals


class TestTRWMarginals:
    """Correctness tests for DifferentiableTRW marginals."""

    def test_trw_vs_exact_n6_q4(self) -> None:
        """TRW marginals vs exact marginals for n=6, q=4, random potentials."""
        key = jax.random.PRNGKey(42)
        n, q = 6, 4

        # Random h and J
        h_key, j_key = jax.random.split(key)
        h = jax.random.normal(h_key, (n, q), dtype=jnp.float32) * 0.5
        J = jax.random.normal(j_key, (n, n, q, q), dtype=jnp.float32) * 0.1

        # Symmetrize J
        J = 0.5 * (J + jnp.transpose(J, (1, 0, 3, 2)))

        # Full mask (all positions valid)
        mask = jnp.ones(n, dtype=jnp.float32)

        # Build adjacency: fully connected
        W = jnp.ones((n, n), dtype=jnp.float32) - jnp.eye(n)

        # Compute TRW marginals with many iterations for better convergence
        trw = DifferentiableTRW(q=q, trw_iters=50)
        marginals_trw, _ = trw(W, J, h)

        # Compute exact marginals
        marginals_exact = exact_marginals(h, J, mask, q=q)

        # Compare with tolerance atol=0.10, rtol=0.10
        # TRW is an approximation, so we use larger tolerance
        assert jnp.allclose(marginals_trw, marginals_exact, atol=0.10, rtol=0.10), (
            f"TRW marginals differ from exact:\n"
            f"max absolute diff: {jnp.max(jnp.abs(marginals_trw - marginals_exact))}\n"
            f"max relative diff: {jnp.max(jnp.abs(marginals_trw - marginals_exact) / (jnp.abs(marginals_exact) + 1e-6))}"
        )

    def test_trw_vs_exact_n4_q4_ferromagnet(self) -> None:
        """TRW marginals for ferromagnetic system: should recover uniform by symmetry."""
        key = jax.random.PRNGKey(123)
        n, q = 4, 4

        # Ferromagnet: J[i,j] = -1 for all i != j (encourages equal states)
        # h = 0 (no unary bias)
        h = jnp.zeros((n, q), dtype=jnp.float32)
        J = -jnp.ones((n, n, q, q), dtype=jnp.float32)
        J = J.at[jnp.arange(n), jnp.arange(n), :, :].set(0.0)  # Zero diagonal

        # Full mask
        mask = jnp.ones(n, dtype=jnp.float32)

        # Fully connected
        W = jnp.ones((n, n), dtype=jnp.float32) - jnp.eye(n)

        # Compute TRW marginals
        trw = DifferentiableTRW(q=q, trw_iters=20)
        marginals_trw, _ = trw(W, J, h)

        # Compute exact marginals
        marginals_exact = exact_marginals(h, J, mask, q=q)

        # By symmetry, each position should have uniform marginals (1/q each)
        uniform = jnp.ones((n, q), dtype=jnp.float32) / q

        # TRW should match exact, and exact should be close to uniform
        # Allow 0.02 deviation from uniform
        max_dev_exact = jnp.max(jnp.abs(marginals_exact - uniform))
        assert max_dev_exact < 0.02, (
            f"Exact marginals deviate from uniform by {max_dev_exact}: {marginals_exact}"
        )

        # TRW should also be close to uniform (via exact match above)
        assert jnp.allclose(marginals_trw, marginals_exact, atol=0.05, rtol=0.05)

    def test_trw_marginals_sum_to_one(self) -> None:
        """Marginals must sum to 1.0 across states for each position."""
        key = jax.random.PRNGKey(999)
        n, q = 5, 4

        # Random potentials
        h_key, j_key = jax.random.split(key)
        h = jax.random.normal(h_key, (n, q), dtype=jnp.float32)
        J = jax.random.normal(j_key, (n, n, q, q), dtype=jnp.float32)
        J = 0.5 * (J + jnp.transpose(J, (1, 0, 3, 2)))

        # Full mask
        mask = jnp.ones(n, dtype=jnp.float32)

        # Fully connected
        W = jnp.ones((n, n), dtype=jnp.float32) - jnp.eye(n)

        # Compute TRW marginals
        trw = DifferentiableTRW(q=q, trw_iters=15)
        marginals, _ = trw(W, J, h)

        # Marginals must sum to 1 across states
        marginal_sums = jnp.sum(marginals, axis=-1)

        assert jnp.allclose(marginal_sums[mask > 0], 1.0, atol=1e-4), (
            f"Marginal sums deviate from 1.0: {marginal_sums}"
        )
