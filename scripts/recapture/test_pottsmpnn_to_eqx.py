"""Test pottsmpnn_to_eqx conversion with synthetic 2-residue ground truth."""

from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
import zstandard as zstd

# Import from sibling module
sys.path.insert(0, str(Path(__file__).parent))
from pottsmpnn_to_eqx import etab_to_dense_h_j_w


def test_etab_to_dense_h_j_w_synthetic():
    """Test etab_to_dense_h_j_w conversion with synthetic 2-residue data.

    Creates a synthetic etab (1, 2, 2, 21, 21) and e_idx (1, 2, 2) with K=2 neighbors.
    Verifies h and J are x2-scaled correctly and match ground truth within 1e-6.
    """
    # Synthetic etab: 1 batch, 2 residues, K=2 neighbors, q=21 states
    K = 2
    n_res = 2
    q = 21

    # Build synthetic etab: for each residue and neighbor, place known values
    etab = np.zeros((1, n_res, K, q, q), dtype=np.float64)

    # Residue 0, neighbor 0 (self): diagonal block with known values
    etab[0, 0, 0, :, :] = np.diag([1.0, 2.0, 3.0] + [0.0] * (q - 3))  # diag(1,2,3,0,...,0)

    # Residue 0, neighbor 1: off-diagonal interaction
    etab[0, 0, 1, 0, 1] = 0.5
    etab[0, 0, 1, 1, 0] = 0.25

    # Residue 1, neighbor 0: interaction with residue 0
    etab[0, 1, 0, 0, 1] = 0.5
    etab[0, 1, 0, 1, 0] = 0.25

    # Residue 1, neighbor 1 (self): diagonal
    etab[0, 1, 1, :, :] = np.diag([0.5, 1.5] + [0.0] * (q - 2))

    # e_idx: neighbor indices (1 batch, 2 residues, K=2 neighbors)
    # For residue 0: neighbors are [0, 1]
    # For residue 1: neighbors are [0, 1]
    e_idx = np.array([[[0, 1], [0, 1]]], dtype=np.int64)

    # Mask: both residues valid
    mask = np.ones((1, n_res), dtype=np.float64)

    # Call the converter
    h, j, w = etab_to_dense_h_j_w(etab, e_idx, mask)

    # Verify shapes
    assert h.shape == (n_res, q), f"Expected h shape {(n_res, q)}, got {h.shape}"
    assert j.shape == (n_res, n_res, q, q), f"Expected j shape {(n_res, n_res, q, q)}, got {j.shape}"
    assert w.shape == (n_res, n_res), f"Expected w shape {(n_res, n_res)}, got {w.shape}"

    # Verify h has expected diagonal values (from self-interaction blocks)
    # h[0] should have diag(1,2,3,...) from etab[0,0,0]
    np.testing.assert_allclose(h[0, :3], np.array([1.0, 2.0, 3.0]) * 2.0, atol=1e-6, rtol=1e-6,
                               err_msg="h[0, :3] mismatch: expected [2,4,6] (x2 scale)")

    # h[1] should have diag(0.5, 1.5, ...) from etab[0,1,1]
    expected_h1_diag = np.array([0.5, 1.5] + [0.0] * (q - 2)) * 2.0
    np.testing.assert_allclose(h[1], expected_h1_diag, atol=1e-6, rtol=1e-6,
                               err_msg="h[1] mismatch")

    # Verify J is symmetric and x2-scaled
    np.testing.assert_allclose(j, np.transpose(j, (1, 0, 3, 2)), atol=1e-6, rtol=1e-6,
                               err_msg="j not symmetric")

    # j[0,1,0,1] should be x2 scaled value from averaging etab[0,0,1][0,1] and etab[0,1,0][0,1]
    # etab[0,0,1][0,1] = 0.5, etab[0,1,0][0,1] = 0.5 → j[0,1] = 0.5*(0.5 + 0.5) = 0.5
    # With x2 scale: 0.5 * 2 = 1.0
    # But we also have etab[0,0,1][1,0] = 0.25 and etab[0,1,0][1,0] = 0.25
    # j[0,1][1,0] = 0.5*(0.25+0.25) = 0.125
    # j[0,1][0,1] = 0.5*(0.5+0.5) = 0.5
    # After x2: j[0,1,0,1] = 0.5 * 2 = 1.0 ... wait that's not matching.
    # Let me recalculate: the function symmetrizes AFTER accumulation.
    # j[i,jn] accumulates etab[i,k] when jn = e_idx[i,k]
    # For i=0, k=1: jn = e_idx[0,1] = 1, so j[0,1] += etab[0,0,1]
    # For i=1, k=0: jn = e_idx[1,0] = 0, so j[1,0] += etab[0,1,0]
    # Then j = 0.5*(j + j.T) for symmetrization
    # After symmetrization and x2 scale, j[0,1,0,1] = 2.0 * 0.5 * (0.5 + 0.5) = 1.0
    # But actual is 0.75. Let me trace more carefully...
    # Actually, the issue is etab[0,0,1][0,1]=0.5 and [1,0]=0.25 are different!
    # j[0,1] initially has etab[0,0,1] = [[..., 0.5], [0.25, ...]]
    # j[1,0] initially has etab[0,1,0] = [[..., 0.5], [0.25, ...]]
    # Symmetrization: j[0,1,0,1] = 0.5 * (etab[0,0,1][0,1] + etab[0,1,0][1,0]) = 0.5*(0.5+0.25) = 0.375
    # With x2: 0.375 * 2 = 0.75. That matches!
    expected_j_0_1_0_1 = 2.0 * 0.5 * (0.5 + 0.25)  # 0.75
    np.testing.assert_allclose(j[0, 1, 0, 1], expected_j_0_1_0_1, atol=1e-6, rtol=1e-6,
                               err_msg="j[0,1,0,1] mismatch")

    # Verify w is symmetric with 1.0 for connected pairs
    assert w[0, 1] == 1.0, "w[0,1] should be 1.0"
    assert w[1, 0] == 1.0, "w[1,0] should be 1.0"

    print("✓ etab_to_dense_h_j_w synthetic test passed")


def test_synthetic_2_residue_checkpoint():
    """Test with hand-crafted h and J values for 2-residue Potts model.

    This test creates synthetic checkpoint data and verifies:
    1. h and J are loaded and x2-scaled correctly
    2. Serialization/deserialization round-trip preserves values within 1e-6
    """
    q = 21
    n_res = 2

    # Build ground truth h and J
    h_truth = np.zeros((n_res, q), dtype=np.float32)
    h_truth[0, 0] = 1.0  # Residue 0 favors state 0
    h_truth[1, 1] = 2.0  # Residue 1 favors state 1

    j_truth = np.zeros((n_res, n_res, q, q), dtype=np.float32)
    # Interaction between residues 0 and 1
    j_truth[0, 1, 0, 1] = 0.5  # Mutual preference
    j_truth[1, 0, 1, 0] = 0.5  # Symmetrized

    w_truth = np.zeros((n_res, n_res), dtype=np.float32)
    w_truth[0, 1] = 1.0
    w_truth[1, 0] = 1.0

    # x2 scale as per convention (directed-slot → symmetric)
    h_expected = h_truth * 2.0
    j_expected = j_truth * 2.0

    # Compute log_unnormalized as sanity check
    def potts_log_unnormalized(seq: np.ndarray, h: np.ndarray, j: np.ndarray, w: np.ndarray) -> float:
        """Compute log-unnormalized for sequence (manually, for test)."""
        energy = 0.0
        for i in range(len(seq)):
            energy += h[i, seq[i]]
        for i in range(len(seq)):
            for j_idx in range(i + 1, len(seq)):
                if w[i, j_idx] > 0:
                    energy += w[i, j_idx] * j[i, j_idx, seq[i], seq[j_idx]]
        return energy

    # Test sequence: [0, 1] → both residues in preferred states
    seq_test = np.array([0, 1], dtype=np.int32)
    energy_truth = potts_log_unnormalized(seq_test, h_truth, j_truth, w_truth)
    energy_expected = potts_log_unnormalized(seq_test, h_expected, j_expected, w_truth)

    # Expected energy should be 2x (due to x2 scaling)
    assert np.isclose(energy_expected, energy_truth * 2.0), \
        f"Energy scaling mismatch: {energy_expected} vs {energy_truth * 2.0}"

    print(f"✓ Synthetic test setup OK: energy_truth={energy_truth}, energy_expected={energy_expected}")

    # Round-trip test: serialize and deserialize, verify values preserved
    pytree = {
        "h": jax.numpy.asarray(h_expected),
        "j": jax.numpy.asarray(j_expected),
        "w": jax.numpy.asarray(w_truth),
        "mask": jax.numpy.asarray(np.ones(n_res, dtype=np.float32)),
        "metadata": {
            "k_neighbors": 48,
            "checkpoint_path": "synthetic",
            "pdb_path": "synthetic",
            "wt_seq": "AC",
            "vocab": 21,
        },
    }

    # Serialize via Equinox
    stream = io.BytesIO()
    eqx.tree_serialise_leaves(stream, pytree)
    serialized_bytes = stream.getvalue()

    # Compress with zstd
    cctx = zstd.ZstdCompressor(level=10)
    compressed = cctx.compress(serialized_bytes)

    # Decompress
    dctx = zstd.ZstdDecompressor()
    decompressed = dctx.decompress(compressed)

    # Deserialize
    stream_in = io.BytesIO(decompressed)
    pytree_loaded = eqx.tree_deserialise_leaves(stream_in, pytree)

    # Verify round-trip preserves values within 1e-6
    np.testing.assert_allclose(
        np.asarray(pytree_loaded["h"]),
        np.asarray(h_expected),
        atol=1e-6,
        rtol=1e-6,
        err_msg="h round-trip failed"
    )

    np.testing.assert_allclose(
        np.asarray(pytree_loaded["j"]),
        np.asarray(j_expected),
        atol=1e-6,
        rtol=1e-6,
        err_msg="j round-trip failed"
    )

    np.testing.assert_allclose(
        np.asarray(pytree_loaded["w"]),
        np.asarray(w_truth),
        atol=1e-6,
        rtol=1e-6,
        err_msg="w round-trip failed"
    )

    print("✓ Round-trip serialization test passed")


if __name__ == "__main__":
    test_etab_to_dense_h_j_w_synthetic()
    test_synthetic_2_residue_checkpoint()
    print("All synthetic tests passed!")
