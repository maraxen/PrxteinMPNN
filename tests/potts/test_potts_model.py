"""Tests for aminx.potts.model.PottsModel."""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from aminx.model import ProteinFeatures
from aminx.potts.model import PottsModel, PottsParams, POTTS_ALPHABET


def _compute_edge_features(
    coords: Float[Array, "n 37 3"],
    mask: Float[Array, " n"],
    residue_index: Int[Array, " n"],
    chain_index: Int[Array, " n"],
    k_neighbors: int,
    key: PRNGKeyArray,
) -> tuple[Float[Array, "n k e"], Int[Array, "n k"]]:
    """Compute k-NN edge features and neighbor indices via ProteinFeatures.

    Helper for tests to convert old API inputs to new API format.
    """
    features = ProteinFeatures(
        node_features=128,
        edge_features=128,
        k_neighbors=k_neighbors,
        key=key,
    )

    edge_knn, nei, _, _ = features(
        key,
        coords,
        mask,
        residue_index,
        chain_index,
        backbone_noise=None,
    )

    return edge_knn, nei


def test_potts_model_construction(rng_key: PRNGKeyArray) -> None:
    """Test that PottsModel can be constructed with required parameters."""
    model = PottsModel(
        hidden_dim=64,
        num_aa=21,
        k_neighbors=16,
        edge_features_dim=128,
        trw_iters=15,
        key=rng_key,
    )

    # Verify static fields
    assert model.hidden_dim == 64
    assert model.num_aa == 21
    assert model.k_neighbors == 16


def test_potts_model_call(
    rng_key: PRNGKeyArray,
    model_inputs: dict,
) -> None:
    """Test that PottsModel.__call__ returns correct shapes."""
    n = int(model_inputs["mask"].sum())
    k_neighbors = 8  # Small k for fast testing

    model = PottsModel(
        hidden_dim=64,
        num_aa=21,
        k_neighbors=k_neighbors,
        edge_features_dim=128,
        trw_iters=2,
        key=rng_key,
    )

    # Compute edge features using ProteinFeatures
    edge_knn, nei = _compute_edge_features(
        model_inputs["structure_coordinates"],
        model_inputs["mask"],
        model_inputs["residue_index"],
        model_inputs["chain_index"],
        k_neighbors,
        rng_key,
    )

    marginals, h, J, rho = model(
        edge_knn=edge_knn,
        nei=nei,
        mask=model_inputs["mask"],
    )

    # Check shapes
    assert marginals.shape == (n, 21)
    assert h.shape == (n, 21)
    assert J.shape == (n, n, 21, 21)
    assert rho.shape == (n, n)

    # Check values are finite
    assert jnp.all(jnp.isfinite(marginals))
    assert jnp.all(jnp.isfinite(h))
    assert jnp.all(jnp.isfinite(J))
    assert jnp.all(jnp.isfinite(rho))


def test_potts_model_infer_params(
    rng_key: PRNGKeyArray,
    model_inputs: dict,
) -> None:
    """Test that infer_params returns PottsParams namedtuple."""
    k_neighbors = 8

    model = PottsModel(
        hidden_dim=64,
        num_aa=21,
        k_neighbors=k_neighbors,
        edge_features_dim=128,
        trw_iters=2,
        key=rng_key,
    )

    # Compute edge features using ProteinFeatures
    edge_knn, nei = _compute_edge_features(
        model_inputs["structure_coordinates"],
        model_inputs["mask"],
        model_inputs["residue_index"],
        model_inputs["chain_index"],
        k_neighbors,
        rng_key,
    )

    params = model.infer_params(
        edge_knn=edge_knn,
        nei=nei,
        mask=model_inputs["mask"],
    )

    # Check that we get a PottsParams namedtuple with expected fields
    assert hasattr(params, "marginals")
    assert hasattr(params, "h")
    assert hasattr(params, "J")
    assert hasattr(params, "rho")
    assert hasattr(params, "W")

    n = int(model_inputs["mask"].sum())
    assert params.marginals.shape == (n, 21)
    assert params.h.shape == (n, 21)
    assert params.J.shape == (n, n, 21, 21)
    assert params.rho.shape == (n, n)
    assert params.W.shape == (n, n)


def test_potts_alphabet_constant() -> None:
    """Test that POTTS_ALPHABET is defined with q=21."""
    assert len(POTTS_ALPHABET) == 21
    assert POTTS_ALPHABET == "ACDEFGHIKLMNPQRSTVWYX"


def test_potts_model_training_guard_fori_raises(rng_key: PRNGKeyArray) -> None:
    """Test that PottsModel raises ValueError with trw_loop=fori and training=True."""
    try:
        from aminx.potts._trw_spec import PottsTRWRunSpec
    except ImportError:
        pytest.skip("aminx not installed")

    fori_spec = PottsTRWRunSpec(
        trw_loop="fori",
        rho_backend="dense_pinv",
    )

    with pytest.raises(
        ValueError,
        match="trw_loop=fori is unsafe for training",
    ):
        PottsModel(
            hidden_dim=64,
            num_aa=21,
            k_neighbors=8,
            edge_features_dim=128,
            trw_iters=2,
            key=rng_key,
            training=True,
            trw_spec=fori_spec,
        )


def test_potts_model_training_guard_fori_inference_ok(rng_key: PRNGKeyArray) -> None:
    """Test that PottsModel allows fori loop when training=False (inference)."""
    try:
        from aminx.potts._trw_spec import PottsTRWRunSpec
    except ImportError:
        pytest.skip("aminx not installed")

    fori_spec = PottsTRWRunSpec(
        trw_loop="fori",
        rho_backend="dense_pinv",
    )

    # Should not raise with training=False
    model = PottsModel(
        hidden_dim=64,
        num_aa=21,
        k_neighbors=8,
        edge_features_dim=128,
        trw_iters=2,
        key=rng_key,
        training=False,
        trw_spec=fori_spec,
    )

    assert model.training is False
    assert model.trw_spec.trw_loop == "fori"


def test_log_prob_with_partial_mask() -> None:
    """Test that log_prob handles partial masks correctly (per-residue masking).

    Unmasked residues should contribute to log_prob.
    Masked residues (mask=0) should contribute 0 to unary and pairwise terms.
    """
    n, q = 5, 21

    # Simple synthetic inputs
    seq = jnp.array([0, 1, 2, 3, 4], dtype=jnp.int32)
    h = jnp.ones((n, q))  # Unary potentials
    j = jnp.zeros((n, n, q, q))  # No pairwise
    w = jnp.eye(n)  # Only self-loops (will be zeroed by log_prob)

    # Mask: residues 0,1,2 unmasked; 3,4 masked
    mask = jnp.array([1.0, 1.0, 1.0, 0.0, 0.0])

    log_prob_val = PottsModel.log_prob(seq, h, j, w, mask)

    # Expected: only first 3 residues contribute unary terms
    # h[0,seq[0]] + h[1,seq[1]] + h[2,seq[2]] = 1.0 + 1.0 + 1.0 = 3.0
    # h[3,...] and h[4,...] should not contribute (masked)
    expected = 3.0

    assert jnp.isfinite(log_prob_val), "log_prob should be finite with partial mask"
    assert jnp.allclose(log_prob_val, expected, atol=1e-5), \
        f"Expected {expected}, got {log_prob_val}"


def test_log_prob_full_mask_still_works() -> None:
    """Test that log_prob with full mask=0 returns 0 (empty sum), not -inf."""
    n, q = 5, 21

    seq = jnp.array([0, 1, 2, 3, 4], dtype=jnp.int32)
    h = jnp.ones((n, q))
    j = jnp.zeros((n, n, q, q))
    w = jnp.zeros((n, n))
    mask = jnp.zeros(n)  # All masked

    log_prob_val = PottsModel.log_prob(seq, h, j, w, mask)

    # With all residues masked, unary and pairwise sums are 0
    # So log_prob should be 0, not -inf
    assert jnp.isfinite(log_prob_val), "log_prob with all masked should be 0, not -inf"
    assert jnp.allclose(log_prob_val, 0.0, atol=1e-5), \
        f"Expected 0.0, got {log_prob_val}"


def test_infer_params_returns_correct_w(
    rng_key: PRNGKeyArray,
    model_inputs: dict,
) -> None:
    """Test that infer_params correctly returns W adjacency matrix.

    Regression test: W should be computed from neighbor indices and mask,
    available without redundant ProteinFeatures calls.
    """
    k_neighbors = 8

    model = PottsModel(
        hidden_dim=64,
        num_aa=21,
        k_neighbors=k_neighbors,
        edge_features_dim=128,
        trw_iters=2,
        key=rng_key,
    )

    # Compute edge features using ProteinFeatures
    edge_knn, nei = _compute_edge_features(
        model_inputs["structure_coordinates"],
        model_inputs["mask"],
        model_inputs["residue_index"],
        model_inputs["chain_index"],
        k_neighbors,
        rng_key,
    )

    params = model.infer_params(
        edge_knn=edge_knn,
        nei=nei,
        mask=model_inputs["mask"],
    )

    # Verify W is returned and has sensible properties
    assert hasattr(params, "W")
    n = int(model_inputs["mask"].sum())
    assert params.W.shape == (n, n), f"Expected W shape ({n}, {n}), got {params.W.shape}"

    # W should be symmetric (undirected graph)
    assert jnp.allclose(params.W, params.W.T), "W should be symmetric"

    # W diagonal should be zero (no self-loops)
    diag = jnp.diag(params.W)
    assert jnp.allclose(diag, 0.0), "W diagonal should be zero (no self-loops)"

    # W should have binary or weighted values in [0, 1]
    assert jnp.all(params.W >= 0.0) and jnp.all(params.W <= 1.0), \
        "W values should be in [0, 1]"
