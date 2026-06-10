"""Tests for aminx.potts.model.PottsModel."""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from aminx.potts.model import PottsModel, PottsParams, POTTS_ALPHABET


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

    model = PottsModel(
        hidden_dim=64,
        num_aa=21,
        k_neighbors=8,  # Small k for fast testing
        edge_features_dim=128,
        trw_iters=2,
        key=rng_key,
    )

    marginals, h, J, rho = model(
        key=rng_key,
        coords=model_inputs["structure_coordinates"],
        mask=model_inputs["mask"],
        residue_index=model_inputs["residue_index"],
        chain_index=model_inputs["chain_index"],
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
    model = PottsModel(
        hidden_dim=64,
        num_aa=21,
        k_neighbors=8,
        edge_features_dim=128,
        trw_iters=2,
        key=rng_key,
    )

    params = model.infer_params(
        key=rng_key,
        coords=model_inputs["structure_coordinates"],
        mask=model_inputs["mask"],
        residue_index=model_inputs["residue_index"],
        chain_index=model_inputs["chain_index"],
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
