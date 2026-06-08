"""COMP-6: Tests for make_encode_fn in inference/encode.py.

RED phase — tests fail until:
  - inference/encode.py exists and exports make_encode_fn
  - make_encode_fn(model, use_rolling_state=False) -> EncodeFn
  - EncodeFn returns EncoderOutput
  - use_rolling_state flag controls scan vs vmap over S
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# 1. Module and import contract
# ---------------------------------------------------------------------------

def test_encode_module_importable():
    """inference.encode module is importable."""
    import aminx.inference.encode  # noqa: F401


def test_make_encode_fn_importable():
    """make_encode_fn is importable from inference.encode."""
    from aminx.inference.encode import make_encode_fn  # noqa: F401


def test_make_encode_fn_has_use_rolling_state_param():
    """make_encode_fn accepts a use_rolling_state keyword argument."""
    import inspect
    from aminx.inference.encode import make_encode_fn

    sig = inspect.signature(make_encode_fn)
    assert "use_rolling_state" in sig.parameters


# ---------------------------------------------------------------------------
# 2. Returned callable contract
# ---------------------------------------------------------------------------

def test_make_encode_fn_returns_callable():
    """make_encode_fn(model) returns a callable."""
    from aminx.inference.encode import make_encode_fn

    # A minimal duck-type model
    import equinox as eqx

    class DummyModel(eqx.Module):
        pass

    fn = make_encode_fn(DummyModel(), use_rolling_state=False)
    assert callable(fn)


# ---------------------------------------------------------------------------
# 3. use_rolling_state flag is honoured at construction time
# ---------------------------------------------------------------------------

def test_make_encode_fn_records_rolling_state_flag():
    """make_encode_fn stores use_rolling_state so it can be inspected."""
    from aminx.inference.encode import make_encode_fn
    import equinox as eqx

    class DummyModel(eqx.Module):
        pass

    fn_vmap = make_encode_fn(DummyModel(), use_rolling_state=False)
    fn_scan = make_encode_fn(DummyModel(), use_rolling_state=True)

    # They must be distinct objects (not the same closure sharing state)
    assert fn_vmap is not fn_scan


# ---------------------------------------------------------------------------
# 4. Scan path (use_rolling_state=True) with EncoderOutput contract
# ---------------------------------------------------------------------------

def test_make_encode_fn_rolling_state_returns_encoder_output():
    """make_encode_fn(use_rolling_state=True) returns EncoderOutput with mask populated."""
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    from aminx.inference.encode import make_encode_fn
    from aminx.types.encodings import EncoderOutput
    from aminx.types.bundles import GeometryBundle, LigandBundle, InferenceBundle, WaveScheduleBundle
    from aminx.types.configs import InferenceConfig

    # Minimal duck-type model
    class DummyModel(eqx.Module):
        def __call__(self, coords, mask, residue_index, chain_index, **kwargs):
            # Return (node_features, edge_features, neighbor_indices)
            L = coords.shape[-2]
            H = 32
            return (
                jnp.zeros((L, H)),  # node_features
                jnp.zeros((L, 10, H)),  # edge_features
                jnp.zeros((L, 10), dtype=jnp.int32),  # neighbor_indices
            )

        def features(self, k, coords, mask, ri, ci, noise, **kwargs):
            # Legacy interface for backward compat (unused now)
            L = coords.shape[-2]
            H = 32
            return (
                jnp.zeros((L, 10, H)),  # edge_features
                jnp.zeros((L, 10), dtype=jnp.int32),  # neighbor_indices
                jnp.zeros((L, H)),  # node_features
                None,  # unused
            )

        def encoder(self, edge_f, edge_i, mask, initial_node_features=None, key=None):
            return initial_node_features, edge_f  # Minimal: return inputs

    # Build minimal bundle
    S = 2
    L = 4
    geo = GeometryBundle(
        coords=jnp.zeros((S, L, 4, 3)),
        mask=jnp.ones((S, L)),
        residue_index=jnp.arange(L)[None, :].repeat(S, axis=0),
        chain_index=jnp.zeros((S, L), dtype=jnp.int32),
        n_states=S,
        n_canonical=L,
        n_flat=L * S,
        structure_mapping=None,
    )
    lig = LigandBundle(ligand_coords=jnp.zeros((S, 0, 14, 3)), ligand_atom_types=jnp.zeros((S, 0, 14), dtype=jnp.int32), ligand_mask=jnp.zeros((S, 0, 14)))
    wave = WaveScheduleBundle.empty(L)
    bundle = InferenceBundle(
        geometry=geo,
        ligand=lig,
        wave=wave,
        backbone_noise=jnp.array(0.0),
        conditioning=None,
    )
    config = InferenceConfig(
        backbone_noise_mode="direct",
        inference=True,
    )

    # Create encode_fn with use_rolling_state=True
    model = DummyModel()
    encode_fn = make_encode_fn(model, use_rolling_state=True)

    # Call it
    key = jax.random.PRNGKey(0)
    output = encode_fn(bundle, key, config)

    # Verify output is EncoderOutput
    assert isinstance(output, EncoderOutput)
    # Verify mask is populated (scan path includes mask in output)
    assert output.mask is not None
    assert output.mask.shape == (S, L)
    assert jnp.allclose(output.mask, jnp.ones((S, L)))


def test_make_encode_fn_scan_vs_vmap_both_return_encoder_output():
    """Both scan and vmap paths return valid EncoderOutput with same structure."""
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    from aminx.inference.encode import make_encode_fn
    from aminx.types.encodings import EncoderOutput
    from aminx.types.bundles import GeometryBundle, LigandBundle, InferenceBundle, WaveScheduleBundle
    from aminx.types.configs import InferenceConfig

    class DummyModel(eqx.Module):
        def __call__(self, coords, mask, residue_index, chain_index, **kwargs):
            # Return (node_features, edge_features, neighbor_indices)
            L = coords.shape[-2]
            H = 32
            return (
                jnp.zeros((L, H)),  # node_features
                jnp.zeros((L, 10, H)),  # edge_features
                jnp.zeros((L, 10), dtype=jnp.int32),  # neighbor_indices
            )

        def features(self, k, coords, mask, ri, ci, noise, **kwargs):
            L = coords.shape[-2]
            H = 32
            return (
                jnp.zeros((L, 10, H)),
                jnp.zeros((L, 10), dtype=jnp.int32),
                jnp.zeros((L, H)),
                None,
            )

        def encoder(self, edge_f, edge_i, mask, initial_node_features=None, key=None):
            return initial_node_features, edge_f

    S = 2
    L = 4
    geo = GeometryBundle(
        coords=jnp.zeros((S, L, 4, 3)),
        mask=jnp.ones((S, L)),
        residue_index=jnp.arange(L)[None, :].repeat(S, axis=0),
        chain_index=jnp.zeros((S, L), dtype=jnp.int32),
        n_states=S,
        n_canonical=L,
        n_flat=L * S,
        structure_mapping=None,
    )
    lig = LigandBundle(ligand_coords=jnp.zeros((S, 0, 14, 3)), ligand_atom_types=jnp.zeros((S, 0, 14), dtype=jnp.int32), ligand_mask=jnp.zeros((S, 0, 14)))
    wave = WaveScheduleBundle.empty(L)
    bundle = InferenceBundle(
        geometry=geo,
        ligand=lig,
        wave=wave,
        backbone_noise=jnp.array(0.0),
        conditioning=None,
    )
    config_scan = InferenceConfig(
        backbone_noise_mode="direct",
        inference=True,
    )
    config_vmap = InferenceConfig(
        backbone_noise_mode="direct",
        inference=True,
    )

    model = DummyModel()
    key = jax.random.PRNGKey(0)

    # Scan path
    encode_scan = make_encode_fn(model, use_rolling_state=True)
    out_scan = encode_scan(bundle, key, config_scan)

    # Vmap path
    encode_vmap = make_encode_fn(model, use_rolling_state=False)
    out_vmap = encode_vmap(bundle, key, config_vmap)

    # Both are EncoderOutput
    assert isinstance(out_scan, EncoderOutput)
    assert isinstance(out_vmap, EncoderOutput)

    # Both have mask populated
    assert out_scan.mask is not None
    assert out_vmap.mask is not None

    # Shapes match
    assert out_scan.node_features.shape == out_vmap.node_features.shape
    assert out_scan.edge_features.shape == out_vmap.edge_features.shape
    assert out_scan.neighbor_indices.shape == out_vmap.neighbor_indices.shape

