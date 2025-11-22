import jax
import jax.numpy as jnp
import pytest
import equinox as eqx
from prxteinmpnn.model.features_direct import ProteinFeaturesDirect

def test_features_direct_shapes():
    key = jax.random.key(0)
    node_features = 128
    edge_features = 128
    k_neighbors = 5

    model = ProteinFeaturesDirect(
        node_features=node_features,
        edge_features=edge_features,
        k_neighbors=k_neighbors,
        key=key
    )

    N = 20
    # Using (N, 4, 3) for backbone coordinates (N, CA, C, O)
    # Some utilities might expect (N, 37, 3). Let's use (N, 37, 3) to be safe as per preprocess.py usage.
    coords = jax.random.normal(key, (N, 37, 3))
    mask = jnp.ones((N,), dtype=jnp.float32)
    residue_index = jnp.arange(N)
    chain_index = jnp.zeros((N,), dtype=jnp.int32)
    backbone_noise = 0.1

    edge_feats, neighbor_indices, new_key = model(
        key, coords, mask, residue_index, chain_index, backbone_noise
    )

    # Expected output shapes:
    # edge_features: (N, K, edge_features)
    # neighbor_indices: (N, K)

    assert edge_feats.shape == (N, k_neighbors, edge_features)
    assert neighbor_indices.shape == (N, k_neighbors)

    # Check determinism with same key
    edge_feats2, _, _ = model(
        key, coords, mask, residue_index, chain_index, backbone_noise
    )
    assert jnp.allclose(edge_feats, edge_feats2)

def test_features_direct_jit():
    key = jax.random.key(1)
    model = ProteinFeaturesDirect(
        node_features=64,
        edge_features=64,
        k_neighbors=5,
        key=key
    )

    N = 10
    coords = jax.random.normal(key, (N, 37, 3))
    mask = jnp.ones((N,), dtype=jnp.int32)
    residue_index = jnp.arange(N)
    chain_index = jnp.zeros((N,), dtype=jnp.int32)

    # JIT the model call
    @jax.jit
    def forward(k, c, m, r, ch):
        return model(k, c, m, r, ch, 0.0)

    edge_feats, _, _ = forward(key, coords, mask, residue_index, chain_index)
    assert edge_feats.shape == (N, 5, 64)
