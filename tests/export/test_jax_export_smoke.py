"""Smoke test for JAX export of score_sequence function.

This test verifies that the score_sequence function can be exported to
a serializable MLIR representation using jax.export.export, a prerequisite
for downstream IREE-WASM and StableHLO export targets.
"""

import functools
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

pytest.importorskip("jax.export")

from aminx.scoring.score import make_score_fn
from aminx.types.protocols import ModelProtocol


class MockDecoder(eqx.Module):
    """Minimal mock decoder."""

    def call_conditional(self, *args, **kwargs):
        """Return zero logits."""
        return jnp.zeros(128)


class MockModel(eqx.Module):
    """Minimal mock model satisfying ModelProtocol.

    Returns zero-filled tensors of the correct shapes to avoid actual
    featurization and encoding during export testing.
    """

    w_s_embed: eqx.nn.Embedding
    w_out: eqx.nn.Linear
    decoder: MockDecoder

    def __init__(self):
        """Initialize with minimal required layers."""
        key = jax.random.PRNGKey(0)
        key1, key2 = jax.random.split(key)

        # Required by ModelProtocol
        self.w_s_embed = eqx.nn.Embedding(num_embeddings=21, embedding_size=128, key=key1)
        self.w_out = eqx.nn.Linear(128, 21, key=key2)
        self.decoder = MockDecoder()

    def __call__(
        self,
        key: jax.Array = None,
        bundle: Any = None,
        config: Any = None,
        **kwargs: Any,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return mock node_features, edge_features, edge_indices.

        Accepts both call signatures:
        - New API: (key, bundle, config)
        - Old API used by score_conditional: (key, coords, mask, ...)

        Returns:
            Tuple of (node_features, edge_features, edge_indices) where:
            - node_features: shape (L, 128) — feature vector per residue
            - edge_features: shape (L, 128) — edge features
            - edge_indices: shape (2, L) — edge index pairs
        """
        # Handle both API signatures to determine L
        if bundle is not None and hasattr(bundle, "geometry"):
            # New API: bundle is InferenceBundle
            L = bundle.geometry.mask.shape[-1]
        elif "mask" in kwargs:
            # Old API: mask passed as kwarg from score_conditional
            mask = kwargs.get("mask")
            L = mask.shape[-1] if mask.ndim > 0 else 4
        else:
            # Fallback
            L = 4

        # Mock outputs
        node_features = jnp.zeros((L, 128), dtype=jnp.float32)
        edge_features = jnp.zeros((L, 128), dtype=jnp.float32)
        edge_indices = jnp.zeros((2, L), dtype=jnp.int32)

        return node_features, edge_features, edge_indices


def test_jax_export_score_sequence_basic():
    """Test that score_sequence can be exported with static args bound."""
    from unittest.mock import patch
    from aminx.inference import score_conditional

    # Create mock model
    mock_model = MockModel()

    # Make score function (already JIT-compiled with static args)
    score_sequence = make_score_fn(mock_model)

    # Define abstract input shapes for L=4
    # Note: sequence should be one-hot (L, V) format for the current scorer implementation
    L = 4
    V = 21  # vocabulary size (amino acids)

    abstract_inputs = (
        jax.ShapeDtypeStruct(shape=(2,), dtype=jnp.uint32),  # prng_key
        jax.ShapeDtypeStruct(shape=(L, V), dtype=jnp.float32),  # sequence (one-hot)
        jax.ShapeDtypeStruct(shape=(L, 4, 3), dtype=jnp.float32),  # structure_coordinates
        jax.ShapeDtypeStruct(shape=(L,), dtype=jnp.int32),  # mask
        jax.ShapeDtypeStruct(shape=(L,), dtype=jnp.int32),  # residue_index
        jax.ShapeDtypeStruct(shape=(L,), dtype=jnp.int32),  # chain_index
    )

    # Mock score_conditional.kernel to return dummy logits
    def mock_kernel(model, prng_key, bundle, config, stage_set):
        """Return dummy logits of shape (L, 21)."""
        L = bundle.geometry.mask.shape[-1]
        return jnp.zeros((L, 21), dtype=jnp.float32)

    # Create wrapper that binds static args
    def _export_wrapper(prng_key, sequence, structure_coordinates, mask, residue_index, chain_index):
        # Patch score_conditional.kernel for this call
        with patch.object(score_conditional, "kernel", mock_kernel):
            return score_sequence(
                prng_key,
                sequence,
                structure_coordinates,
                mask,
                residue_index,
                chain_index,
                multi_state_strategy="arithmetic_mean",
                use_rolling_state=False,
            )

    # JIT the wrapper
    jitted_wrapper = jax.jit(_export_wrapper)

    # Export to MLIR
    exported = jax.export.export(jitted_wrapper)(*abstract_inputs)

    # Verify we got a serialized MLIR module
    assert exported.mlir_module_serialized is not None
    assert isinstance(exported.mlir_module_serialized, bytes)
    assert len(exported.mlir_module_serialized) > 0


def test_jax_export_score_sequence_with_optional_args():
    """Test export with additional optional static bindings."""
    from unittest.mock import patch
    from aminx.inference import score_conditional

    mock_model = MockModel()
    score_sequence = make_score_fn(mock_model)

    L = 4
    V = 21

    abstract_inputs = (
        jax.ShapeDtypeStruct(shape=(2,), dtype=jnp.uint32),
        jax.ShapeDtypeStruct(shape=(L, V), dtype=jnp.float32),  # sequence (one-hot)
        jax.ShapeDtypeStruct(shape=(L, 4, 3), dtype=jnp.float32),
        jax.ShapeDtypeStruct(shape=(L,), dtype=jnp.int32),
        jax.ShapeDtypeStruct(shape=(L,), dtype=jnp.int32),
        jax.ShapeDtypeStruct(shape=(L,), dtype=jnp.int32),
    )

    # Mock score_conditional.kernel
    def mock_kernel(model, prng_key, bundle, config, stage_set):
        """Return dummy logits of shape (L, 21)."""
        L = bundle.geometry.mask.shape[-1]
        return jnp.zeros((L, 21), dtype=jnp.float32)

    def _export_wrapper(prng_key, sequence, structure_coordinates, mask, residue_index, chain_index):
        with patch.object(score_conditional, "kernel", mock_kernel):
            return score_sequence(
                prng_key,
                sequence,
                structure_coordinates,
                mask,
                residue_index,
                chain_index,
                multi_state_strategy="geometric_mean",
                use_rolling_state=False,
            )

    jitted_wrapper = jax.jit(_export_wrapper)
    exported = jax.export.export(jitted_wrapper)(*abstract_inputs)

    assert exported.mlir_module_serialized is not None
    assert isinstance(exported.mlir_module_serialized, bytes)
    assert len(exported.mlir_module_serialized) > 0


def test_jax_export_returns_callable():
    """Test that exported module provides a callable interface."""
    from unittest.mock import patch
    from aminx.inference import score_conditional

    mock_model = MockModel()
    score_sequence = make_score_fn(mock_model)

    L = 4
    V = 21

    abstract_inputs = (
        jax.ShapeDtypeStruct(shape=(2,), dtype=jnp.uint32),
        jax.ShapeDtypeStruct(shape=(L, V), dtype=jnp.float32),  # sequence (one-hot)
        jax.ShapeDtypeStruct(shape=(L, 4, 3), dtype=jnp.float32),
        jax.ShapeDtypeStruct(shape=(L,), dtype=jnp.int32),
        jax.ShapeDtypeStruct(shape=(L,), dtype=jnp.int32),
        jax.ShapeDtypeStruct(shape=(L,), dtype=jnp.int32),
    )

    # Mock score_conditional.kernel
    def mock_kernel(model, prng_key, bundle, config, stage_set):
        """Return dummy logits of shape (L, 21)."""
        L = bundle.geometry.mask.shape[-1]
        return jnp.zeros((L, 21), dtype=jnp.float32)

    def _export_wrapper(prng_key, sequence, structure_coordinates, mask, residue_index, chain_index):
        with patch.object(score_conditional, "kernel", mock_kernel):
            return score_sequence(
                prng_key,
                sequence,
                structure_coordinates,
                mask,
                residue_index,
                chain_index,
                multi_state_strategy="arithmetic_mean",
                use_rolling_state=False,
            )

    jitted_wrapper = jax.jit(_export_wrapper)
    exported = jax.export.export(jitted_wrapper)(*abstract_inputs)

    # Exported module should be callable
    assert callable(exported.call)

    # Check output shapes via lowered module
    assert exported.mlir_module_serialized is not None
