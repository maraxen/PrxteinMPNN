"""Equivalence tests for MLP fusion (nested-vmap → flat-batch reshape).

Tests that verify the mathematical equivalence of replacing nested vmaps with
flat reshape + single vmap. Tests use simple eqx.nn.Linear modules to isolate
the vmap transformation logic.
"""
import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest


class TestEncoderMessageMLPFusion(chex.TestCase):
    """Equivalence test for encoder.py:216 double-vmap edge_message_mlp."""

    def setUp(self):
        """Set up a simple MLP for testing."""
        self.key = jax.random.PRNGKey(42)
        # Create a simple MLP to test the vmap transformation
        self.mlp = eqx.nn.MLP(
            in_size=96,
            out_size=32,
            width_size=64,
            depth=2,
            key=self.key,
        )
        # Create test input: (L, K, H_in) where L=sequence, K=neighbors
        self.L, self.K = 5, 8
        H_in = 96
        self.mlp_input = jax.random.normal(jax.random.PRNGKey(1), (self.L, self.K, H_in))

    def test_encoder_message_mlp_equivalence(self):
        """Verify nested vmap == reshape + flat vmap for edge_message_mlp."""
        # Reference: nested double-vmap
        nested_result = jax.vmap(jax.vmap(self.mlp))(self.mlp_input)
        chex.assert_shape(nested_result, (self.L, self.K, 32))

        # Fused: reshape to flat batch, vmap once, reshape back
        H_in = self.mlp_input.shape[-1]
        H_out = 32
        flat = self.mlp_input.reshape(-1, H_in)  # (L*K, H_in)
        fused_flat = jax.vmap(self.mlp)(flat)  # (L*K, H_out)
        fused_result = fused_flat.reshape(self.L, self.K, H_out)

        # Compare
        np.testing.assert_allclose(nested_result, fused_result, atol=1e-5)

    @pytest.mark.slow
    def test_encoder_message_mlp_downstream_ops(self):
        """Verify downstream shape compatibility after fusion."""
        # Reference
        nested_result = jax.vmap(jax.vmap(self.mlp))(self.mlp_input)

        # Simulate downstream: mask and sum
        mask_attend = jax.random.bernoulli(jax.random.PRNGKey(2), 0.8, (self.L, self.K))
        masked_nested = nested_result * mask_attend[..., None]
        sum_nested = jnp.sum(masked_nested, -2)

        # Fused version
        H_in = self.mlp_input.shape[-1]
        H_out = 32
        flat = self.mlp_input.reshape(-1, H_in)
        fused_flat = jax.vmap(self.mlp)(flat)
        fused_result = fused_flat.reshape(self.L, self.K, H_out)
        masked_fused = fused_result * mask_attend[..., None]
        sum_fused = jnp.sum(masked_fused, -2)

        np.testing.assert_allclose(sum_nested, sum_fused, atol=1e-5)


class TestTripleVmapFusion(chex.TestCase):
    """Equivalence test for decoder.py:489-493 triple-vmap fusion chain."""

    def setUp(self):
        """Set up three linear layers for the fusion test."""
        self.key = jax.random.PRNGKey(42)
        keys = jax.random.split(self.key, 3)

        # Three linear layers: in(64)->64->64->64 (matching w1->w2->w3 pattern)
        self.w1 = eqx.nn.Linear(64, 64, key=keys[0])
        self.w2 = eqx.nn.Linear(64, 64, key=keys[1])
        self.w3 = eqx.nn.Linear(64, 64, key=keys[2])

        # Input shape: (L, M, M, D) for decoder
        self.L, self.M, self.D = 5, 4, 64
        self.h_ev = jax.random.normal(jax.random.PRNGKey(1), (self.L, self.M, self.M, self.D))

    def test_triple_vmap_w1_w2_w3_chain_equivalence(self):
        """Verify triple vmap chain == flat reshape + sequential vmaps."""

        def _gelu(x):
            """GELU activation."""
            return jax.nn.gelu(x)

        # Reference: nested triple-vmap chain
        h_nested = jax.vmap(jax.vmap(jax.vmap(self.w1)))(self.h_ev)
        h_nested = _gelu(h_nested)
        h_nested = jax.vmap(jax.vmap(jax.vmap(self.w2)))(h_nested)
        h_nested = _gelu(h_nested)
        h_nested = jax.vmap(jax.vmap(jax.vmap(self.w3)))(h_nested)
        chex.assert_shape(h_nested, (self.L, self.M, self.M, 64))

        # Fused: flatten ALL leading dims to one batch axis
        L, M_rows, M_cols, D_in = self.h_ev.shape
        flat = self.h_ev.reshape(-1, D_in)  # (L*M*M, D_in)

        # Apply the chain on the flat batch
        h_flat = jax.vmap(self.w1)(flat)
        h_flat = _gelu(h_flat)
        h_flat = jax.vmap(self.w2)(h_flat)
        h_flat = _gelu(h_flat)
        h_flat = jax.vmap(self.w3)(h_flat)

        # Reshape back
        h_fused = h_flat.reshape(L, M_rows, M_cols, 64)

        np.testing.assert_allclose(h_nested, h_fused, atol=1e-5)

    @pytest.mark.slow
    def test_triple_vmap_fusion_downstream(self):
        """Verify fused chain handles downstream operations correctly."""

        def _gelu(x):
            return jax.nn.gelu(x)

        # Reference
        h_nested = jax.vmap(jax.vmap(jax.vmap(self.w1)))(self.h_ev)
        h_nested = _gelu(h_nested)
        h_nested = jax.vmap(jax.vmap(jax.vmap(self.w2)))(h_nested)
        h_nested = _gelu(h_nested)
        h_nested = jax.vmap(jax.vmap(jax.vmap(self.w3)))(h_nested)

        # Simulate downstream: mask and sum
        mask_attend = jax.random.bernoulli(jax.random.PRNGKey(2), 0.8, (self.L, self.M, self.M))
        masked_nested = jnp.expand_dims(mask_attend, -1) * h_nested
        sum_nested = jnp.sum(masked_nested, axis=-2)

        # Fused
        L, M_rows, M_cols, D_in = self.h_ev.shape
        flat = self.h_ev.reshape(-1, D_in)
        h_flat = jax.vmap(self.w1)(flat)
        h_flat = _gelu(h_flat)
        h_flat = jax.vmap(self.w2)(h_flat)
        h_flat = _gelu(h_flat)
        h_flat = jax.vmap(self.w3)(h_flat)
        h_fused = h_flat.reshape(L, M_rows, M_cols, 64)

        masked_fused = jnp.expand_dims(mask_attend, -1) * h_fused
        sum_fused = jnp.sum(masked_fused, axis=-2)

        np.testing.assert_allclose(sum_nested, sum_fused, atol=1e-5)


class TestDoubleVmapWOut(chex.TestCase):
    """Equivalence test for decode/_kernel.py:108 double-vmap w_out."""

    def setUp(self):
        """Set up w_out layer."""
        self.key = jax.random.PRNGKey(42)
        # w_out: node_features -> 21 (vocab size)
        self.w_out = eqx.nn.Linear(32, 21, key=self.key)
        # Input shape: (S, L, H) where S=samples, L=sequence, H=hidden
        self.S, self.L, self.H = 3, 5, 32
        self.decoded = jax.random.normal(jax.random.PRNGKey(1), (self.S, self.L, self.H))

    def test_w_out_double_vmap_equivalence(self):
        """Verify nested vmap == reshape + flat vmap for w_out."""
        nested_result = jax.vmap(jax.vmap(self.w_out, in_axes=0), in_axes=0)(self.decoded)
        chex.assert_shape(nested_result, (self.S, self.L, 21))

        # Fused
        S, L, H_in = self.decoded.shape
        H_out = 21
        flat = self.decoded.reshape(-1, H_in)  # (S*L, H_in)
        fused_flat = jax.vmap(self.w_out)(flat)  # (S*L, H_out)
        fused_result = fused_flat.reshape(S, L, H_out)

        np.testing.assert_allclose(nested_result, fused_result, atol=1e-5)
