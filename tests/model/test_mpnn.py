"""Tests for the Aminx model."""
import chex
import jax
import jax.numpy as jnp

from aminx.inference.bundle_builder import build_inference_bundle
from aminx.inference.logits import make_stage_set
from aminx.inference import (
    score_conditional,
    score_unconditional,
    sample_autoregressive,
)
from aminx.model.mpnn import Aminx


class TestMPNN(chex.TestCase):
    def setUp(self):
        """Set up the model and input data for tests."""
        self.model_key = jax.random.PRNGKey(0)
        self.model = Aminx(
            node_features=128,
            edge_features=128,
            hidden_features=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            k_neighbors=30,
            key=self.model_key,
        )
        self.coords = jnp.ones((10, 4, 3))
        self.mask = jnp.ones((10,))
        self.residue_index = jnp.arange(10)
        self.chain_index = jnp.zeros((10,), dtype=jnp.int32)
        self.prng_key = jax.random.PRNGKey(1)

    @chex.variants(with_jit=True, without_jit=True, with_device=True)
    def test_score_unconditional(self):
        """Test unconditional scoring through inference API."""
        coords_batch = self.coords[None, ...]
        mask_batch = self.mask[None, ...]
        residue_index_batch = self.residue_index[None, ...]
        chain_index_batch = self.chain_index[None, ...]

        bundle, config = build_inference_bundle(
            coords=coords_batch,
            mask=mask_batch,
            residue_index=residue_index_batch,
            chain_index=chain_index_batch,
            mode="score_unconditional",
        )
        stage_set = make_stage_set()

        @self.variant
        def score_fn():
            return score_unconditional.kernel(
                self.model, self.prng_key, bundle, config, stage_set
            )

        logits = score_fn()
        chex.assert_shape(logits, (10, 21))
        chex.assert_type(logits, float)
        chex.assert_tree_all_finite(logits)

    @chex.variants(with_jit=True, without_jit=True, with_device=True)
    def test_score_conditional(self):
        """Test conditional scoring through inference API."""
        coords_batch = self.coords[None, ...]
        mask_batch = self.mask[None, ...]
        residue_index_batch = self.residue_index[None, ...]
        chain_index_batch = self.chain_index[None, ...]

        bundle, config = build_inference_bundle(
            coords=coords_batch,
            mask=mask_batch,
            residue_index=residue_index_batch,
            chain_index=chain_index_batch,
            sequence=jnp.zeros((10, 21)),
            mode="score_conditional",
        )
        stage_set = make_stage_set()

        @self.variant
        def score_fn():
            return score_conditional.kernel(
                self.model, self.prng_key, bundle, config, stage_set
            )

        logits = score_fn()
        chex.assert_shape(logits, (10, 21))
        chex.assert_type(logits, float)
        chex.assert_tree_all_finite(logits)

    def test_sample_autoregressive(self):
        """Test autoregressive sampling through inference API."""
        coords_batch = self.coords[None, ...]
        mask_batch = self.mask[None, ...]
        residue_index_batch = self.residue_index[None, ...]
        chain_index_batch = self.chain_index[None, ...]

        bundle, config = build_inference_bundle(
            coords=coords_batch,
            mask=mask_batch,
            residue_index=residue_index_batch,
            chain_index=chain_index_batch,
            mode="sample_autoregressive",
        )
        stage_set = make_stage_set()

        result = sample_autoregressive.kernel(
            self.model, self.prng_key, bundle, config, stage_set
        )

        chex.assert_shape(result.sequence, (10,))
        chex.assert_shape(result.logits, (10, 21))
        chex.assert_type(result.sequence, jnp.int32)
        chex.assert_type(result.logits, float)
        chex.assert_tree_all_finite(result.logits)

    def test_encoder_basic(self):
        """Test encoder output shape without bundle machinery."""
        node_f, edge_f, neighbors = self.model(
            self.coords,
            self.mask,
            self.residue_index,
            self.chain_index,
            prng_key=self.prng_key,
        )
        chex.assert_shape(node_f, (10, 128))
        # edge_f shape is (L, min(k_neighbors, L), edge_features)
        # Since L=10 and k_neighbors=30, it's (10, 10, 128)
        chex.assert_shape(edge_f, (10, 10, 128))
        chex.assert_shape(neighbors, (10, 10))
        chex.assert_tree_all_finite((node_f, edge_f))
