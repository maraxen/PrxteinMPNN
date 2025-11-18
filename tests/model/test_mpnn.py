"""Tests for the PrxteinMPNN model."""
import chex
import jax
import jax.numpy as jnp
from functools import partial
from prxteinmpnn.model.mpnn import PrxteinMPNN


class TestMPNN(chex.TestCase):
    def setUp(self):
        """Set up the model and input data for tests."""
        self.model_key = jax.random.PRNGKey(0)
        self.model = PrxteinMPNN(
            node_features=128,
            edge_features=128,
            hidden_features=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            k_neighbors=30,
            key=self.model_key,
        )
        self.input_data = {
            "structure_coordinates": jnp.ones((10, 4, 3)),
            "mask": jnp.ones((10,)),
            "residue_index": jnp.arange(10),
            "chain_index": jnp.zeros((10,), dtype=jnp.int32),
            "prng_key": jax.random.PRNGKey(1),
        }

    @chex.variants(with_jit=True, without_jit=True, with_device=True)
    def test_call_all_approaches(self):
        """Test the __call__ method with all decoding approaches."""

        @partial(
            self.variant, static_argnames=["decoding_approach", "multi_state_strategy"]
        )
        def call_fn(decoding_approach, multi_state_strategy, **kwargs):
            return self.model(
                **kwargs,
                decoding_approach=decoding_approach,
                multi_state_strategy=multi_state_strategy,
            )

        for decoding_approach in ["unconditional", "conditional", "autoregressive"]:
            seq, logits = call_fn(
                decoding_approach=decoding_approach,
                multi_state_strategy="max_min",
                **self.input_data,
            )

            chex.assert_shape(seq, (10, 21))
            chex.assert_shape(logits, (10, 21))
            chex.assert_type(seq, float)
            chex.assert_type(logits, float)
            chex.assert_tree_all_finite((seq, logits))

    def test_call_no_key(self):
        """Test the __call__ method without a prng_key."""
        self.input_data.pop("prng_key")
        seq, logits = self.model(**self.input_data, decoding_approach="unconditional")
        chex.assert_shape(seq, (10, 21))
        chex.assert_shape(logits, (10, 21))
        chex.assert_type(seq, float)
        chex.assert_type(logits, float)
        chex.assert_tree_all_finite((seq, logits))
