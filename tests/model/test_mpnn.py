"""Tests for the PrxteinMPNN model."""
import chex
import jax
import jax.numpy as jnp
import pytest
import equinox as eqx
from prxteinmpnn.model.mpnn import PrxteinMPNN
from prxteinmpnn.utils.types import (
    StructureAtomicCoordinates,
    AlphaCarbonMask,
    ResidueIndex,
    ChainIndex,
    OneHotProteinSequence,
    AutoRegressiveMask,
    Logits,
)

@pytest.fixture
def model_key():
    """Provides a fixed JAX random key for model initialization."""
    return jax.random.PRNGKey(0)

@pytest.fixture
def model(model_key):
    """Initializes the PrxteinMPNN model with fixed parameters."""
    return PrxteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_features=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=30,
        key=model_key,
    )

@pytest.fixture
def input_data():
    """Provides dummy input data for testing."""
    key = jax.random.PRNGKey(1)
    return {
        "structure_coordinates": jnp.ones((10, 4, 3)),
        "mask": jnp.ones((10,)),
        "residue_index": jnp.arange(10),
        "chain_index": jnp.zeros((10,), dtype=jnp.int32),
        "prng_key": key,
    }

def test_call_unconditional(model):
    """Test the _call_unconditional method."""
    edge_features = jnp.ones((10, 30, 128))
    neighbor_indices = jnp.arange(300).reshape(10, 30)
    mask = jnp.ones((10,))

    dummy_ar_mask = jnp.zeros((10, 10))
    dummy_one_hot_sequence = jnp.zeros((10, 21))
    dummy_prng_key = jax.random.PRNGKey(0)
    dummy_temperature = jnp.array(1.0)
    dummy_bias = jnp.zeros((10, 21))

    seq, logits = model._call_unconditional(
        edge_features,
        neighbor_indices,
        mask,
        dummy_ar_mask,
        dummy_one_hot_sequence,
        dummy_prng_key,
        dummy_temperature,
        dummy_bias,
    )

    chex.assert_shape(seq, (10, 21))
    chex.assert_shape(logits, (10, 21))
    chex.assert_type(seq, float)
    chex.assert_type(logits, float)

def test_call_conditional(model):
    """Test the _call_conditional method."""
    edge_features = jnp.ones((10, 30, 128))
    neighbor_indices = jnp.arange(300).reshape(10, 30)
    mask = jnp.ones((10,))
    ar_mask = jnp.ones((10, 10))
    one_hot_sequence = jax.nn.one_hot(jnp.arange(10), 21)

    dummy_prng_key = jax.random.PRNGKey(0)
    dummy_temperature = jnp.array(1.0)
    dummy_bias = jnp.zeros((10, 21))

    out_seq, logits = model._call_conditional(
        edge_features,
        neighbor_indices,
        mask,
        ar_mask,
        one_hot_sequence,
        dummy_prng_key,
        dummy_temperature,
        dummy_bias,
    )

    chex.assert_trees_all_equal(out_seq, one_hot_sequence)
    chex.assert_shape(logits, (10, 21))
    chex.assert_type(out_seq, float)
    chex.assert_type(logits, float)

def test_call_autoregressive(model):
    """Test the _call_autoregressive method."""
    edge_features = jnp.ones((10, 30, 128))
    neighbor_indices = jnp.arange(300).reshape(10, 30)
    mask = jnp.ones((10,))
    ar_mask = jnp.ones((10, 10))
    prng_key = jax.random.PRNGKey(0)
    temperature = jnp.array(1.0)

    dummy_one_hot_sequence = jnp.zeros((10, 21))
    dummy_bias = jnp.zeros((10, 21))

    seq, logits = model._call_autoregressive(
        edge_features,
        neighbor_indices,
        mask,
        ar_mask,
        dummy_one_hot_sequence,
        prng_key,
        temperature,
        dummy_bias,
    )

    chex.assert_shape(seq, (10, 21))
    chex.assert_shape(logits, (10, 21))
    chex.assert_type(seq, float)
    chex.assert_type(logits, float)

@pytest.mark.parametrize(
    "decoding_approach", ["unconditional", "conditional", "autoregressive"]
)
def test_call(model, input_data, decoding_approach):
    """Test the __call__ method."""
    seq, logits = model(**input_data, decoding_approach=decoding_approach)
    chex.assert_shape(seq, (10, 21))
    chex.assert_shape(logits, (10, 21))
    chex.assert_type(seq, float)
    chex.assert_type(logits, float)

def test_call_no_key(model, input_data):
    """Test the __call__ method without a prng_key."""
    input_data.pop("prng_key")
    seq, logits = model(**input_data, decoding_approach="unconditional")
    chex.assert_shape(seq, (10, 21))
    chex.assert_shape(logits, (10, 21))
    chex.assert_type(seq, float)
    chex.assert_type(logits, float)
