"""Tests for equivalence between functional and Equinox implementations.

These tests ensure that the Equinox modules produce identical outputs to their
functional counterparts, validating the migration path.

tests.test_eqx_equivalence
"""

import equinox
import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn import conversion, eqx
from prxteinmpnn.functional import dense_layer, layer_normalization


class TestLayerNormEquivalence:
  """Test LayerNorm equivalence between functional and Equinox."""

  def test_layernorm_output_equivalence(self) -> None:
    """LayerNorm should produce identical outputs for functional vs Equinox."""
    # Setup
    batch_size = 4
    seq_len = 10
    features = 128

    # Create input
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch_size, seq_len, features))

    # Create scale and offset parameters
    scale = jnp.ones((features,))
    offset = jnp.zeros((features,))

    # Functional version (takes parameter dict)
    layer_params = {"scale": scale, "offset": offset}
    functional_out = layer_normalization(x, layer_params)

    # Equinox version
    eqx_layernorm = conversion.create_layernorm(scale, offset)
    eqx_out = jax.vmap(jax.vmap(eqx_layernorm))(x)

    # Assert equivalence
    assert jnp.allclose(functional_out, eqx_out, atol=1e-6)

  def test_layernorm_with_random_params(self) -> None:
    """LayerNorm should match with non-trivial scale/offset parameters."""
    # Setup
    features = 64
    key = jax.random.PRNGKey(42)
    key_x, key_scale, key_offset = jax.random.split(key, 3)

    # Create random input and parameters
    x = jax.random.normal(key_x, (2, 5, features))
    scale = jax.random.normal(key_scale, (features,))
    offset = jax.random.normal(key_offset, (features,))

    # Functional version (takes parameter dict)
    layer_params = {"scale": scale, "offset": offset}
    functional_out = layer_normalization(x, layer_params)

    # Equinox version
    eqx_layernorm = conversion.create_layernorm(scale, offset)
    eqx_out = jax.vmap(jax.vmap(eqx_layernorm))(x)

    # Assert equivalence
    assert jnp.allclose(functional_out, eqx_out, atol=1e-5)

  def test_layernorm_shape_preservation(self) -> None:
    """LayerNorm should preserve input shape."""
    features = 32
    scale = jnp.ones((features,))
    offset = jnp.zeros((features,))

    eqx_layernorm = conversion.create_layernorm(scale, offset)

    # Test various input shapes
    shapes = [(features,), (10, features), (5, 8, features)]
    for shape in shapes:
      key = jax.random.PRNGKey(0)
      x = jax.random.normal(key, shape)

      if len(shape) == 1:
        out = eqx_layernorm(x)
      elif len(shape) == 2:
        out = jax.vmap(eqx_layernorm)(x)
      else:
        out = jax.vmap(jax.vmap(eqx_layernorm))(x)

      assert out.shape == shape


class TestDenseLayerEquivalence:
  """Test DenseLayer equivalence between functional and Equinox."""

  def test_denselayer_output_equivalence(self) -> None:
    """DenseLayer should produce identical outputs for functional vs Equinox."""
    # Setup dimensions
    batch_size = 8
    in_features = 128
    hidden_features = 512
    out_features = 128

    # Create input
    key = jax.random.PRNGKey(1)
    key_x, key_w_in, key_b_in, key_w_out, key_b_out = jax.random.split(key, 5)

    x = jax.random.normal(key_x, (batch_size, in_features))

    # Create weight parameters
    w_in = jax.random.normal(key_w_in, (in_features, hidden_features))
    b_in = jax.random.normal(key_b_in, (hidden_features,))
    w_out = jax.random.normal(key_w_out, (hidden_features, out_features))
    b_out = jax.random.normal(key_b_out, (out_features,))

    p_dict = {
      "dense_W_in": {"w": w_in, "b": b_in},
      "dense_W_out": {"w": w_out, "b": b_out},
    }

    # Functional version (note: takes params first, then features)
    functional_out = dense_layer(p_dict, x)

    # Equinox version
    eqx_dense = conversion.create_dense(p_dict, key=key)
    eqx_out = jax.vmap(eqx_dense)(x)

    # Assert equivalence
    assert jnp.allclose(functional_out, eqx_out, atol=1e-5)

  def test_denselayer_single_input(self) -> None:
    """DenseLayer should work with single (non-batched) input."""
    # Setup
    in_features = 64
    hidden_features = 256
    out_features = 64

    key = jax.random.PRNGKey(2)
    key_x, key_w_in, key_b_in, key_w_out, key_b_out, key_dense = jax.random.split(key, 6)

    x = jax.random.normal(key_x, (in_features,))

    # Create parameters
    w_in = jax.random.normal(key_w_in, (in_features, hidden_features))
    b_in = jax.random.normal(key_b_in, (hidden_features,))
    w_out = jax.random.normal(key_w_out, (hidden_features, out_features))
    b_out = jax.random.normal(key_b_out, (out_features,))

    p_dict = {
      "dense_W_in": {"w": w_in, "b": b_in},
      "dense_W_out": {"w": w_out, "b": b_out},
    }

    # Functional version (note: takes params first, then features)
    functional_out = dense_layer(p_dict, x)

    # Equinox version
    eqx_dense = conversion.create_dense(p_dict, key=key_dense)
    eqx_out = eqx_dense(x)

    # Assert equivalence (slightly relaxed tolerance due to floating point accumulation)
    assert jnp.allclose(functional_out, eqx_out, atol=1e-4)

  def test_denselayer_jit_compatibility(self) -> None:
    """DenseLayer should be JIT-compatible."""
    # Setup
    in_features = 32
    hidden_features = 128
    out_features = 32

    key = jax.random.PRNGKey(3)
    key_x, key_params, key_dense = jax.random.split(key, 3)

    x = jax.random.normal(key_x, (16, in_features))

    # Create parameters
    key_w_in, key_b_in, key_w_out, key_b_out = jax.random.split(key_params, 4)
    w_in = jax.random.normal(key_w_in, (in_features, hidden_features))
    b_in = jax.random.normal(key_b_in, (hidden_features,))
    w_out = jax.random.normal(key_w_out, (hidden_features, out_features))
    b_out = jax.random.normal(key_b_out, (out_features,))

    p_dict = {
      "dense_W_in": {"w": w_in, "b": b_in},
      "dense_W_out": {"w": w_out, "b": b_out},
    }

    # Create Equinox module
    eqx_dense = conversion.create_dense(p_dict, key=key_dense)

    # JIT compile
    @jax.jit
    def apply_dense(x):
      return jax.vmap(eqx_dense)(x)

    # Should not raise
    out = apply_dense(x)
    assert out.shape == (16, out_features)


class TestLinearEquivalence:
  """Test Linear layer creation from weights."""

  def test_create_linear_basic(self) -> None:
    """create_linear should create a working Linear layer."""
    # Setup
    in_features = 64
    out_features = 128

    key = jax.random.PRNGKey(4)
    key_w, key_b, key_x = jax.random.split(key, 3)

    w = jax.random.normal(key_w, (in_features, out_features))
    b = jax.random.normal(key_b, (out_features,))
    x = jax.random.normal(key_x, (32, in_features))

    # Create Linear layer
    linear = conversion.create_linear(w, b)

    # Apply
    out = jax.vmap(linear)(x)

    # Check output shape
    assert out.shape == (32, out_features)

    # Manual computation for verification
    expected = x @ w + b
    assert jnp.allclose(out, expected, atol=1e-5)

  def test_create_linear_weight_transpose(self) -> None:
    """create_linear should handle weight transpose correctly."""
    # Equinox uses transposed weight convention: (out, in)
    in_features = 10
    out_features = 20

    key = jax.random.PRNGKey(5)
    key_w, key_b = jax.random.split(key)

    # Input weights in (in, out) format
    w = jax.random.normal(key_w, (in_features, out_features))
    b = jax.random.normal(key_b, (out_features,))

    linear = conversion.create_linear(w, b)

    # Verify internal weight shape is (out, in)
    assert linear.weight.shape == (out_features, in_features)

    # Verify computation is correct
    x = jnp.ones((in_features,))
    out = linear(x)
    expected = w.sum(axis=0) + b  # Sum over in_features + bias
    assert jnp.allclose(out, expected, atol=1e-5)


class TestEncoderLayerCreation:
  """Test EncoderLayer creation and structure."""

  def test_create_encoder_layer_structure(self) -> None:
    """create_encoder_layer should create a properly structured EncoderLayer."""
    # Setup dimensions
    node_features = 128
    edge_features = 384  # Typically node_features * 3 after concatenation
    hidden_features = 512

    # Create mock parameter dictionary
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 20)

    p_dict = {
      "W1": {
        "w": jax.random.normal(keys[0], (edge_features, hidden_features)),
        "b": jax.random.normal(keys[1], (hidden_features,)),
      },
      "W2": {
        "w": jax.random.normal(keys[2], (hidden_features, hidden_features)),
        "b": jax.random.normal(keys[3], (hidden_features,)),
      },
      "W3": {
        "w": jax.random.normal(keys[4], (hidden_features, node_features)),
        "b": jax.random.normal(keys[5], (node_features,)),
      },
      "norm1": {
        "scale": jnp.ones((node_features,)),
        "offset": jnp.zeros((node_features,)),
      },
      "dense_W_in": {
        "w": jax.random.normal(keys[6], (node_features, hidden_features)),
        "b": jax.random.normal(keys[7], (hidden_features,)),
      },
      "dense_W_out": {
        "w": jax.random.normal(keys[8], (hidden_features, node_features)),
        "b": jax.random.normal(keys[9], (node_features,)),
      },
      "norm2": {
        "scale": jnp.ones((node_features,)),
        "offset": jnp.zeros((node_features,)),
      },
      "W11": {
        "w": jax.random.normal(keys[10], (edge_features, hidden_features)),
        "b": jax.random.normal(keys[11], (hidden_features,)),
      },
      "W12": {
        "w": jax.random.normal(keys[12], (hidden_features, hidden_features)),
        "b": jax.random.normal(keys[13], (hidden_features,)),
      },
      "W13": {
        "w": jax.random.normal(keys[14], (hidden_features, node_features)),
        "b": jax.random.normal(keys[15], (node_features,)),
      },
      "norm3": {
        "scale": jnp.ones((node_features,)),
        "offset": jnp.zeros((node_features,)),
      },
    }

    # Create encoder layer
    encoder = conversion.create_encoder_layer(p_dict, key=keys[16])

    # Verify structure
    assert isinstance(encoder, eqx.EncoderLayer)
    assert isinstance(encoder.w1, equinox.nn.Linear)
    assert isinstance(encoder.w2, equinox.nn.Linear)
    assert isinstance(encoder.w3, equinox.nn.Linear)
    assert isinstance(encoder.norm1, eqx.LayerNorm)
    assert isinstance(encoder.dense, eqx.DenseLayer)
    assert isinstance(encoder.norm2, eqx.LayerNorm)
    assert isinstance(encoder.w11, equinox.nn.Linear)
    assert isinstance(encoder.w12, equinox.nn.Linear)
    assert isinstance(encoder.w13, equinox.nn.Linear)
    assert isinstance(encoder.norm3, eqx.LayerNorm)

  def test_encoder_layer_weight_dimensions(self) -> None:
    """EncoderLayer should have correct weight dimensions."""
    node_features = 64
    edge_features = 192
    hidden_features = 256

    key = jax.random.PRNGKey(1)
    keys = jax.random.split(key, 20)

    p_dict = {
      "W1": {
        "w": jax.random.normal(keys[0], (edge_features, hidden_features)),
        "b": jax.random.normal(keys[1], (hidden_features,)),
      },
      "W2": {
        "w": jax.random.normal(keys[2], (hidden_features, hidden_features)),
        "b": jax.random.normal(keys[3], (hidden_features,)),
      },
      "W3": {
        "w": jax.random.normal(keys[4], (hidden_features, node_features)),
        "b": jax.random.normal(keys[5], (node_features,)),
      },
      "norm1": {
        "scale": jnp.ones((node_features,)),
        "offset": jnp.zeros((node_features,)),
      },
      "dense_W_in": {
        "w": jax.random.normal(keys[6], (node_features, hidden_features)),
        "b": jax.random.normal(keys[7], (hidden_features,)),
      },
      "dense_W_out": {
        "w": jax.random.normal(keys[8], (hidden_features, node_features)),
        "b": jax.random.normal(keys[9], (node_features,)),
      },
      "norm2": {
        "scale": jnp.ones((node_features,)),
        "offset": jnp.zeros((node_features,)),
      },
      "W11": {
        "w": jax.random.normal(keys[10], (edge_features, hidden_features)),
        "b": jax.random.normal(keys[11], (hidden_features,)),
      },
      "W12": {
        "w": jax.random.normal(keys[12], (hidden_features, hidden_features)),
        "b": jax.random.normal(keys[13], (hidden_features,)),
      },
      "W13": {
        "w": jax.random.normal(keys[14], (hidden_features, node_features)),
        "b": jax.random.normal(keys[15], (node_features,)),
      },
      "norm3": {
        "scale": jnp.ones((node_features,)),
        "offset": jnp.zeros((node_features,)),
      },
    }

    encoder = conversion.create_encoder_layer(p_dict, key=keys[16])

    # Check weight shapes (Equinox uses transposed convention)
    assert encoder.w1.weight.shape == (hidden_features, edge_features)
    assert encoder.w2.weight.shape == (hidden_features, hidden_features)
    assert encoder.w3.weight.shape == (node_features, hidden_features)
    assert encoder.w11.weight.shape == (hidden_features, edge_features)
    assert encoder.w12.weight.shape == (hidden_features, hidden_features)
    assert encoder.w13.weight.shape == (node_features, hidden_features)


class TestDecoderLayerCreation:
  """Test DecoderLayer creation and structure."""

  def test_create_decoder_layer_structure(self) -> None:
    """create_decoder_layer should create a properly structured DecoderLayer."""
    # Setup dimensions
    sequence_features = 128
    edge_features = 384
    hidden_features = 512

    # Create mock parameter dictionary
    key = jax.random.PRNGKey(2)
    keys = jax.random.split(key, 15)

    p_dict = {
      "W1": {
        "w": jax.random.normal(keys[0], (edge_features, hidden_features)),
        "b": jax.random.normal(keys[1], (hidden_features,)),
      },
      "W2": {
        "w": jax.random.normal(keys[2], (hidden_features, hidden_features)),
        "b": jax.random.normal(keys[3], (hidden_features,)),
      },
      "W3": {
        "w": jax.random.normal(keys[4], (hidden_features, sequence_features)),
        "b": jax.random.normal(keys[5], (sequence_features,)),
      },
      "norm1": {
        "scale": jnp.ones((sequence_features,)),
        "offset": jnp.zeros((sequence_features,)),
      },
      "dense_W_in": {
        "w": jax.random.normal(keys[6], (sequence_features, hidden_features)),
        "b": jax.random.normal(keys[7], (hidden_features,)),
      },
      "dense_W_out": {
        "w": jax.random.normal(keys[8], (hidden_features, sequence_features)),
        "b": jax.random.normal(keys[9], (sequence_features,)),
      },
      "norm2": {
        "scale": jnp.ones((sequence_features,)),
        "offset": jnp.zeros((sequence_features,)),
      },
    }

    # Create decoder layer
    decoder = conversion.create_decoder_layer(p_dict, key=keys[10])

    # Verify structure
    assert isinstance(decoder, eqx.DecoderLayer)
    assert isinstance(decoder.w1, equinox.nn.Linear)
    assert isinstance(decoder.w2, equinox.nn.Linear)
    assert isinstance(decoder.w3, equinox.nn.Linear)
    assert isinstance(decoder.norm1, eqx.LayerNorm)
    assert isinstance(decoder.dense, eqx.DenseLayer)
    assert isinstance(decoder.norm2, eqx.LayerNorm)

  def test_decoder_layer_weight_dimensions(self) -> None:
    """DecoderLayer should have correct weight dimensions."""
    sequence_features = 64
    edge_features = 192
    hidden_features = 256

    key = jax.random.PRNGKey(3)
    keys = jax.random.split(key, 15)

    p_dict = {
      "W1": {
        "w": jax.random.normal(keys[0], (edge_features, hidden_features)),
        "b": jax.random.normal(keys[1], (hidden_features,)),
      },
      "W2": {
        "w": jax.random.normal(keys[2], (hidden_features, hidden_features)),
        "b": jax.random.normal(keys[3], (hidden_features,)),
      },
      "W3": {
        "w": jax.random.normal(keys[4], (hidden_features, sequence_features)),
        "b": jax.random.normal(keys[5], (sequence_features,)),
      },
      "norm1": {
        "scale": jnp.ones((sequence_features,)),
        "offset": jnp.zeros((sequence_features,)),
      },
      "dense_W_in": {
        "w": jax.random.normal(keys[6], (sequence_features, hidden_features)),
        "b": jax.random.normal(keys[7], (hidden_features,)),
      },
      "dense_W_out": {
        "w": jax.random.normal(keys[8], (hidden_features, sequence_features)),
        "b": jax.random.normal(keys[9], (sequence_features,)),
      },
      "norm2": {
        "scale": jnp.ones((sequence_features,)),
        "offset": jnp.zeros((sequence_features,)),
      },
    }

    decoder = conversion.create_decoder_layer(p_dict, key=keys[10])

    # Check weight shapes (Equinox uses transposed convention)
    assert decoder.w1.weight.shape == (hidden_features, edge_features)
    assert decoder.w2.weight.shape == (hidden_features, hidden_features)
    assert decoder.w3.weight.shape == (sequence_features, hidden_features)
