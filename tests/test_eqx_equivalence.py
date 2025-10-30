"""Tests for equivalence between functional and Equinox implementations.

These tests ensure that the Equinox modules produce identical outputs to their
functional counterparts, validating the migration path.

tests.test_eqx_equivalence
"""

import equinox
import jax
import jax.numpy as jnp

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


class TestEncoderDecoderStacks:
  """Test creation of encoder and decoder stacks from model parameters."""

  def test_create_encoder_stack(self) -> None:
    """create_encoder_stack should create a list of EncoderLayers."""
    # Load actual model parameters
    from prxteinmpnn.functional import get_functional_model

    model_params = get_functional_model()
    key = jax.random.PRNGKey(0)

    # Create encoder stack
    num_layers = 3
    encoders = conversion.create_encoder_stack(model_params, num_layers, key=key)

    # Verify structure
    assert isinstance(encoders, tuple)
    assert len(encoders) == num_layers

    for encoder in encoders:
      assert isinstance(encoder, eqx.EncoderLayer)
      assert isinstance(encoder.w1, equinox.nn.Linear)
      assert isinstance(encoder.dense, eqx.DenseLayer)
      assert isinstance(encoder.norm1, eqx.LayerNorm)

  def test_create_decoder_stack(self) -> None:
    """create_decoder_stack should create a list of DecoderLayers."""
    # Load actual model parameters
    from prxteinmpnn.functional import get_functional_model

    model_params = get_functional_model()
    key = jax.random.PRNGKey(1)

    # Create decoder stack
    num_layers = 3
    decoders = conversion.create_decoder_stack(model_params, num_layers, key=key)

    # Verify structure
    assert isinstance(decoders, tuple)
    assert len(decoders) == num_layers

    for decoder in decoders:
      assert isinstance(decoder, eqx.DecoderLayer)
      assert isinstance(decoder.w1, equinox.nn.Linear)
      assert isinstance(decoder.dense, eqx.DenseLayer)
      assert isinstance(decoder.norm1, eqx.LayerNorm)

  def test_encoder_stack_different_num_layers(self) -> None:
    """Encoder stack should support different numbers of layers."""
    from prxteinmpnn.functional import get_functional_model

    model_params = get_functional_model()
    key = jax.random.PRNGKey(2)

    # Test with 1, 2, and 3 layers
    for num_layers in [1, 2, 3]:
      encoders = conversion.create_encoder_stack(model_params, num_layers, key=key)
      assert len(encoders) == num_layers

  def test_decoder_stack_different_num_layers(self) -> None:
    """Decoder stack should support different numbers of layers."""
    from prxteinmpnn.functional import get_functional_model

    model_params = get_functional_model()
    key = jax.random.PRNGKey(3)

    # Test with 1, 2, and 3 layers
    for num_layers in [1, 2, 3]:
      decoders = conversion.create_decoder_stack(model_params, num_layers, key=key)
      assert len(decoders) == num_layers


class TestFullEncoderDecoder:
  """Test creation of full Encoder and Decoder modules."""

  def test_create_encoder_module(self) -> None:
    """create_encoder should create a full Encoder module."""
    from prxteinmpnn.functional import get_functional_model

    model_params = get_functional_model()
    key = jax.random.PRNGKey(0)

    # Create encoder
    num_layers = 3
    encoder = conversion.create_encoder(model_params, num_layers, key=key)

    # Verify structure
    assert isinstance(encoder, eqx.Encoder)
    assert isinstance(encoder.layers, tuple)
    assert len(encoder.layers) == num_layers
    assert encoder.scale == 30.0
    assert encoder.node_feature_dim == 128  # Standard ProteinMPNN dimension

    # Verify each layer
    for layer in encoder.layers:
      assert isinstance(layer, eqx.EncoderLayer)

  def test_create_decoder_module(self) -> None:
    """create_decoder should create a full Decoder module."""
    from prxteinmpnn.functional import get_functional_model

    model_params = get_functional_model()
    key = jax.random.PRNGKey(1)

    # Create decoder
    num_layers = 3
    decoder = conversion.create_decoder(model_params, num_layers, key=key)

    # Verify structure
    assert isinstance(decoder, eqx.Decoder)
    assert isinstance(decoder.layers, tuple)
    assert len(decoder.layers) == num_layers
    assert decoder.scale == 30.0

    # Verify each layer
    for layer in decoder.layers:
      assert isinstance(layer, eqx.DecoderLayer)

  def test_encoder_with_custom_scale(self) -> None:
    """Encoder should support custom scale parameter."""
    from prxteinmpnn.functional import get_functional_model

    model_params = get_functional_model()
    key = jax.random.PRNGKey(2)

    # Create encoder with custom scale
    custom_scale = 15.0
    encoder = conversion.create_encoder(model_params, num_layers=2, scale=custom_scale, key=key)

    assert encoder.scale == custom_scale
    assert len(encoder.layers) == 2

  def test_decoder_with_custom_scale(self) -> None:
    """Decoder should support custom scale parameter."""
    from prxteinmpnn.functional import get_functional_model

    model_params = get_functional_model()
    key = jax.random.PRNGKey(3)

    # Create decoder with custom scale
    custom_scale = 20.0
    decoder = conversion.create_decoder(model_params, num_layers=2, scale=custom_scale, key=key)

    assert decoder.scale == custom_scale
    assert len(decoder.layers) == 2

  def test_encoder_decoder_are_pytrees(self) -> None:
    """Encoder and Decoder should be valid JAX PyTrees."""
    from prxteinmpnn.functional import get_functional_model

    model_params = get_functional_model()
    key = jax.random.PRNGKey(4)

    encoder = conversion.create_encoder(model_params, num_layers=2, key=key)
    decoder = conversion.create_decoder(model_params, num_layers=2, key=key)

    # Should be able to flatten/unflatten (PyTree requirement)
    encoder_flat, encoder_treedef = jax.tree_util.tree_flatten(encoder)
    encoder_reconstructed = jax.tree_util.tree_unflatten(encoder_treedef, encoder_flat)
    assert isinstance(encoder_reconstructed, eqx.Encoder)

    decoder_flat, decoder_treedef = jax.tree_util.tree_flatten(decoder)
    decoder_reconstructed = jax.tree_util.tree_unflatten(decoder_treedef, decoder_flat)
    assert isinstance(decoder_reconstructed, eqx.Decoder)


class TestEncoderDecoderForwardPass:
  """Test forward pass methods of Encoder and Decoder modules."""

  def test_encoder_forward_pass_shape(self) -> None:
    """Encoder forward pass should produce correct output shapes."""
    from prxteinmpnn.functional import get_functional_model

    model_params = get_functional_model()
    key = jax.random.PRNGKey(0)

    # Create encoder
    encoder = conversion.create_encoder(model_params, num_layers=3, key=key)

    # Create test inputs - raw edge features before encoder processing
    num_atoms = 50
    num_neighbors = 30
    edge_dim = 128  # Raw edge dimension (before concatenation)
    edge_features = jax.random.normal(key, (num_atoms, num_neighbors, edge_dim))
    neighbor_indices = jnp.arange(num_atoms)[:, None].repeat(num_neighbors, axis=1)
    mask = jnp.ones(num_atoms)

    # Run forward pass
    node_features, updated_edge_features = encoder(edge_features, neighbor_indices, mask)

    # Check shapes
    assert node_features.shape == (num_atoms, 128)  # node_feature_dim = 128
    assert updated_edge_features.shape == edge_features.shape

  def test_decoder_forward_pass_shape(self) -> None:
    """Decoder forward pass should produce correct output shapes."""
    from prxteinmpnn.functional import get_functional_model

    model_params = get_functional_model()
    key = jax.random.PRNGKey(1)

    # Create decoder
    decoder = conversion.create_decoder(model_params, num_layers=3, key=key)

    # Create test inputs
    # In decoder, edge_features come from encoder output or conditional setup
    # For unconditional decoder: concatenates [nodes, zeros, edges] = 128+128+128 = 384
    # But this gets concatenated again in decode_message with nodes: 128+384 = 512
    num_atoms = 50
    num_neighbors = 30
    sequence_dim = 128
    # Edge features dimension after encoder processing or from extract_features
    edge_dim = 128
    node_features = jax.random.normal(key, (num_atoms, sequence_dim))
    edge_features = jax.random.normal(key, (num_atoms, num_neighbors, edge_dim))
    mask = jnp.ones(num_atoms)

    # But decoder's __call__ needs the pre-concatenated edge features
    # that will become [nodes_expanded, zeros_expanded, edges] inside
    # Let me check what the actual input should be...
    # Actually, looking at functional decoder, it takes processed edge_features
    # So edge_dim should be 128 (output from encoder or raw from extract_features)

    # Run forward pass
    updated_node_features = decoder(node_features, edge_features, mask)

    # Check shape
    assert updated_node_features.shape == node_features.shape

  def test_encoder_forward_pass_with_mask(self) -> None:
    """Encoder should properly apply mask to outputs."""
    from prxteinmpnn.functional import get_functional_model

    model_params = get_functional_model()
    key = jax.random.PRNGKey(2)

    encoder = conversion.create_encoder(model_params, num_layers=2, key=key)

    # Create test inputs with partial mask
    num_atoms = 20
    num_neighbors = 15
    edge_features = jax.random.normal(key, (num_atoms, num_neighbors, 128))
    neighbor_indices = jnp.arange(num_atoms)[:, None].repeat(num_neighbors, axis=1)
    mask = jnp.concatenate([jnp.ones(10), jnp.zeros(10)])  # Half masked

    node_features, _ = encoder(edge_features, neighbor_indices, mask)

    # Check that masked positions are zero
    assert jnp.allclose(node_features[10:], 0.0)
    # Check that unmasked positions are non-zero
    assert jnp.any(node_features[:10] != 0.0)

  def test_decoder_forward_pass_with_mask(self) -> None:
    """Decoder should properly apply mask to outputs."""
    from prxteinmpnn.functional import get_functional_model

    model_params = get_functional_model()
    key = jax.random.PRNGKey(3)

    decoder = conversion.create_decoder(model_params, num_layers=2, key=key)

    # Create test inputs with partial mask
    num_atoms = 20
    num_neighbors = 15
    node_features = jax.random.normal(key, (num_atoms, 128))
    edge_features = jax.random.normal(key, (num_atoms, num_neighbors, 128))
    mask = jnp.concatenate([jnp.ones(10), jnp.zeros(10)])  # Half masked

    updated_node_features = decoder(node_features, edge_features, mask)

    # Check that masked positions are zero
    assert jnp.allclose(updated_node_features[10:], 0.0)
    # Check that unmasked positions are non-zero
    assert jnp.any(updated_node_features[:10] != 0.0)

  def test_encoder_jit_compatibility(self) -> None:
    """Encoder forward pass should be JIT-compilable."""
    from prxteinmpnn.functional import get_functional_model

    model_params = get_functional_model()
    key = jax.random.PRNGKey(4)

    encoder = conversion.create_encoder(model_params, num_layers=2, key=key)

    # JIT compile forward pass
    @jax.jit
    def jit_encoder(edge_features, neighbor_indices, mask):
      return encoder(edge_features, neighbor_indices, mask)

    # Create test inputs
    edge_features = jax.random.normal(key, (20, 15, 128))
    neighbor_indices = jnp.arange(20)[:, None].repeat(15, axis=1)
    mask = jnp.ones(20)

    # Should not raise
    node_features, updated_edges = jit_encoder(edge_features, neighbor_indices, mask)
    assert node_features.shape == (20, 128)

  def test_decoder_jit_compatibility(self) -> None:
    """Decoder forward pass should be JIT-compilable."""
    from prxteinmpnn.functional import get_functional_model

    model_params = get_functional_model()
    key = jax.random.PRNGKey(5)

    decoder = conversion.create_decoder(model_params, num_layers=2, key=key)

    # JIT compile forward pass
    @jax.jit
    def jit_decoder(node_features, edge_features, mask):
      return decoder(node_features, edge_features, mask)

    # Create test inputs
    node_features = jax.random.normal(key, (20, 128))
    edge_features = jax.random.normal(key, (20, 15, 128))
    mask = jnp.ones(20)

    # Should not raise
    updated_node_features = jit_decoder(node_features, edge_features, mask)
    assert updated_node_features.shape == (20, 128)


class TestPrxteinMPNNForwardPass:
  """Test complete PrxteinMPNN model forward pass."""

  def test_model_forward_pass_shape(self) -> None:
    """Full model forward pass should produce correct logit shapes."""
    from prxteinmpnn.functional import get_functional_model

    model_params = get_functional_model()
    key = jax.random.PRNGKey(42)

    # Create full model
    model = conversion.create_prxteinmpnn(
      model_params,
      num_encoder_layers=3,
      num_decoder_layers=3,
      key=key,
    )

    # Create test inputs
    num_atoms = 50
    num_neighbors = 30
    edge_dim = 128  # Raw edge dimension
    edge_features = jax.random.normal(key, (num_atoms, num_neighbors, edge_dim))
    neighbor_indices = jnp.arange(num_atoms)[:, None].repeat(num_neighbors, axis=1)
    mask = jnp.ones(num_atoms)

    # Run forward pass
    logits = model(edge_features, neighbor_indices, mask)

    # Check shape: should be (num_atoms, 21) for amino acid predictions
    assert logits.shape == (num_atoms, 21)

  def test_model_with_mask(self) -> None:
    """Model should correctly handle masking."""
    from prxteinmpnn.functional import get_functional_model

    model_params = get_functional_model()
    key = jax.random.PRNGKey(43)

    # Create model
    model = conversion.create_prxteinmpnn(
      model_params,
      num_encoder_layers=2,
      num_decoder_layers=2,
      key=key,
    )

    # Create test inputs with partial mask
    num_atoms = 20
    num_neighbors = 15
    edge_dim = 128
    edge_features = jax.random.normal(key, (num_atoms, num_neighbors, edge_dim))
    neighbor_indices = jnp.arange(num_atoms)[:, None].repeat(num_neighbors, axis=1)
    mask = jnp.concatenate([jnp.ones(10), jnp.zeros(10)])  # Half masked

    # Run forward pass
    logits = model(edge_features, neighbor_indices, mask)

    # Check shape
    assert logits.shape == (num_atoms, 21)

  def test_model_jit_compatibility(self) -> None:
    """Model should be JIT-compilable."""
    from prxteinmpnn.functional import get_functional_model

    model_params = get_functional_model()
    key = jax.random.PRNGKey(44)

    # Create model
    model = conversion.create_prxteinmpnn(
      model_params,
      num_encoder_layers=2,
      num_decoder_layers=2,
      key=key,
    )

    # JIT compile a function that calls the model
    @jax.jit
    def model_fn(edge_features, neighbor_indices, mask):
      return model(edge_features, neighbor_indices, mask)

    # Create test inputs
    edge_features = jax.random.normal(key, (20, 15, 128))
    neighbor_indices = jnp.arange(20)[:, None].repeat(15, axis=1)
    mask = jnp.ones(20)

    # Should not raise
    logits = model_fn(edge_features, neighbor_indices, mask)
    assert logits.shape == (20, 21)


class TestNumericalEquivalence:
  """Test numerical equivalence between Equinox and functional implementations.

  These tests ensure that the full forward pass produces identical results
  between the Equinox model and the original functional implementation.
  """

  def test_encoder_numerical_equivalence(self) -> None:
    """Encoder should produce numerically equivalent outputs to functional encoder."""
    from prxteinmpnn.functional import get_functional_model, make_encoder

    model_params = get_functional_model()
    key = jax.random.PRNGKey(123)

    # Create test inputs
    num_atoms = 20
    num_neighbors = 15
    edge_dim = 128
    edge_features = jax.random.normal(key, (num_atoms, num_neighbors, edge_dim))
    neighbor_indices = jnp.arange(num_atoms)[:, None].repeat(num_neighbors, axis=1)
    mask = jnp.ones(num_atoms)

    # Create Equinox encoder
    eqx_encoder = conversion.create_encoder(model_params, num_layers=3, key=key)

    # Run Equinox encoder
    eqx_nodes, eqx_edges = eqx_encoder(edge_features, neighbor_indices, mask)

    # Run functional encoder
    func_encoder = make_encoder(model_params, num_encoder_layers=3, scale=30.0)
    func_nodes, func_edges = func_encoder(edge_features, neighbor_indices, mask)

    # Compare outputs (should be very close, within floating point tolerance)
    # Use slightly looser tolerance for float32 operations
    assert jnp.allclose(eqx_nodes, func_nodes, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(eqx_edges, func_edges, rtol=1e-4, atol=1e-5)

  def test_full_model_numerical_equivalence(self) -> None:
    """Full model should produce identical logits to functional implementation."""
    from prxteinmpnn.functional import (
      final_projection,
      get_functional_model,
      make_decoder,
      make_encoder,
    )

    model_params = get_functional_model()
    key = jax.random.PRNGKey(789)

    # Create test inputs
    num_atoms = 25
    num_neighbors = 20
    edge_dim = 128
    edge_features = jax.random.normal(key, (num_atoms, num_neighbors, edge_dim))
    neighbor_indices = jnp.arange(num_atoms)[:, None].repeat(num_neighbors, axis=1)
    mask = jnp.ones(num_atoms)

    # ===== Equinox Model =====
    eqx_model = conversion.create_prxteinmpnn(
      model_params,
      num_encoder_layers=3,
      num_decoder_layers=3,
      key=key,
    )
    eqx_logits = eqx_model(edge_features, neighbor_indices, mask)

    # ===== Functional Model =====
    # Encoder
    func_encoder = make_encoder(model_params, num_encoder_layers=3, scale=30.0)
    func_nodes, func_edges = func_encoder(edge_features, neighbor_indices, mask)

    # Decoder
    func_decoder = make_decoder(
      model_params,
      attention_mask_type=None,
      num_decoder_layers=3,
      scale=30.0,
    )
    func_nodes = func_decoder(func_nodes, func_edges, mask)

    # Projection
    func_logits = final_projection(model_params, func_nodes)

    # Compare final logits (use tolerance appropriate for float32 and accumulated ops)
    assert jnp.allclose(eqx_logits, func_logits, rtol=1e-5, atol=1e-5)

  def test_model_save_load_equivalence(self) -> None:
    """Saved and loaded model should produce identical outputs."""
    import pathlib
    import tempfile

    from prxteinmpnn.functional import get_functional_model

    model_params = get_functional_model()
    key = jax.random.PRNGKey(999)

    # Create original model
    original_model = conversion.create_prxteinmpnn(
      model_params,
      num_encoder_layers=3,
      num_decoder_layers=3,
      key=key,
    )

    # Create test inputs
    num_atoms = 15
    num_neighbors = 10
    edge_features = jax.random.normal(key, (num_atoms, num_neighbors, 128))
    neighbor_indices = jnp.arange(num_atoms)[:, None].repeat(num_neighbors, axis=1)
    mask = jnp.ones(num_atoms)

    # Get original output
    original_logits = original_model(edge_features, neighbor_indices, mask)

    # Save and load model
    with tempfile.TemporaryDirectory() as tmpdir:
      model_path = pathlib.Path(tmpdir) / "test_model.eqx"
      equinox.tree_serialise_leaves(model_path, original_model)

      # Load model
      loaded_model = eqx.load_prxteinmpnn(str(model_path))

      # Get loaded output
      loaded_logits = loaded_model(edge_features, neighbor_indices, mask)

    # Compare outputs
    assert jnp.allclose(original_logits, loaded_logits, rtol=1e-7, atol=1e-8)

  def test_model_with_different_batch_sizes(self) -> None:
    """Model should handle different batch sizes correctly."""
    from prxteinmpnn.functional import get_functional_model

    model_params = get_functional_model()
    key = jax.random.PRNGKey(111)

    # Create model
    model = conversion.create_prxteinmpnn(
      model_params,
      num_encoder_layers=3,
      num_decoder_layers=3,
      key=key,
    )

    # Test with different sizes
    for num_atoms in [10, 20, 50]:
      for num_neighbors in [5, 15, 30]:
        edge_features = jax.random.normal(key, (num_atoms, num_neighbors, 128))
        neighbor_indices = jnp.arange(num_atoms)[:, None].repeat(num_neighbors, axis=1)
        mask = jnp.ones(num_atoms)

        logits = model(edge_features, neighbor_indices, mask)

        # Verify output shape
        assert logits.shape == (num_atoms, 21)

  def test_model_with_partial_masking(self) -> None:
    """Model should handle partial masking correctly."""
    from prxteinmpnn.functional import (
      final_projection,
      get_functional_model,
      make_decoder,
      make_encoder,
    )

    model_params = get_functional_model()
    key = jax.random.PRNGKey(222)

    # Create test inputs with partial mask
    num_atoms = 30
    num_neighbors = 20
    edge_features = jax.random.normal(key, (num_atoms, num_neighbors, 128))
    neighbor_indices = jnp.arange(num_atoms)[:, None].repeat(num_neighbors, axis=1)

    # Create partial mask (first half valid, second half masked)
    mask = jnp.concatenate([jnp.ones(15), jnp.zeros(15)])

    # Equinox model
    eqx_model = conversion.create_prxteinmpnn(
      model_params,
      num_encoder_layers=3,
      num_decoder_layers=3,
      key=key,
    )
    eqx_logits = eqx_model(edge_features, neighbor_indices, mask)

    # Functional model
    func_encoder = make_encoder(model_params, num_encoder_layers=3, scale=30.0)
    func_nodes, func_edges = func_encoder(edge_features, neighbor_indices, mask)

    func_decoder = make_decoder(
      model_params,
      attention_mask_type=None,
      num_decoder_layers=3,
      scale=30.0,
    )
    func_nodes = func_decoder(func_nodes, func_edges, mask)
    func_logits = final_projection(model_params, func_nodes)

    # Compare (use float32-appropriate tolerance)
    assert jnp.allclose(eqx_logits, func_logits, rtol=1e-5, atol=1e-5)

    # Verify masked positions are consistent between implementations
    assert jnp.allclose(eqx_logits[15:], func_logits[15:], rtol=1e-5, atol=1e-5)
