"""Weight conversion helpers for migrating from functional to Equinox models.

This module provides utilities to convert PyTree parameter dictionaries from
the functional API to Equinox module instances.

prxteinmpnn.conversion
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax

from prxteinmpnn.eqx import Decoder, DecoderLayer, DenseLayer, Encoder, EncoderLayer, LayerNorm

if TYPE_CHECKING:
  from jaxtyping import Array, PRNGKeyArray

  from prxteinmpnn.utils.types import ModelParameters


def create_linear(w: Array, b: Array) -> eqx.nn.Linear:
  """Create an Equinox Linear layer from weight and bias arrays.

  Args:
    w: Weight matrix of shape (in_features, out_features).
    b: Bias vector of shape (out_features,).

  Returns:
    An eqx.nn.Linear module with the given weights and biases.

  Example:
    >>> import jax.numpy as jnp
    >>> w = jnp.ones((128, 256))
    >>> b = jnp.zeros((256,))
    >>> linear = create_linear(w, b)
    >>> x = jnp.ones((10, 128))
    >>> y = linear(x)
    >>> y.shape
    (10, 256)

  """
  in_features, out_features = w.shape
  # Create a dummy Linear layer to get the structure
  dummy_key = jax.random.PRNGKey(0)
  linear = eqx.nn.Linear(in_features, out_features, key=dummy_key)

  # Replace weights and bias
  linear = eqx.tree_at(lambda m: m.weight, linear, w.T)  # Transpose for Equinox convention
  return eqx.tree_at(lambda m: m.bias, linear, b)


def create_layernorm(
  scale: Array,
  offset: Array,
) -> LayerNorm:
  """Create an Equinox LayerNorm from scale and offset arrays.

  Args:
    scale: Scale parameter (gamma) of shape matching the normalized dimensions.
    offset: Offset parameter (beta) of shape matching the normalized dimensions.

  Returns:
    An eqx.nn.LayerNorm module with the given scale and offset.

  Example:
    >>> import jax.numpy as jnp
    >>> scale = jnp.ones((128,))
    >>> offset = jnp.zeros((128,))
    >>> layer_norm = create_layernorm(scale, offset)
    >>> x = jax.random.normal(jax.random.PRNGKey(0), (10, 128))
    >>> y = layer_norm(x)

  """
  # Create a LayerNorm with the right shape (inferred from scale/offset)
  shape = scale.shape
  layer_norm = eqx.nn.LayerNorm(shape)

  # Replace scale and offset with pretrained weights
  layer_norm = eqx.tree_at(lambda m: m.weight, layer_norm, scale)
  return eqx.tree_at(lambda m: m.bias, layer_norm, offset)


def create_dense(p_dict: ModelParameters, *, key: PRNGKeyArray) -> DenseLayer:
  """Create an Equinox DenseLayer from parameter dictionary.

  Args:
    p_dict: Parameter dictionary containing dense layer weights.
    key: PRNG key for initialization (used for structure, weights replaced).

  Returns:
    A DenseLayer module with weights from the parameter dictionary.

  Example:
    >>> import jax
    >>> import jax.numpy as jnp
    >>> params = {
    ...   "dense_W_in": {"w": jnp.ones((128, 512)), "b": jnp.zeros((512,))},
    ...   "dense_W_out": {"w": jnp.ones((512, 128)), "b": jnp.zeros((128,))},
    ... }
    >>> key = jax.random.PRNGKey(0)
    >>> dense = create_dense(params, "test", key=key)

  """
  w_in = p_dict["dense_W_in"]["w"]
  b_in = p_dict["dense_W_in"]["b"]
  w_out = p_dict["dense_W_out"]["w"]
  b_out = p_dict["dense_W_out"]["b"]

  in_features = w_in.shape[0]
  hidden_features = w_in.shape[1]
  out_features = w_out.shape[1]

  # Create a DenseLayer with the right dimensions
  dense = DenseLayer(in_features, hidden_features, out_features, key=key)

  # Replace the weights
  linear_in = create_linear(w_in, b_in)
  linear_out = create_linear(w_out, b_out)

  dense = eqx.tree_at(lambda m: m.linear_in, dense, linear_in)
  return eqx.tree_at(lambda m: m.linear_out, dense, linear_out)


def create_encoder_layer(p_dict: ModelParameters, *, key: PRNGKeyArray) -> EncoderLayer:
  """Create an Equinox EncoderLayer from parameter dictionary.

  Args:
    p_dict: Parameter dictionary containing encoder layer weights.
      Expected keys: W1, W2, W3, norm1, dense_W_in, dense_W_out, norm2, W11, W12, W13, norm3.
    key: PRNG key for initialization (used for structure, weights replaced).

  Returns:
    An EncoderLayer module with weights from the parameter dictionary.

  Example:
    >>> import jax
    >>> import jax.numpy as jnp
    >>> # Assume p_dict contains all necessary weights
    >>> key = jax.random.PRNGKey(0)
    >>> encoder = create_encoder_layer(p_dict, key=key)

  """
  # Extract weight dimensions from the parameter dictionary
  node_features = p_dict["W3"]["w"].shape[1]  # Output dimension of W3
  edge_features = p_dict["W1"]["w"].shape[0]  # Input dimension of W1
  hidden_features = p_dict["W1"]["w"].shape[1]  # Hidden dimension

  # Create encoder layer with correct dimensions
  encoder = EncoderLayer(node_features, edge_features, hidden_features, key=key)

  # Replace edge message computation weights (W1, W2, W3)
  encoder = eqx.tree_at(
    lambda m: m.w1,
    encoder,
    create_linear(p_dict["W1"]["w"], p_dict["W1"]["b"]),
  )
  encoder = eqx.tree_at(
    lambda m: m.w2,
    encoder,
    create_linear(p_dict["W2"]["w"], p_dict["W2"]["b"]),
  )
  encoder = eqx.tree_at(
    lambda m: m.w3,
    encoder,
    create_linear(p_dict["W3"]["w"], p_dict["W3"]["b"]),
  )

  # Replace normalization layers
  encoder = eqx.tree_at(
    lambda m: m.norm1,
    encoder,
    create_layernorm(p_dict["norm1"]["scale"], p_dict["norm1"]["offset"]),
  )
  encoder = eqx.tree_at(
    lambda m: m.norm2,
    encoder,
    create_layernorm(p_dict["norm2"]["scale"], p_dict["norm2"]["offset"]),
  )
  encoder = eqx.tree_at(
    lambda m: m.norm3,
    encoder,
    create_layernorm(p_dict["norm3"]["scale"], p_dict["norm3"]["offset"]),
  )

  # Replace dense layer
  dense_params = {
    "dense_W_in": p_dict["dense_W_in"],
    "dense_W_out": p_dict["dense_W_out"],
  }
  encoder = eqx.tree_at(lambda m: m.dense, encoder, create_dense(dense_params, key=key))

  # Replace edge update weights (W11, W12, W13)
  encoder = eqx.tree_at(
    lambda m: m.w11,
    encoder,
    create_linear(p_dict["W11"]["w"], p_dict["W11"]["b"]),
  )
  encoder = eqx.tree_at(
    lambda m: m.w12,
    encoder,
    create_linear(p_dict["W12"]["w"], p_dict["W12"]["b"]),
  )
  return eqx.tree_at(
    lambda m: m.w13,
    encoder,
    create_linear(p_dict["W13"]["w"], p_dict["W13"]["b"]),
  )


def create_decoder_layer(p_dict: ModelParameters, *, key: PRNGKeyArray) -> DecoderLayer:
  """Create an Equinox DecoderLayer from parameter dictionary.

  Args:
    p_dict: Parameter dictionary containing decoder layer weights.
      Expected keys: W1, W2, W3, norm1, dense_W_in, dense_W_out, norm2.
    key: PRNG key for initialization (used for structure, weights replaced).

  Returns:
    A DecoderLayer module with weights from the parameter dictionary.

  Example:
    >>> import jax
    >>> import jax.numpy as jnp
    >>> # Assume p_dict contains all necessary weights
    >>> key = jax.random.PRNGKey(0)
    >>> decoder = create_decoder_layer(p_dict, key=key)

  """
  # Extract weight dimensions from the parameter dictionary
  sequence_features = p_dict["W3"]["w"].shape[1]  # Output dimension of W3
  edge_features = p_dict["W1"]["w"].shape[0]  # Input dimension of W1
  hidden_features = p_dict["W1"]["w"].shape[1]  # Hidden dimension

  # Create decoder layer with correct dimensions
  decoder = DecoderLayer(sequence_features, edge_features, hidden_features, key=key)

  # Replace edge message computation weights (W1, W2, W3)
  decoder = eqx.tree_at(
    lambda m: m.w1,
    decoder,
    create_linear(p_dict["W1"]["w"], p_dict["W1"]["b"]),
  )
  decoder = eqx.tree_at(
    lambda m: m.w2,
    decoder,
    create_linear(p_dict["W2"]["w"], p_dict["W2"]["b"]),
  )
  decoder = eqx.tree_at(
    lambda m: m.w3,
    decoder,
    create_linear(p_dict["W3"]["w"], p_dict["W3"]["b"]),
  )

  # Replace normalization layers
  decoder = eqx.tree_at(
    lambda m: m.norm1,
    decoder,
    create_layernorm(p_dict["norm1"]["scale"], p_dict["norm1"]["offset"]),
  )
  decoder = eqx.tree_at(
    lambda m: m.norm2,
    decoder,
    create_layernorm(p_dict["norm2"]["scale"], p_dict["norm2"]["offset"]),
  )

  # Replace dense layer
  dense_params = {
    "dense_W_in": p_dict["dense_W_in"],
    "dense_W_out": p_dict["dense_W_out"],
  }
  return eqx.tree_at(lambda m: m.dense, decoder, create_dense(dense_params, key=key))


def create_encoder_stack(
  model_parameters: ModelParameters,
  num_layers: int = 3,
  *,
  key: PRNGKeyArray,
) -> tuple[EncoderLayer, ...]:
  """Create a stack of EncoderLayers from model parameters.

  Args:
    model_parameters: Full model parameter dictionary.
    num_layers: Number of encoder layers to create.
    key: PRNG key for initialization.

  Returns:
    A list of EncoderLayer modules, one for each layer.

  Example:
    >>> import jax
    >>> from prxteinmpnn.functional import get_functional_model
    >>> model_params = get_functional_model()
    >>> key = jax.random.PRNGKey(0)
    >>> encoders = create_encoder_stack(model_params, num_layers=3, key=key)
    >>> len(encoders)
    3

  """
  # Import here to avoid circular dependency
  from prxteinmpnn.functional import encoder_parameter_pytree  # noqa: PLC0415

  # Get stacked parameters
  all_encoder_params = encoder_parameter_pytree(model_parameters, num_layers)

  # Create encoder layers
  encoder_layers = []
  keys = jax.random.split(key, num_layers)

  for i in range(num_layers):
    # Extract parameters for this layer
    layer_params = jax.tree_util.tree_map(lambda x, idx=i: x[idx], all_encoder_params)
    encoder = create_encoder_layer(layer_params, key=keys[i])
    encoder_layers.append(encoder)

  return tuple(encoder_layers)


def create_decoder_stack(
  model_parameters: ModelParameters,
  num_layers: int = 3,
  *,
  key: PRNGKeyArray,
) -> tuple[DecoderLayer, ...]:
  """Create a stack of DecoderLayers from model parameters.

  Args:
    model_parameters: Full model parameter dictionary.
    num_layers: Number of decoder layers to create.
    key: PRNG key for initialization.

  Returns:
    A tuple of DecoderLayer modules, one for each layer.

  Example:
    >>> import jax
    >>> from prxteinmpnn.functional import get_functional_model
    >>> model_params = get_functional_model()
    >>> key = jax.random.PRNGKey(0)
    >>> decoders = create_decoder_stack(model_params, num_layers=3, key=key)
    >>> len(decoders)
    3

  """
  # Import here to avoid circular dependency
  from prxteinmpnn.functional import decoder_parameter_pytree  # noqa: PLC0415

  # Get stacked parameters
  all_decoder_params = decoder_parameter_pytree(model_parameters, num_layers)

  # Create decoder layers
  decoder_layers = []
  keys = jax.random.split(key, num_layers)

  for i in range(num_layers):
    # Extract parameters for this layer
    layer_params = jax.tree_util.tree_map(lambda x, idx=i: x[idx], all_decoder_params)
    decoder = create_decoder_layer(layer_params, key=keys[i])
    decoder_layers.append(decoder)

  return tuple(decoder_layers)


def create_encoder(
  model_parameters: ModelParameters,
  num_layers: int = 3,
  scale: float = 30.0,
  *,
  key: PRNGKeyArray,
) -> Encoder:
  """Create a full Encoder module from model parameters.

  Args:
    model_parameters: Full model parameter dictionary.
    num_layers: Number of encoder layers.
    scale: Scaling factor for message aggregation.
    key: PRNG key for initialization.

  Returns:
    An Encoder module with all layers populated from the model parameters.

  Example:
    >>> import jax
    >>> from prxteinmpnn.functional import get_functional_model
    >>> model_params = get_functional_model()
    >>> key = jax.random.PRNGKey(0)
    >>> encoder = create_encoder(model_params, num_layers=3, key=key)

  """
  # Import here to avoid circular dependency
  from prxteinmpnn.functional import encoder_parameter_pytree  # noqa: PLC0415

  # Get stacked parameters to infer dimensions
  all_encoder_params = encoder_parameter_pytree(model_parameters, num_layers)

  # Infer dimensions from first layer
  first_layer = jax.tree_util.tree_map(lambda x: x[0], all_encoder_params)
  node_features = first_layer["W3"]["w"].shape[1]
  edge_features = first_layer["W1"]["w"].shape[0]
  hidden_features = first_layer["W1"]["w"].shape[1]

  # Create encoder with correct dimensions
  encoder = Encoder(
    node_features,
    edge_features,
    hidden_features,
    num_layers=num_layers,
    scale=scale,
    key=key,
  )

  # Replace each layer with populated version
  encoder_layers = create_encoder_stack(model_parameters, num_layers, key=key)
  return eqx.tree_at(lambda e: e.layers, encoder, encoder_layers)


def create_decoder(
  model_parameters: ModelParameters,
  num_layers: int = 3,
  scale: float = 30.0,
  *,
  key: PRNGKeyArray,
) -> Decoder:
  """Create a full Decoder module from model parameters.

  Args:
    model_parameters: Full model parameter dictionary.
    num_layers: Number of decoder layers.
    scale: Scaling factor for message aggregation.
    key: PRNG key for initialization.

  Returns:
    A Decoder module with all layers populated from the model parameters.

  Example:
    >>> import jax
    >>> from prxteinmpnn.functional import get_functional_model
    >>> model_params = get_functional_model()
    >>> key = jax.random.PRNGKey(0)
    >>> decoder = create_decoder(model_params, num_layers=3, key=key)

  """
  # Import here to avoid circular dependency
  from prxteinmpnn.functional import decoder_parameter_pytree  # noqa: PLC0415

  # Get stacked parameters to infer dimensions
  all_decoder_params = decoder_parameter_pytree(model_parameters, num_layers)

  # Infer dimensions from first layer
  first_layer = jax.tree_util.tree_map(lambda x: x[0], all_decoder_params)
  sequence_features = first_layer["W3"]["w"].shape[1]
  edge_features = first_layer["W1"]["w"].shape[0]
  hidden_features = first_layer["W1"]["w"].shape[1]

  # Create decoder with correct dimensions
  decoder = Decoder(
    sequence_features,
    edge_features,
    hidden_features,
    num_layers=num_layers,
    scale=scale,
    key=key,
  )

  # Replace each layer with populated version
  decoder_layers = create_decoder_stack(model_parameters, num_layers, key=key)
  return eqx.tree_at(lambda d: d.layers, decoder, decoder_layers)


def create_prxteinmpnn(
  model_parameters: ModelParameters,
  num_encoder_layers: int = 3,
  num_decoder_layers: int = 3,
  scale: float = 30.0,
  *,
  key: PRNGKeyArray,
):
  """Create a full PrxteinMPNN model from model parameters.

  Args:
    model_parameters: Full model parameter dictionary from pkl file.
    num_encoder_layers: Number of encoder layers.
    num_decoder_layers: Number of decoder layers.
    scale: Scaling factor for message aggregation.
    key: PRNG key for initialization.

  Returns:
    A PrxteinMPNN module with all components populated.

  Example:
    >>> import jax
    >>> from prxteinmpnn.functional import get_functional_model
    >>> model_params = get_functional_model()
    >>> key = jax.random.PRNGKey(0)
    >>> model = create_prxteinmpnn(
    ...     model_params, num_encoder_layers=3, num_decoder_layers=3, key=key
    ... )

  """
  # Import here to avoid circular dependency
  from prxteinmpnn.eqx import PrxteinMPNN  # noqa: PLC0415

  # Split key for encoder and decoder
  key_encoder, key_decoder = jax.random.split(key)

  # Create encoder and decoder
  encoder = create_encoder(
    model_parameters,
    num_layers=num_encoder_layers,
    scale=scale,
    key=key_encoder,
  )
  decoder = create_decoder(
    model_parameters,
    num_layers=num_decoder_layers,
    scale=scale,
    key=key_decoder,
  )

  # Extract output projection parameters
  w_out = model_parameters["protein_mpnn/~/W_out"]["w"]
  b_out = model_parameters["protein_mpnn/~/W_out"]["b"]

  # Extract edge embedding parameters
  w_e = model_parameters["protein_mpnn/~/protein_features/~/edge_embedding"]["w"]

  # Extract positional encoding parameters
  pos_key = "protein_mpnn/~/protein_features/~/positional_encodings/~/embedding_linear"
  w_pos = model_parameters[pos_key]["w"]
  b_pos = model_parameters[pos_key]["b"]

  # Create output projection layer
  w_out_linear = create_linear(w_out, b_out)

  return PrxteinMPNN(
    encoder=encoder,
    decoder=decoder,
    w_out=w_out_linear,
    b_out=b_out,
    w_e=w_e,
    w_pos=w_pos,
    b_pos=b_pos,
  )
