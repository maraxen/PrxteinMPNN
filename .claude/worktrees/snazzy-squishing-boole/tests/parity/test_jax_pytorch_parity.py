"""Low-level JAX/PyTorch parity checks for shared layers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prxteinmpnn.model.decoder import DecoderLayer as JAXDecoderLayer
from prxteinmpnn.model.encoder import EncoderLayer as JAXEncoderLayer
from tests.parity.reference_utils import prepend_reference_to_syspath

torch = pytest.importorskip("torch")


def test_weight_sharing_linear_layer() -> None:
  """Validate linear layer weight transfer parity."""
  in_size = 128
  out_size = 64
  torch_linear = torch.nn.Linear(in_size, out_size)
  pt_weight = torch_linear.weight.detach().numpy()
  pt_bias = torch_linear.bias.detach().numpy()

  import equinox as eqx

  jax_linear = eqx.nn.Linear(in_size, out_size, key=jax.random.PRNGKey(0))
  jax_linear = eqx.tree_at(lambda layer: layer.weight, jax_linear, jnp.array(pt_weight))
  jax_linear = eqx.tree_at(lambda layer: layer.bias, jax_linear, jnp.array(pt_bias))

  np.random.seed(42)
  x_np = np.random.randn(10, in_size).astype(np.float32)
  y_torch = torch_linear(torch.tensor(x_np)).detach().numpy()
  y_jax = np.asarray(jax.vmap(jax_linear)(jnp.array(x_np)))

  max_diff = float(np.max(np.abs(y_torch - y_jax)))
  assert max_diff < 1e-5


def test_encoder_layer_shape_parity() -> None:
  """Ensure encoder layers produce matching output shapes."""
  prepend_reference_to_syspath()
  import model_utils

  seq_len = 10
  node_features = 128
  edge_features = 128
  k_neighbors = 8

  jax_layer = JAXEncoderLayer(
    node_features=node_features,
    edge_features=edge_features,
    hidden_features=node_features,
    dropout_rate=0.0,
    key=jax.random.PRNGKey(42),
  )
  torch_layer = model_utils.EncLayer(num_hidden=node_features, num_in=node_features * 2, dropout=0.0)
  torch_layer.eval()

  np.random.seed(42)
  h_v = np.random.randn(seq_len, node_features).astype(np.float32)
  h_e = np.random.randn(seq_len, k_neighbors, edge_features).astype(np.float32)
  e_idx = np.random.randint(0, seq_len, (seq_len, k_neighbors)).astype(np.int64)
  mask = np.ones((seq_len,), dtype=np.float32)

  with torch.no_grad():
    h_v_out_torch, h_e_out_torch = torch_layer(
      torch.tensor(h_v).unsqueeze(0),
      torch.tensor(h_e).unsqueeze(0),
      torch.tensor(e_idx).unsqueeze(0).long(),
      mask_V=torch.tensor(mask).unsqueeze(0),
      mask_attend=None,
    )

  h_v_out_jax, h_e_out_jax = jax_layer(
    jnp.array(h_v),
    jnp.array(h_e),
    jnp.array(e_idx, dtype=jnp.int32),
    mask=jnp.array(mask),
    mask_attend=None,
    key=None,
  )

  assert h_v_out_torch.shape[1:] == h_v_out_jax.shape
  assert h_e_out_torch.shape[1:] == h_e_out_jax.shape
  assert np.isfinite(np.asarray(h_v_out_jax)).all()
  assert np.isfinite(np.asarray(h_e_out_jax)).all()


def test_decoder_layer_shape_parity() -> None:
  """Ensure decoder layers produce matching output shapes."""
  prepend_reference_to_syspath()
  import model_utils

  seq_len = 10
  node_features = 128
  k_neighbors = 8
  edge_context_features = node_features * 3

  jax_layer = JAXDecoderLayer(
    node_features=node_features,
    edge_context_features=edge_context_features,
    _hidden_features=node_features,
    dropout_rate=0.0,
    key=jax.random.PRNGKey(42),
  )
  torch_layer = model_utils.DecLayer(
    num_hidden=node_features,
    num_in=edge_context_features,
    dropout=0.0,
  )
  torch_layer.eval()

  np.random.seed(42)
  h_v = np.random.randn(seq_len, node_features).astype(np.float32)
  h_e = np.random.randn(seq_len, k_neighbors, edge_context_features).astype(np.float32)
  mask = np.ones((seq_len,), dtype=np.float32)

  with torch.no_grad():
    h_v_out_torch = torch_layer(
      torch.tensor(h_v).unsqueeze(0),
      torch.tensor(h_e).unsqueeze(0),
      mask_V=torch.tensor(mask).unsqueeze(0),
      mask_attend=None,
    )

  h_v_out_jax = jax_layer(
    jnp.array(h_v),
    jnp.array(h_e),
    mask=jnp.array(mask),
    attention_mask=None,
    key=None,
  )

  assert h_v_out_torch.shape[1:] == h_v_out_jax.shape
  assert np.isfinite(np.asarray(h_v_out_jax)).all()
