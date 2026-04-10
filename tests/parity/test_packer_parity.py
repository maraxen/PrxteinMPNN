"""Numerical parity checks for the side-chain packer against LigandMPNN reference."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prxteinmpnn.model.packer import Packer as JAXPacker
from scripts.convert_weights import convert_packer_model
from tests.parity.reference_utils import import_reference_module, require_heavy_parity_prereqs

torch = pytest.importorskip("torch")


def _build_synthetic_features(
  *,
  seq_len: int,
  num_context_atoms: int,
) -> tuple[dict[str, jnp.ndarray], dict[str, torch.Tensor]]:
  """Create deterministic synthetic features for JAX and PyTorch packers."""
  rng = np.random.default_rng(42)
  s_np = rng.integers(0, 21, (1, seq_len), dtype=np.int64)
  x_np = rng.standard_normal((1, seq_len, 14, 3), dtype=np.float32)
  y_np = rng.standard_normal((1, seq_len, num_context_atoms, 3), dtype=np.float32)
  y_m_np = np.ones((1, seq_len, num_context_atoms), dtype=np.float32)
  y_t_np = rng.integers(0, 119, (1, seq_len, num_context_atoms), dtype=np.int64)
  mask_np = np.ones((1, seq_len), dtype=np.float32)
  r_idx_np = np.tile(np.arange(seq_len), (1, 1)).astype(np.int64)
  chain_labels_np = np.zeros((1, seq_len), dtype=np.int64)
  x_m_np = np.ones((1, seq_len, 14), dtype=np.float32)

  feature_dict_jax = {
    "S": jnp.array(s_np[0]),
    "X": jnp.array(x_np[0]),
    "Y": jnp.array(y_np[0]),
    "Y_m": jnp.array(y_m_np[0]),
    "Y_t": jnp.array(y_t_np[0]),
    "mask": jnp.array(mask_np[0]),
    "R_idx": jnp.array(r_idx_np[0]),
    "chain_labels": jnp.array(chain_labels_np[0]),
    "X_m": jnp.array(x_m_np[0]),
  }
  feature_dict_pt = {
    "S": torch.from_numpy(s_np),
    "X": torch.from_numpy(x_np),
    "Y": torch.from_numpy(y_np),
    "Y_m": torch.from_numpy(y_m_np),
    "Y_t": torch.from_numpy(y_t_np),
    "mask": torch.from_numpy(mask_np),
    "R_idx": torch.from_numpy(r_idx_np),
    "chain_labels": torch.from_numpy(chain_labels_np),
    "X_m": torch.from_numpy(x_m_np),
  }
  return feature_dict_jax, feature_dict_pt


def _assert_numeric_parity(
  name: str,
  observed: np.ndarray,
  expected: np.ndarray,
  *,
  atol: float,
  rtol: float,
  corr_min: float = 0.999,
) -> None:
  """Assert numerical parity with tolerance and correlation checks."""
  assert observed.shape == expected.shape
  assert np.isfinite(observed).all()
  assert np.isfinite(expected).all()
  np.testing.assert_allclose(observed, expected, atol=atol, rtol=rtol)

  observed_std = float(np.std(observed))
  expected_std = float(np.std(expected))
  if observed_std == 0.0 or expected_std == 0.0:
    correlation = 1.0 if np.allclose(observed, expected, atol=atol, rtol=rtol) else 0.0
  else:
    correlation = float(np.corrcoef(observed.ravel(), expected.ravel())[0, 1])
  assert correlation >= corr_min, f"{name} correlation too low: {correlation:.6f} < {corr_min}"


@pytest.mark.parity_heavy
def test_packer_forward_numeric_parity() -> None:
  """Validate numerical parity for packer torsion outputs on shared deterministic inputs."""
  reference_root, _ = require_heavy_parity_prereqs(
    python_modules=["Bio"],
    reference_rel_paths=["model_params/ligandmpnn_sc_v_32_002_16.pt"],
  )
  sc_utils = import_reference_module("sc_utils")

  hidden_dim = 128
  num_layers = 3
  num_mix = 3
  seq_len = 20
  num_context_atoms = 16

  pt_packer = sc_utils.Packer(
    edge_features=128,
    node_features=128,
    num_positional_embeddings=16,
    num_chain_embeddings=16,
    num_rbf=16,
    top_k=30,
    augment_eps=0.0,
    atom37_order=False,
    device="cpu",
    atom_context_num=num_context_atoms,
    lower_bound=0.0,
    upper_bound=20.0,
    hidden_dim=hidden_dim,
    num_encoder_layers=num_layers,
    num_decoder_layers=num_layers,
    dropout=0.0,
    num_mix=num_mix,
  )
  checkpoint = torch.load(
    reference_root / "model_params/ligandmpnn_sc_v_32_002_16.pt",
    map_location="cpu",
  )
  pt_packer.load_state_dict(checkpoint["model_state_dict"])
  pt_packer.eval()

  jax_packer = JAXPacker(
    edge_features=128,
    node_features=128,
    num_positional_embeddings=16,
    num_rbf=16,
    top_k=30,
    atom37_order=False,
    atom_context_num=num_context_atoms,
    hidden_dim=hidden_dim,
    num_encoder_layers=num_layers,
    num_decoder_layers=num_layers,
    dropout=0.0,
    num_mix=num_mix,
    key=jax.random.PRNGKey(0),
  )
  pt_state_dict = {name: tensor.detach().cpu().numpy() for name, tensor in pt_packer.state_dict().items()}
  jax_packer = convert_packer_model(pt_state_dict, jax_packer)

  feature_dict_jax, feature_dict_pt = _build_synthetic_features(
    seq_len=seq_len,
    num_context_atoms=num_context_atoms,
  )

  mean_jax, conc_jax, mix_jax = jax_packer(feature_dict_jax)
  with torch.no_grad():
    h_v_pt, h_e_pt, e_idx_pt = pt_packer.encode(feature_dict_pt)
    feature_dict_pt.update({"h_V": h_v_pt, "h_E": h_e_pt, "E_idx": e_idx_pt})
    mean_pt, conc_pt, mix_pt = pt_packer.decode(feature_dict_pt)

  _assert_numeric_parity(
    "mean",
    np.asarray(mean_jax),
    mean_pt.detach().cpu().numpy()[0],
    atol=1e-4,
    rtol=1e-4,
  )
  _assert_numeric_parity(
    "concentration",
    np.asarray(conc_jax),
    conc_pt.detach().cpu().numpy()[0],
    atol=1e-3,
    rtol=1e-4,
  )
  _assert_numeric_parity(
    "mix_logits",
    np.asarray(mix_jax),
    mix_pt.detach().cpu().numpy()[0],
    atol=1e-4,
    rtol=1e-4,
  )
