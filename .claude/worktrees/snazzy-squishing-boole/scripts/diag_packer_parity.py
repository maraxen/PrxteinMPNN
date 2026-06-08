"""Diagnose JAX vs PyTorch side-chain packer ``mean`` / ``concentration`` / ``mix_logits``.

Run from ``prxteinmpnn`` root::

  export REFERENCE_PATH=/path/to/LigandMPNN
  PYTHONPATH=scripts:src:tests uv run python scripts/diag_packer_parity.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _repo_root() -> Path:
  return Path(__file__).resolve().parents[1]


def _stats(name: str, pt: object, jax_arr: object) -> None:
  import numpy as np

  from prxteinmpnn.parity.evidence import safe_pearson

  p = np.asarray(pt)
  j = np.asarray(jax_arr)
  if p.shape != j.shape:
    print(f"  {name}: shape pt{p.shape} jax{j.shape}")
    return
  d = np.abs(p - j)
  print(
    f"  {name}: max_abs={float(d.max()):.6g}  pearson={safe_pearson(p, j):.6f}",
  )


def main() -> None:
  repo = _repo_root()
  sys.path.insert(0, str(repo / "src"))
  sys.path.insert(0, str(repo))

  import jax
  import jax.numpy as jnp
  import jaxlib
  import torch

  from prxteinmpnn.model.packer import Packer as JAXPacker
  from scripts.convert_weights import convert_packer_model
  from tests.parity.reference_utils import import_reference_module, require_heavy_parity_prereqs
  from tests.parity.test_packer_parity import _build_synthetic_features, _forward_jax_packer_for_parity

  print("=== diag: packer forward ===")
  print(f"  jax {jax.__version__}  jaxlib {jaxlib.__version__}  default_backend={jax.default_backend()}")
  print(f"  jax devices: {jax.devices()}")
  print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
  print(f"  JAX_DEFAULT_MATMUL_PRECISION={os.environ.get('JAX_DEFAULT_MATMUL_PRECISION', '(unset)')}")

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

  mean_jax, conc_jax, mix_jax = _forward_jax_packer_for_parity(jax_packer, feature_dict_jax)
  with torch.no_grad():
    h_v_pt, h_e_pt, e_idx_pt = pt_packer.encode(feature_dict_pt)
    feature_dict_pt.update({"h_V": h_v_pt, "h_E": h_e_pt, "E_idx": e_idx_pt})
    mean_pt, conc_pt, mix_pt = pt_packer.decode(feature_dict_pt)

  print("  PyTorch packer device: cpu (reference)")
  print("  JAX forward uses ``_forward_jax_packer_for_parity`` (matmul highest + CPU if available)")
  _stats("mean", mean_pt.detach().cpu().numpy()[0], mean_jax)
  _stats("concentration", conc_pt.detach().cpu().numpy()[0], conc_jax)
  _stats("mix_logits", mix_pt.detach().cpu().numpy()[0], mix_jax)


if __name__ == "__main__":
  main()
