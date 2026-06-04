"""Diagnose JAX vs PyTorch ligand ``features()`` outputs (``y_nodes``, ``y_edges``, ``y_m``).

Run from ``aminx`` root::

  export REFERENCE_PATH=/path/to/LigandMPNN
  PYTHONPATH=scripts:src:tests uv run python scripts/diag_ligand_feature_parity.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
  return Path(__file__).resolve().parents[1]


def _stats(name: str, pt: object, jax_arr: object) -> None:
  from aminx.parity.evidence import safe_pearson

  p = np.asarray(pt)
  j = np.asarray(jax_arr)
  if p.shape != j.shape:
    print(f"  {name}: shape pt{p.shape} jax{j.shape}")
    return
  d = np.abs(p - j)
  print(
    f"  {name}: max_abs={float(d.max()):.6g}  mean_abs={float(d.mean()):.6g}  "
    f"pearson={safe_pearson(p, j):.6f}",
  )


def main() -> None:
  repo = _repo_root()
  sys.path.insert(0, str(repo / "src"))
  sys.path.insert(0, str(repo))

  import jax
  import jax.numpy as jnp
  import jaxlib
  import torch

  jax.config.update("jax_default_matmul_precision", "highest")

  from tests.model.test_ligandmpnn_equivalence import (
    build_ligand_batch,
    load_ligand_parity_bundle,
    _to_torch_feature_dict,
  )

  print("=== diag: ligand feature extraction ===")
  print(f"  jax {jax.__version__}  jaxlib {jaxlib.__version__}")
  print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
  print(f"  JAX_DEFAULT_MATMUL_PRECISION={os.environ.get('JAX_DEFAULT_MATMUL_PRECISION', '(unset)')}")

  bundle = load_ligand_parity_bundle()
  pt_model, jax_model = bundle.pt_model, bundle.jax_from_pt_convert
  batch = build_ligand_batch()
  feature_dict = _to_torch_feature_dict(batch, torch)

  with torch.no_grad():
    v_pt, e_pt, e_idx_pt, y_nodes_pt, y_edges_pt, y_m_pt = pt_model.features(feature_dict)

  v_jax, e_jax, e_idx_jax, y_nodes_jax, y_edges_jax, y_m_jax = jax_model.features(
    jax.random.PRNGKey(17),
    jnp.array(batch.x[0]),
    jnp.array(batch.mask[0]),
    jnp.array(batch.residue_index[0]),
    jnp.array(batch.chain_index[0]),
    jnp.array(batch.y[0]),
    jnp.array(batch.y_t[0]),
    jnp.array(batch.y_m[0]),
  )

  print("  ligand_l_chunk (JAX):", int(jax_model.features.ligand_l_chunk))
  print("  comparisons (pt vs jax, batch slice 0):")
  _stats("V (node embed pre-encode)", v_pt.numpy()[0], v_jax)
  _stats("E (edge pre-encode)", e_pt.numpy()[0], e_jax)
  _stats("y_nodes", y_nodes_pt.numpy()[0], y_nodes_jax)
  _stats("y_edges", y_edges_pt.numpy()[0], y_edges_jax)
  _stats("y_m", y_m_pt.numpy()[0], y_m_jax)
  print("  e_idx match:", bool(np.array_equal(e_idx_pt.numpy()[0], np.asarray(e_idx_jax))))


if __name__ == "__main__":
  main()
