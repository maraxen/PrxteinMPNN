"""Diagnose JAX vs PyTorch protein feature extraction parity (steps 1–2).

1) Prints stack versions (JAX / PyTorch / CUDA / matmul precision env).
2) Runs the same batch as ``test_protein_feature_extraction_parity`` and compares
   each JAX edge stage to reference ``features()`` raw edges and ``W_e`` output.

Run from ``prxteinmpnn`` repo root (or tev_design with prxteinmpnn on PYTHONPATH)::

  export REFERENCE_PATH=/path/to/LigandMPNN
  PYTHONPATH=scripts:src:tests uv run python scripts/diag_protein_feature_parity.py

Requires torch, reference checkout, and converted ``proteinmpnn_v_48_020.eqx.zst``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np


def _repo_root() -> Path:
  return Path(__file__).resolve().parents[1]


def _pearson(lhs: np.ndarray, rhs: np.ndarray) -> float:
  from prxteinmpnn.parity.evidence import safe_pearson

  return safe_pearson(lhs, rhs)


def _max_abs(lhs: np.ndarray, rhs: np.ndarray) -> float:
  if lhs.shape != rhs.shape:
    return float("nan")
  return float(np.max(np.abs(lhs - rhs)))


def _compare_stage(label: str, pt: np.ndarray, jax_arr: Any) -> None:
  j = np.asarray(jax_arr)
  if pt.shape != j.shape:
    print(f"  {label}: shape mismatch pt{pt.shape} jax{j.shape}")
    return
  print(
    f"  {label}: max_abs={_max_abs(pt, j):.6g}  pearson={_pearson(pt, j):.6f}",
  )


def main() -> None:
  repo = _repo_root()
  sys.path.insert(0, str(repo / "src"))
  sys.path.insert(0, str(repo))

  import jax
  import jax.numpy as jnp
  import jaxlib
  import torch

  from tests.parity.test_full_model_parity import (
    JaxHeavyWeightSource,
    _build_parity_batch,
    _build_torch_feature_dict,
    _jax_protein_for_source,
    _load_heavy_parity_models_impl,
  )

  print("=== 1) Stack versions ===")
  print(f"  JAX_PLATFORM={os.environ.get('JAX_PLATFORM', '(unset)')}")
  print(f"  JAX_DEFAULT_MATMUL_PRECISION={os.environ.get('JAX_DEFAULT_MATMUL_PRECISION', '(unset)')}")
  print(f"  jax {jax.__version__}  jaxlib {jaxlib.__version__}")
  print(f"  torch {torch.__version__}")
  print(f"  torch cuda available: {torch.cuda.is_available()}")
  if torch.cuda.is_available():
    print(f"  torch cuda: {torch.version.cuda}")
  print(f"  jax default backend: {jax.default_backend()}")
  print(f"  jax devices: {jax.devices()}")

  print("\n=== 2) Edge pipeline stage binary-search (vs PyTorch) ===")
  models = _load_heavy_parity_models_impl()
  batch = _build_parity_batch()
  feature_dict = _build_torch_feature_dict(models.torch, batch)

  with models.torch.no_grad():
    pt_edges, _pt_nei = models.pt_model.features(feature_dict)
    pt_proj = models.pt_model.W_e(pt_edges).numpy()[0]
  pt_edges_np = pt_edges.numpy()[0]

  feat_key = jax.random.PRNGKey(0)
  mask0 = jnp.array(batch.mask[0])
  ridx0 = jnp.array(batch.residue_index[0])
  cidx0 = jnp.array(batch.chain_index[0])
  noise0 = jnp.array(0.0, dtype=jnp.float32)

  for source in ("eqx", "pt_convert"):
    print(f"\n-- JAX weight source: {source} --")
    jax_model = _jax_protein_for_source(models, cast(JaxHeavyWeightSource, source))
    stages = jax_model.features.forward_edge_stages(
      feat_key,
      batch.x_jax_atom37,
      mask0,
      ridx0,
      cidx0,
      noise0,
    )
    jax_final = np.asarray(stages.final)
    print(
      f"  final vs pt_W_e(features): max_abs={_max_abs(pt_proj, jax_final):.6g}  pearson={_pearson(pt_proj, jax_final):.6f}",
    )
    _compare_stage("after_w_e vs pt_edges (raw)", pt_edges_np, stages.after_w_e)
    _compare_stage("after_norm vs pt_edges", pt_edges_np, stages.after_norm)
    _compare_stage("final vs pt_edges", pt_edges_np, stages.final)
    print(
      f"  encoded_positions shape {stages.encoded_positions.shape}  "
      f"rbf {stages.rbf.shape}  edges_concat {stages.edges_concat.shape}  pt_edges {pt_edges_np.shape}",
    )

  jax_eqx = _jax_protein_for_source(models, "eqx")
  jax_pt = _jax_protein_for_source(models, "pt_convert")
  s_eqx = jax_eqx.features.forward_edge_stages(
    feat_key, batch.x_jax_atom37, mask0, ridx0, cidx0, noise0
  )
  s_pt = jax_pt.features.forward_edge_stages(
    feat_key, batch.x_jax_atom37, mask0, ridx0, cidx0, noise0
  )
  print("\n-- eqx vs pt_convert (same PRNGKey, final edges) --")
  d = np.asarray(s_eqx.final) - np.asarray(s_pt.final)
  print(f"  max_abs(eqx - pt_convert)={float(np.max(np.abs(d))):.6g}")


if __name__ == "__main__":
  main()
