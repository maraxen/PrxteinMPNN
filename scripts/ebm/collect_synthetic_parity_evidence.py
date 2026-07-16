"""Collect real original(PyTorch)-vs-JAX parity evidence for the ProteinEBM port (E3.5 gate).

This is a report-prep companion to ``scripts/ebm/checkpoint_parity_check.py`` (E3.5): instead of a
single fixed-size allclose check, it sweeps a grid of synthetic structure sizes/seeds through the
*same* real reference PyTorch model and real ported JAX model (identical checkpoint, identical
inputs), and persists every pointwise energy/score value plus a per-residue cosine-similarity
metric on the conservative-score and auxiliary-score 3-vectors -- the raw evidence a scatter plot
(original vs. JAX) or a cosine-similarity swarm plot needs. It does **not** attempt to reproduce the
paper's own Rosetta-decoy or ProteinGym-stability benchmarks (see ``decoy_ranking.py``/
``ddg_stability.py`` module docstrings for why those require multi-GB downloads out of scope here)
-- this only re-exercises the port's own internal E3.5 gate at more sample points.

Reuses ``scripts/ebm/checkpoint_parity_check.py``'s private helpers (loaded via
``importlib`` -- ``scripts/`` is not a package, and this is the same reference/JAX-construction
logic, not a fork of it) and ``aminx.parity.evidence``'s metric primitives/record schemas (the
same ``EvidenceMetricRecord``/``EvidencePointRecord``/``safe_cosine_similarity`` machinery the
existing LigandMPNN parity harness already uses), so this evidence is directly comparable in shape
to the rest of aminx's parity reporting rather than inventing a parallel schema.

Requires (dev-only, matching checkpoint_parity_check.py): ``torch``, the ProteinEBM reference repo,
and the real checkpoint file -- see that script's module docstring for sourcing.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from aminx.parity.evidence import (
  EvidenceMetricRecord,
  EvidencePointRecord,
  mean_abs_error,
  root_mean_square_error,
  safe_cosine_similarity,
  safe_pearson,
  write_metric_records_csv,
  write_metric_records_json,
  write_point_records_csv,
)

if TYPE_CHECKING:
  import types

log = logging.getLogger("collect_synthetic_parity_evidence")

_HERE = Path(__file__).parent
PATH_ID = "ebm_e3_5_checkpoint_parity"
TIER = "synthetic_fixed_input"


def _load_checkpoint_parity_module() -> "types.ModuleType":
  spec = importlib.util.spec_from_file_location(
    "checkpoint_parity_check", _HERE / "checkpoint_parity_check.py",
  )
  if spec is None or spec.loader is None:
    msg = "Could not load checkpoint_parity_check.py"
    raise ImportError(msg)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--reference-repo", type=Path, required=True)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 32, 64, 128])
  parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
  parser.add_argument("--diffusion-time", type=float, default=0.05, help="ProteinEBM-x MVP target t.")
  return parser.parse_args()


def main() -> int:  # noqa: PLR0914
  logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
  args = _parse_args()

  if not args.checkpoint.exists():
    log.error("Checkpoint not found: %s", args.checkpoint)
    return 1
  if not args.reference_repo.exists():
    log.error("Reference repo not found: %s", args.reference_repo)
    return 1

  cpc = _load_checkpoint_parity_module()

  import torch  # noqa: PLC0415

  log.info("Loading checkpoint: %s", args.checkpoint)
  ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

  log.info("Building + strict-loading reference PyTorch model...")
  ref_model = cpc._build_reference_model(args.reference_repo, ckpt)  # noqa: SLF001

  log.info("Building + porting JAX ProteinEBMModel...")
  ported_model, report = cpc._build_ported_jax_model(ckpt, seed=0)  # noqa: SLF001
  log.info(
    "Loaded %d checkpoint keys, skipped %d (itemized reasons in checkpoint_parity_check.py's own run).",
    len(report.loaded_keys),
    len(report.skipped_keys),
  )

  checkpoint_id = args.checkpoint.name
  point_records: list[EvidencePointRecord] = []
  metric_records: list[EvidenceMetricRecord] = []

  for n_residues in args.sizes:
    for seed in args.seeds:
      case_id = f"n{n_residues}_seed{seed}"
      t0 = time.time()
      fixed_input = cpc._build_fixed_input(n_residues, seed)  # noqa: SLF001
      ref_energy, ref_per_res, ref_score, ref_aux = cpc._reference_energy_and_score(  # noqa: SLF001
        ref_model, fixed_input, args.diffusion_time,
      )
      jax_energy, jax_per_res, jax_score, jax_aux = cpc._jax_energy_and_score(  # noqa: SLF001
        ported_model, fixed_input, args.diffusion_time,
      )
      elapsed = time.time() - t0
      mask = np.asarray(fixed_input["residue_mask"])
      log.info(
        "[%s] total_energy ref=%.6f jax=%.6f  (%.2fs)",
        case_id, float(ref_energy), float(jax_energy), elapsed,
      )

      valid = np.flatnonzero(mask)

      for i in valid:
        point_records.append(
          EvidencePointRecord(
            path_id=PATH_ID, tier=TIER, case_id=case_id, case_kind="per_residue_energy",
            backbone_id="synthetic", seed=seed, sequence_length=n_residues,
            reference_value=float(ref_per_res[i]), observed_value=float(jax_per_res[i]),
            point_kind="energy",
          ),
        )
        for comp in range(3):
          point_records.append(
            EvidencePointRecord(
              path_id=PATH_ID, tier=TIER, case_id=case_id,
              case_kind=f"conservative_score_component_{comp}", backbone_id="synthetic",
              seed=seed, sequence_length=n_residues,
              reference_value=float(ref_score[i, comp]), observed_value=float(jax_score[i, comp]),
              point_kind="score",
            ),
          )

        cos_score = safe_cosine_similarity(ref_score[i], jax_score[i])
        cos_aux = safe_cosine_similarity(ref_aux[i], jax_aux[i])
        metric_records.append(
          EvidenceMetricRecord(
            path_id=PATH_ID, tier=TIER, case_id=f"{case_id}_res{i}", case_kind="conservative_score",
            backbone_id="synthetic", seed=seed, sequence_length=n_residues,
            checkpoint_id=checkpoint_id, metric_name="cosine_similarity", metric_value=cos_score,
            metric_group="score_direction",
          ),
        )
        metric_records.append(
          EvidenceMetricRecord(
            path_id=PATH_ID, tier=TIER, case_id=f"{case_id}_res{i}", case_kind="aux_score",
            backbone_id="synthetic", seed=seed, sequence_length=n_residues,
            checkpoint_id=checkpoint_id, metric_name="cosine_similarity", metric_value=cos_aux,
            metric_group="score_direction",
          ),
        )

      ref_per_res_m = np.asarray(ref_per_res)[valid]
      jax_per_res_m = np.asarray(jax_per_res)[valid]
      ref_score_m = np.asarray(ref_score)[valid]
      jax_score_m = np.asarray(jax_score)[valid]
      for metric_name, fn in (
        ("mean_abs_error", mean_abs_error),
        ("root_mean_square_error", root_mean_square_error),
        ("pearson_r", safe_pearson),
      ):
        metric_records.append(
          EvidenceMetricRecord(
            path_id=PATH_ID, tier=TIER, case_id=case_id, case_kind="per_residue_energy",
            backbone_id="synthetic", seed=seed, sequence_length=n_residues,
            checkpoint_id=checkpoint_id, metric_name=metric_name,
            metric_value=fn(ref_per_res_m, jax_per_res_m), metric_group="energy",
          ),
        )
        metric_records.append(
          EvidenceMetricRecord(
            path_id=PATH_ID, tier=TIER, case_id=case_id, case_kind="conservative_score",
            backbone_id="synthetic", seed=seed, sequence_length=n_residues,
            checkpoint_id=checkpoint_id, metric_name=metric_name,
            metric_value=fn(ref_score_m.ravel(), jax_score_m.ravel()), metric_group="score",
          ),
        )

  args.out_dir.mkdir(parents=True, exist_ok=True)
  write_point_records_csv(point_records, args.out_dir / "synthetic_parity_points.csv")
  write_metric_records_json(metric_records, args.out_dir / "synthetic_parity_metrics.json")
  write_metric_records_csv(metric_records, args.out_dir / "synthetic_parity_metrics.csv")
  log.info(
    "Wrote %d point records + %d metric records to %s",
    len(point_records), len(metric_records), args.out_dir,
  )

  cos_scores = [
    m.metric_value for m in metric_records
    if m.metric_name == "cosine_similarity" and m.case_kind == "conservative_score"
  ]
  energy_mae_by_case = [
    m.metric_value for m in metric_records
    if m.metric_name == "mean_abs_error" and m.case_kind == "per_residue_energy"
  ]
  log.info(
    "=== conservative_score cosine similarity across %d (residue, size, seed) points: "
    "min=%.6f mean=%.6f max=%.6f ===",
    len(cos_scores), min(cos_scores), float(np.mean(cos_scores)), max(cos_scores),
  )

  summary = {
    "n_cases": len(args.sizes) * len(args.seeds),
    "n_point_records": len(point_records),
    "n_metric_records": len(metric_records),
    "cosine_similarity_min": float(min(cos_scores)),
    "cosine_similarity_mean": float(np.mean(cos_scores)),
    "mean_abs_error_energy_max": float(max(energy_mae_by_case)),
  }
  (args.out_dir / "synthetic_parity_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
  )
  log.info("Summary: %s", summary)

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
