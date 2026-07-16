"""Consolidate ProteinEBM parity evidence (E3.5 synthetic sweep + E7 real LplA check) into one
compact JSON blob for the report Artifact's charts.

Reads the bathos-tracked outputs already written under ``outputs/ebm_benchmarks/`` (by
``collect_synthetic_parity_evidence.py`` and ``lpla_biasing_check.py``) plus the hardcoded
throughput-benchmark numbers already published in
``.praxia/docs/plans/260709_proteinebm-epic-backlog-dag.md`` (sections 7/9/11), and writes a single
``ebm_parity_report_data.json`` for the HTML report to embed inline -- no runtime data fetches
(the artifact must be self-contained).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--evidence-dir", type=Path, default=Path("outputs/ebm_benchmarks/synthetic_parity"))
  parser.add_argument("--lpla-json", type=Path, default=Path("outputs/ebm_benchmarks/lpla_biasing_real.json"))
  parser.add_argument("--benchmarks-dir", type=Path, default=Path("outputs/ebm_benchmarks"))
  parser.add_argument("--out", type=Path, default=Path("outputs/ebm_benchmarks/ebm_parity_report_data.json"))
  return parser.parse_args()


# Track A depth benchmarks (A1-A3b): wall-clock + batch-size sweep + 3-way PyTorch variant,
# heterogeneous-batch strategy comparison, and outer Langevin-annealing/E10 throughput. Each entry
# names the file(s) `run_cluster_benchmarks.sh` writes for that script; missing files resolve to
# `None` in the output rather than erroring, since not every depth script has completed a cluster
# run yet (see .praxia/docs/audits/260716_proteinebm-parity-report.md sec 7 for current status).
DEPTH_BENCHMARK_FILES = {
  "decoy": ["decoy_benchmark_full_L64-128.json", "decoy_benchmark_full_L256.json", "decoy_benchmark_full_L512.json"],
  "ddg": ["ddg_benchmark_full_L64-128.json", "ddg_benchmark_full_L256.json", "ddg_benchmark_full_L512.json"],
  "biasing": ["biasing_benchmark_full.json"],
  "langevin": ["langevin_benchmark_full_L64-128.json", "langevin_benchmark_full_L256.json", "langevin_benchmark_full_L512.json"],
  "heterogeneous": ["heterogeneous_batch_benchmark_full.json"],
  "annealing": ["langevin_annealing_benchmark_full.json"],
}


def _load_rows(path: Path) -> list[dict]:
  data = json.loads(path.read_text())
  return data if isinstance(data, list) else data.get("results", data.get("rows", []))


def _load_depth_benchmarks(benchmarks_dir: Path) -> dict[str, dict | None]:
  out: dict[str, dict | None] = {}
  for key, filenames in DEPTH_BENCHMARK_FILES.items():
    paths = [benchmarks_dir / name for name in filenames]
    present = [p for p in paths if p.exists()]
    if not present:
      out[key] = None
      continue
    rows: list[dict] = []
    for p in present:
      rows.extend(_load_rows(p))
    out[key] = {"rows": rows, "n_missing_files": len(paths) - len(present), "source_files": [p.name for p in present]}
  return out


# Throughput numbers already published (real GPU runs) -- see plan doc sections 7 (titanix), 9 +
# 11 (engaging/Blackwell, decoy+ddg re-run with the jaxlib 0.9.2 pin after the grad-path bug fix).
THROUGHPUT = {
  "decoy": {
    "device": "titanix (Turing)",
    "rows": [
      {"length": 64, "jax": 286.0, "pytorch": 25.2, "speedup": 11.4},
      {"length": 128, "jax": 173.9, "pytorch": 6.9, "speedup": 25.3},
      {"length": 256, "jax": 70.8, "pytorch": 1.9, "speedup": 37.0},
      {"length": 512, "jax": 22.0, "pytorch": 0.5, "speedup": 44.1},
    ],
  },
  "decoy_blackwell_pinned": {
    "device": "engaging/Blackwell (jaxlib 0.9.2 pin)",
    "rows": [
      {"length": 64, "jax": 1840.0, "pytorch": 40.3, "speedup": 45.7},
      {"length": 128, "jax": 1032.9, "pytorch": 11.1, "speedup": 92.9},
      {"length": 256, "jax": 321.0, "pytorch": 4.1, "speedup": 79.2},
      {"length": 512, "jax": 92.7, "pytorch": 1.1, "speedup": 87.6},
    ],
  },
  "ddg": {
    "device": "titanix (Turing)",
    "rows": [
      {"length": 64, "jax": 304.2, "pytorch": 19.0, "speedup": 16.0},
      {"length": 128, "jax": 170.5, "pytorch": 6.3, "speedup": 27.0},
      {"length": 256, "jax": 70.9, "pytorch": 1.9, "speedup": 38.1},
      {"length": 512, "jax": 21.9, "pytorch": 0.5, "speedup": 43.8},
    ],
  },
  "ddg_blackwell_pinned": {
    "device": "engaging/Blackwell (jaxlib 0.9.2 pin)",
    "rows": [
      {"length": 64, "jax": 1799.4, "pytorch": 44.5, "speedup": 40.5},
      {"length": 128, "jax": 1032.1, "pytorch": 19.3, "speedup": 53.6},
      {"length": 256, "jax": 321.0, "pytorch": 7.0, "speedup": 46.0},
      {"length": 512, "jax": 92.1, "pytorch": 1.7, "speedup": 54.8},
    ],
  },
  "biasing": {
    "device": "titanix (Turing)",
    "rows": [
      {"length": 64, "jax": 184.0, "pytorch": 15.4, "speedup": 11.9},
      {"length": 128, "jax": 128.9, "pytorch": 7.7, "speedup": 16.8},
      {"length": 256, "jax": 64.4, "pytorch": 1.8, "speedup": 36.8},
      {"length": 512, "jax": 21.3, "pytorch": 0.5, "speedup": 41.5},
    ],
  },
  "biasing_blackwell": {
    "device": "engaging/Blackwell",
    "rows": [
      {"length": 64, "jax": 1040.6, "pytorch": 25.5, "speedup": 40.8},
      {"length": 128, "jax": 721.1, "pytorch": 13.7, "speedup": 52.8},
      {"length": 256, "jax": 326.3, "pytorch": 5.5, "speedup": 59.6},
      {"length": 512, "jax": 84.9, "pytorch": 1.6, "speedup": 54.2},
    ],
  },
  "langevin_blackwell": {
    "device": "engaging/Blackwell",
    "rows": [
      {"length": 64, "jax": 2016.8, "pytorch": 46.3, "speedup": 43.5},
      {"length": 128, "jax": 992.7, "pytorch": 21.2, "speedup": 46.9},
      {"length": 256, "jax": 284.3, "pytorch": 6.4, "speedup": 44.4},
      {"length": 512, "jax": 79.5, "pytorch": 1.4, "speedup": 54.9},
    ],
  },
}

# jaxlib grad-path regression bisection (plan doc section 11).
JAXLIB_BISECTION = [
  {"version": "0.10.2 (pinned)", "result": "FAIL"},
  {"version": "0.9.2", "result": "PASS"},
  {"version": "0.9.0", "result": "PASS"},
  {"version": "0.8.3", "result": "PASS"},
  {"version": "0.8.0", "result": "PASS"},
]


def _load_points(evidence_dir: Path) -> list[dict]:
  path = evidence_dir / "synthetic_parity_points.csv"
  with path.open(newline="", encoding="utf-8") as fh:
    return list(csv.DictReader(fh))


def _load_metrics(evidence_dir: Path) -> list[dict]:
  path = evidence_dir / "synthetic_parity_metrics.json"
  return json.loads(path.read_text())


def main() -> int:
  args = _parse_args()

  points = _load_points(args.evidence_dir)
  metrics = _load_metrics(args.evidence_dir)
  summary = json.loads((args.evidence_dir / "synthetic_parity_summary.json").read_text())

  energy_scatter = [
    {
      "ref": float(row["reference_value"]),
      "jax": float(row["observed_value"]),
      "size": int(row["sequence_length"]),
    }
    for row in points
    if row["case_kind"] == "per_residue_energy"
  ]

  cosine_swarm = [
    {
      "one_minus_cos": max(1.0 - float(m["metric_value"]), 1e-17),
      "size": int(m["sequence_length"]),
      "case_kind": m["case_kind"],
    }
    for m in metrics
    if m["metric_name"] == "cosine_similarity"
  ]

  energy_mae_by_case = sorted(
    {
      (m["case_id"], m["sequence_length"]): m["metric_value"]
      for m in metrics
      if m["metric_name"] == "mean_abs_error" and m["case_kind"] == "per_residue_energy"
    }.items(),
  )

  lpla = None
  if args.lpla_json.exists():
    lpla_raw = json.loads(args.lpla_json.read_text())
    lpla = {
      "n_mutants": lpla_raw["n_mutants"],
      "spearman_r": lpla_raw["spearman_r"],
      "spearman_p": lpla_raw["spearman_p"],
      "n_canonical_residues": lpla_raw["n_canonical_residues"],
      "n_shared_residues": lpla_raw["n_shared_residues"],
      "points": [
        {"activity": m["promiscuous_activity"], "delta_e": m["delta_e"], "label": m["label"]}
        for m in lpla_raw["mutants"]
      ],
    }

  depth = _load_depth_benchmarks(args.benchmarks_dir)

  out = {
    "synthetic_e3_5": {
      "summary": summary,
      "energy_scatter": energy_scatter,
      "cosine_swarm": cosine_swarm,
      "energy_mae_by_case": [{"case": k[0], "size": k[1], "mae": v} for k, v in energy_mae_by_case],
    },
    "lpla_e7": lpla,
    "throughput": THROUGHPUT,
    "jaxlib_bisection": JAXLIB_BISECTION,
    "throughput_depth": depth,
  }

  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(out, indent=2))
  n_depth_ready = sum(1 for v in depth.values() if v is not None)
  print(
    f"Wrote {args.out} ({len(energy_scatter)} energy points, {len(cosine_swarm)} cosine points, "
    f"{n_depth_ready}/{len(depth)} depth benchmarks present)"
  )
  for key, val in depth.items():
    status = "MISSING" if val is None else f"{len(val['rows'])} rows from {val['source_files']}"
    print(f"  throughput_depth.{key}: {status}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
