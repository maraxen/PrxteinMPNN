"""Summarize the CORRECTED ProteinEBM JAX-vs-PyTorch throughput parity from the batch-swept
benchmark JSONs, for the parity/benchmark report.

Supersedes the stale single-batch (n_decoys=4) "11-92x" figures still hardcoded in
``scripts/ebm/render_parity_report_data.py`` and baked into ``ebm_parity_report_data.json``.
Those came from ``decoy_benchmark_full_engaging_pinned.json`` / ``decoy_benchmark_full.json``
(no batch sweep) and do NOT reflect real GPU-saturated throughput. The honest numbers come from
the batch-swept runs (batch_sizes [4,16,64,256]) which expose both the real speedup and the
PyTorch-OOM cells where JAX still runs.

Reads ``{decoy,ddg}_benchmark_full_L*.json`` from one or more platform directories (e.g. the
committed Blackwell ``outputs/ebm_benchmarks/`` and the pulled H200 ``outputs/ebm_benchmarks_h200/``),
pairs the JAX row against each PyTorch variant at matching (protein_length, batch_size), and reports
energy-throughput and score-gradient speedups plus OOM cells.

Speedup convention: ``speedup = pytorch_time / jax_time`` (>1 means JAX faster). "eager" is the fair
apples-to-apples baseline; "shipped" is the reference ProteinEBM repo's own implementation (the
real-world comparison); "compiled" is torch.compile.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("summarize_throughput_parity")

BENCHMARKS = ("decoy", "ddg")
LENGTH_FILES = ("L64-128", "L256", "L512")
PT_VARIANTS = ("eager", "shipped", "compiled")


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--platform", action="append", nargs=2, metavar=("NAME", "DIR"), required=True,
    help="Platform label + directory of batch-swept JSONs. Repeatable, e.g. "
    "--platform Blackwell outputs/ebm_benchmarks --platform H200 outputs/ebm_benchmarks_h200",
  )
  parser.add_argument("--out", type=Path, default=Path("outputs/ebm_benchmarks/throughput_parity_corrected.json"))
  return parser.parse_args()


def _load_rows(bench_dir: Path, benchmark: str) -> list[dict]:
  """Concatenate the results rows across the three length-bucket files for one benchmark."""
  rows: list[dict] = []
  for tag in LENGTH_FILES:
    path = bench_dir / f"{benchmark}_benchmark_full_{tag}.json"
    if not path.exists():
      log.warning("missing %s", path)
      continue
    payload = json.loads(path.read_text())
    # Two schemas exist: newer files wrap rows in {"meta", "results"}; older files are a bare list.
    raw_rows = payload["results"] if isinstance(payload, dict) else payload
    for r in raw_rows:
      r = dict(r)
      r["_source"] = path.name
      rows.append(r)
  return rows


def _cell_key(row: dict) -> tuple[int, int]:
  return int(row["protein_length"]), int(row["batch_size"])


def _summarize_benchmark(rows: list[dict]) -> list[dict]:
  """One entry per (length, batch): jax time, per-variant pytorch time + speedup, OOM flags."""
  jax_by_cell: dict[tuple[int, int], dict] = {}
  pt_by_cell: dict[tuple[int, int], dict[str, dict]] = defaultdict(dict)
  oom_cells: dict[tuple[int, int], str] = {}

  for row in rows:
    impl = row.get("impl")
    if impl == "jax":
      jax_by_cell[_cell_key(row)] = row
    elif impl == "pytorch":
      pt_by_cell[_cell_key(row)][row.get("pytorch_variant") or "?"] = row
    elif impl == "error":
      # OOM/crash cell: identify which framework from the error text.
      err = (row.get("error") or "")[:200]
      oom_cells[_cell_key(row)] = err

  cells = sorted(set(jax_by_cell) | set(pt_by_cell) | set(oom_cells))
  out: list[dict] = []
  for cell in cells:
    length, batch = cell
    jax = jax_by_cell.get(cell)
    entry: dict = {
      "protein_length": length,
      "batch_size": batch,
      "jax_energy_ms": jax["energy_wall_clock_mean_ms"] if jax else None,
      "jax_grad_ms": jax["score_grad_ms"] if jax else None,
      "variants": {},
    }
    for variant in PT_VARIANTS:
      pt = pt_by_cell.get(cell, {}).get(variant)
      if pt is None:
        continue
      v: dict = {
        "pytorch_energy_ms": pt["energy_wall_clock_mean_ms"],
        "pytorch_grad_ms": pt["score_grad_ms"],
      }
      if jax and jax["energy_wall_clock_mean_ms"] and pt["energy_wall_clock_mean_ms"]:
        v["energy_speedup"] = pt["energy_wall_clock_mean_ms"] / jax["energy_wall_clock_mean_ms"]
      if jax and jax["score_grad_ms"] and pt["score_grad_ms"]:
        v["grad_speedup"] = pt["score_grad_ms"] / jax["score_grad_ms"]
      entry["variants"][variant] = v
    if cell in oom_cells:
      err = oom_cells[cell]
      framework = "pytorch" if "PYTORCH" in err.upper() or "OutOfMemoryError" in err else "jax"
      entry["oom"] = {"framework": framework, "jax_survived": jax is not None, "error_head": err}
    out.append(entry)
  return out


def main() -> int:
  logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
  args = _parse_args()

  summary: dict = {"convention": "speedup = pytorch_time / jax_time (>1 = JAX faster)", "platforms": {}}
  for name, dir_str in args.platform:
    bench_dir = Path(dir_str)
    summary["platforms"][name] = {}
    for benchmark in BENCHMARKS:
      rows = _load_rows(bench_dir, benchmark)
      if not rows:
        continue
      cells = _summarize_benchmark(rows)
      summary["platforms"][name][benchmark] = cells

      # Console table
      log.info("\n=== %s / %s ===", name, benchmark)
      log.info("  L    bs    jax_ms   pt_eager  eSpd   gSpd(eager)  notes")
      for c in cells:
        eager = c["variants"].get("eager", {})
        espd = eager.get("energy_speedup")
        gspd = eager.get("grad_speedup")
        note = ""
        if "oom" in c:
          note = f"{c['oom']['framework']} OOM" + (" (JAX ok)" if c["oom"]["jax_survived"] else "")
        log.info(
          "  %-4d %-5d %-8s %-9s %-6s %-11s %s",
          c["protein_length"], c["batch_size"],
          f"{c['jax_energy_ms']:.2f}" if c["jax_energy_ms"] else "-",
          f"{eager.get('pytorch_energy_ms'):.2f}" if eager.get("pytorch_energy_ms") else "-",
          f"{espd:.2f}x" if espd else "-",
          f"{gspd:.2f}x" if gspd else "-",
          note,
        )

  # Headline ranges (eager, energy) across all non-OOM cells, per platform.
  for name, benches in summary["platforms"].items():
    espds = [
      v["energy_speedup"]
      for benches_cells in benches.values()
      for c in benches_cells
      for v in [c["variants"].get("eager", {})]
      if v.get("energy_speedup")
    ]
    if espds:
      log.info("\n[%s] energy speedup (eager) range: %.2fx - %.2fx across %d cells",
               name, min(espds), max(espds), len(espds))

  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(summary, indent=2))
  log.info("\nWrote %s", args.out)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
