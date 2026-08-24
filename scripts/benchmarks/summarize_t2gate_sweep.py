"""Aggregate the T2.GATE GPU sweep into a per-architecture verdict.

Reads the per-(gpu, repeat) result JSONs produced by
scripts/slurm/t2gate_gpu_sweep.slurm and reduces them the way the gate should be
read: parity legs are all-or-nothing, and the throughput ratio is reported as a
SPREAD across repeats rather than a single point.

Why the spread matters (see .praxia/docs/audits/260817_aminx-deeper-work-assessment.md
§3.2): the CPU sibling of this bench reduces across cases with max() over
microsecond-scale timings, and its verdict proved irreproducible -- five observations
spanned 1.01-1.64 and straddled all three of the sidecar's outcome bands. Reporting a
single run of a max()-reduced metric is therefore not evidence of anything. This script
exists so the gate is read off min/mean/max across repeats, and so a per-architecture
verdict is only claimed when EVERY repeat agrees.

Usage:
    uv run python scripts/benchmarks/summarize_t2gate_sweep.py --results-dir results
    uv run python scripts/benchmarks/summarize_t2gate_sweep.py --results-dir results \
        --out results/t2gate_sweep_summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# T2.GATE's stated DoD bar for the adapter-vs-legacy throughput ratio.
PASS_BAR = 1.10
# The bench sidecar's residual (fail) threshold, kept for band labelling only.
MARGINAL_BAR = 1.50

# Architecture labels, ordered oldest -> newest. Ordering is load-bearing for
# reading the table: the whole point of the sweep is that the ratio is an XLA
# backend property, so a trend across SM generations is the signal.
ARCH = {
  "titanrtx": ("TITAN RTX", "Turing", 75),
  "a100": ("A100", "Ampere", 80),
  "l40s": ("L40S", "Ada", 89),
  "h100": ("H100", "Hopper", 90),
  "h200": ("H200", "Hopper", 90),
  "blackwell": ("RTX PRO 6000", "Blackwell", 120),
}

FNAME = re.compile(r"t2gate_(?P<gpu>[a-z0-9_]+)_rep(?P<rep>\d+)\.json$")


def band(ratio: float) -> str:
  """Label a ratio against the gate's bands."""
  if ratio <= PASS_BAR:
    return "pass"
  if ratio <= MARGINAL_BAR:
    return "marginal"
  return "fail"


def load(results_dir: Path) -> dict[str, list[dict]]:
  """Group result JSONs by GPU label, sorted by repeat index."""
  grouped: dict[str, list[dict]] = {}
  for path in sorted(results_dir.glob("t2gate_*.json")):
    m = FNAME.search(path.name)
    if not m:
      logger.warning("skipping unrecognised filename: %s", path.name)
      continue
    payload = json.loads(path.read_text())
    payload["_rep"] = int(m.group("rep"))
    payload["_path"] = str(path)
    grouped.setdefault(m.group("gpu"), []).append(payload)
  for runs in grouped.values():
    runs.sort(key=lambda r: r["_rep"])
  return grouped


def summarize(grouped: dict[str, list[dict]]) -> dict:
  """Reduce to a per-architecture verdict plus an overall one."""
  rows = []
  for gpu, runs in grouped.items():
    name, arch, sm = ARCH.get(gpu, (gpu, "unknown", 0))
    ratios = [r["max_adapter_vs_legacy_throughput_ratio"] for r in runs]
    parity = [bool(r.get("all_recompile_parity")) for r in runs]
    bands = [band(x) for x in ratios]
    rows.append({
      "gpu": gpu,
      "name": name,
      "arch": arch,
      "sm": sm,
      "n_reps": len(runs),
      "recompile_parity_all": all(parity),
      "ratio_min": min(ratios),
      "ratio_mean": statistics.fmean(ratios),
      "ratio_max": max(ratios),
      "ratio_spread": max(ratios) - min(ratios),
      "bands": bands,
      # A verdict is only claimed when every repeat lands in the same band --
      # otherwise the honest answer is that this architecture straddles a bar.
      "verdict": bands[0] if len(set(bands)) == 1 else "straddles:" + "/".join(sorted(set(bands))),
      "ratios": ratios,
    })
  rows.sort(key=lambda r: (r["sm"], r["gpu"]))

  parity_everywhere = all(r["recompile_parity_all"] for r in rows)
  clean_pass = [r for r in rows if r["verdict"] == "pass"]
  return {
    "architectures": rows,
    "n_architectures": len(rows),
    "n_runs": sum(r["n_reps"] for r in rows),
    "recompile_parity_all_architectures": parity_everywhere,
    "architectures_clean_pass": [r["gpu"] for r in clean_pass],
    "architectures_not_clean_pass": [r["gpu"] for r in rows if r["verdict"] != "pass"],
    "pass_bar": PASS_BAR,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--results-dir", type=Path, default=Path("results"))
  parser.add_argument("--out", type=Path, default=None)
  parser.add_argument("--verbose", action="store_true")
  args = parser.parse_args()
  logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

  grouped = load(args.results_dir)
  if not grouped:
    logger.error("no t2gate_*.json found under %s", args.results_dir)
    sys.exit(1)

  summary = summarize(grouped)

  print(f"\nT2.GATE GPU sweep — {summary['n_runs']} runs across "
        f"{summary['n_architectures']} architectures (pass bar {PASS_BAR})\n")
  print(f"{'GPU':<14}{'arch':<10}{'SM':>4}  {'reps':>4}  {'min':>7}{'mean':>8}{'max':>8}"
        f"{'spread':>8}  {'parity':<7} verdict")
  for r in summary["architectures"]:
    print(f"{r['name']:<14}{r['arch']:<10}{r['sm']:>4}  {r['n_reps']:>4}  "
          f"{r['ratio_min']:>7.4f}{r['ratio_mean']:>8.4f}{r['ratio_max']:>8.4f}"
          f"{r['ratio_spread']:>8.4f}  {str(r['recompile_parity_all']):<7} {r['verdict']}")

  print(f"\nrecompile parity on every architecture: "
        f"{summary['recompile_parity_all_architectures']}")
  if summary["architectures_not_clean_pass"]:
    print(f"NOT a clean pass on: {', '.join(summary['architectures_not_clean_pass'])}")

  if args.out:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
  main()
