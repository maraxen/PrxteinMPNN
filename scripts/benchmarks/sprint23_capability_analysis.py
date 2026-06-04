#!/usr/bin/env python3
"""Sprint 23 capability benchmark analysis.

Reads temperature-array, dedup-hetero, and mixed-length JSON files produced
by the Sprint 23 cluster jobs and generates a comprehensive Markdown report.

Usage:
    python3 scripts/benchmarks/sprint23_capability_analysis.py \
        --results-dir outputs/results/benchmarks/ \
        --output-dir reports/

Exit codes:
    0: SUCCESS
    1: FAILURE
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

HARDWARE_DISPLAY = {
    "H200": "H200",
    "A100": "A100",
    "L40s": "L40s",
    "Blackwell": "Blackwell (SM120)",
    "Blackwell_SM120": "Blackwell (SM120)",
}

HARDWARE_ORDER = ["H200", "A100", "L40s", "Blackwell", "Blackwell_SM120"]


def load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def find_latest_temperature_files(results_dir: Path) -> dict[str, Path]:
    """Find the latest temperature benchmark JSON per hardware.

    Temperature files land in the same results_dir (pulled from logs/slurm/).
    Naming: bench_temperature_{hw}_{jobid}.json — pick highest job ID per hardware.
    """
    hw_files: dict[str, list[tuple[int, Path]]] = {}
    for p in results_dir.glob("bench_temperature_*.json"):
        parts = p.stem.split("_")
        if len(parts) < 4:
            continue
        try:
            job_id = int(parts[-1])
        except ValueError:
            continue
        hw_raw = "_".join(parts[2:-1])  # e.g. "h200", "a100", "l40s", "blackwell"
        hw_normalized = hw_raw.lower()
        hw_map = {
            "h200": "H200",
            "a100": "A100",
            "l40s": "L40s",
            "blackwell": "Blackwell",
        }
        hw = hw_map.get(hw_normalized, hw_raw.upper())
        hw_files.setdefault(hw, []).append((job_id, p))

    latest: dict[str, Path] = {}
    for hw, entries in hw_files.items():
        entries.sort(key=lambda x: x[0])
        _job_id, path = entries[-1]
        # Skip stub files (< 1 KB means job failed early)
        if path.stat().st_size > 1024:
            latest[hw] = path

    return latest


def load_dedup_files(results_dir: Path) -> dict[str, dict[str, Any]]:
    """Load *_dedup_hetero_bench.json files keyed by hardware name."""
    result = {}
    for p in results_dir.glob("*_dedup_hetero_bench.json"):
        hw = p.stem.replace("_dedup_hetero_bench", "")
        data = load_json(p)
        result[hw] = data
    return result


def load_mixed_length_files(results_dir: Path) -> dict[str, dict[str, Any]]:
    """Load *_mixed_length_bench.json files keyed by hardware name."""
    result = {}
    for p in results_dir.glob("*_mixed_length_bench.json"):
        hw = p.stem.replace("_mixed_length_bench", "")
        data = load_json(p)
        result[hw] = data
    return result


def hw_label(hw: str) -> str:
    return HARDWARE_DISPLAY.get(hw, hw)


def sorted_hardware(keys: list[str]) -> list[str]:
    order_map = {h: i for i, h in enumerate(HARDWARE_ORDER)}
    return sorted(keys, key=lambda h: order_map.get(h, 99))


def generate_dedup_section(dedup_data: dict[str, dict[str, Any]]) -> list[str]:
    """DedupGather capability: K-proportional speedup table per hardware."""
    lines = []
    lines.append("## DedupGather Heterogeneous Batch\n\n")
    lines.append(
        "K unique structures deduplicated before scoring; N=32 total. "
        "prxteinmpnn scores only unique structures and scatters results, "
        "while ColabDesign and PyTorch score all N.\n\n"
    )

    k_show = [1, 2, 4, 8, 16, 32]

    for hw in sorted_hardware(list(dedup_data.keys())):
        data = dedup_data[hw]
        cells = data.get("cells", [])
        if not cells:
            continue

        lines.append(f"### {hw_label(hw)}\n\n")
        lines.append(
            "| K unique | dedup_ratio | prxteinmpnn (ms) | ColabDesign (ms) "
            "| PyTorch (ms) | Speedup vs CD | Speedup vs PT |\n"
        )
        lines.append("|---|---|---|---|---|---|---|\n")

        for k in k_show:
            cell = next((c for c in cells if c.get("k") == k), None)
            if not cell:
                continue
            prx = cell.get("prxteinmpnn_latency_ms")
            cd = cell.get("colabdesign_latency_ms")
            pt = cell.get("pytorch_latency_ms")
            ratio = cell.get("dedup_ratio")

            prx_s = f"{prx:.2f}" if prx else "—"
            cd_s = f"{cd:.2f}" if cd else "—"
            pt_s = f"{pt:.2f}" if pt else "—"
            ratio_s = f"{ratio:.4f}" if ratio else "—"
            spd_cd = f"{cd / prx:.1f}×" if (cd and prx and prx > 0) else "—"
            spd_pt = f"{pt / prx:.1f}×" if (pt and prx and prx > 0) else "—"

            lines.append(
                f"| {k} | {ratio_s} | {prx_s} | {cd_s} | {pt_s} | {spd_cd} | {spd_pt} |\n"
            )

        lines.append("\n")

    # Cross-hardware summary at K=1 (best case) and K=32 (worst case)
    lines.append("### Cross-Hardware Summary (K=1 and K=32)\n\n")
    lines.append(
        "| Hardware | K | prxteinmpnn (ms) | PyTorch (ms) | Speedup vs PT |\n"
    )
    lines.append("|---|---|---|---|---|\n")

    for hw in sorted_hardware(list(dedup_data.keys())):
        data = dedup_data[hw]
        cells = data.get("cells", [])
        for k in [1, 32]:
            cell = next((c for c in cells if c.get("k") == k), None)
            if not cell:
                continue
            prx = cell.get("prxteinmpnn_latency_ms")
            pt = cell.get("pytorch_latency_ms")
            prx_s = f"{prx:.2f}" if prx else "—"
            pt_s = f"{pt:.2f}" if pt else "—"
            spd_pt = f"{pt / prx:.1f}×" if (pt and prx and prx > 0) else "—"
            lines.append(f"| {hw_label(hw)} | {k} | {prx_s} | {pt_s} | {spd_pt} |\n")

    lines.append("\n")
    return lines


def generate_mixed_length_section(mixed_data: dict[str, dict[str, Any]]) -> list[str]:
    """Mixed-length batch: flat scalar results per hardware."""
    lines = []
    lines.append("## Mixed-Length Heterogeneous Batch\n\n")
    lines.append(
        "Batch of 4 sequences with lengths [76, 150, 300, 500]. "
        "prxteinmpnn packs into a single padded batch; "
        "ColabDesign and PyTorch run sequentially.\n\n"
    )

    lines.append(
        "| Hardware | batch_lengths | prxteinmpnn (ms) | ColabDesign seq (ms) "
        "| PyTorch padded (ms) | PyTorch seq (ms) "
        "| Speedup vs PT padded | Speedup vs PT seq |\n"
    )
    lines.append("|---|---|---|---|---|---|---|---|\n")

    for hw in sorted_hardware(list(mixed_data.keys())):
        data = mixed_data[hw]
        lengths = data.get("batch_lengths", [])
        prx = data.get("mixed_latency_ms")
        cd = data.get("colabdesign_sequential_latency_ms")
        pt_padded = data.get("pytorch_padded_latency_ms")
        pt_seq = data.get("pytorch_sequential_latency_ms")

        lengths_s = str(lengths) if lengths else "—"
        prx_s = f"{prx:.2f}" if prx else "—"
        cd_s = f"{cd:.2f}" if cd else "—"
        pt_padded_s = f"{pt_padded:.2f}" if pt_padded else "—"
        pt_seq_s = f"{pt_seq:.2f}" if pt_seq else "—"
        spd_pad = f"{pt_padded / prx:.1f}×" if (pt_padded and prx and prx > 0) else "—"
        spd_seq = f"{pt_seq / prx:.1f}×" if (pt_seq and prx and prx > 0) else "—"

        lines.append(
            f"| {hw_label(hw)} | {lengths_s} | {prx_s} | {cd_s} "
            f"| {pt_padded_s} | {pt_seq_s} | {spd_pad} | {spd_seq} |\n"
        )

    lines.append("\n")
    return lines


def generate_temperature_section(temp_files: dict[str, Path]) -> list[str]:
    """Temperature-array sweep: prxteinmpnn per-temp latency across M and hardware."""
    lines = []
    lines.append("## Temperature Array Sweep\n\n")
    lines.append(
        "M simultaneous temperatures JIT-compiled into a single forward pass. "
        "ColabDesign and PyTorch baselines were not run in this sprint "
        "(they require M sequential calls; dedup section covers that comparison).\n\n"
    )

    m_show = [1, 2, 4, 8]
    tasks_show = ["ar_sample", "score_conditional"]

    for task in tasks_show:
        task_label = "Autoregressive Sample" if task == "ar_sample" else "Score Conditional"
        lines.append(f"### Task: {task_label}\n\n")
        lines.append(
            "Latency is per-temperature (total / M), seq_len=76, batch=1.\n\n"
        )
        lines.append("| Hardware | M | prxteinmpnn per-temp (ms) | prxteinmpnn total (ms) |\n")
        lines.append("|---|---|---|---|\n")

        for hw in sorted_hardware(list(temp_files.keys())):
            data = load_json(temp_files[hw])
            cells = data.get("cells", [])
            hw_cells = [
                c for c in cells
                if c.get("task") == task and c.get("seq_len") == 76 and c.get("batch_size") == 1
            ]

            for m in m_show:
                cell = next((c for c in hw_cells if c.get("m_value") == m), None)
                if not cell:
                    continue
                total = cell.get("prxteinmpnn_latency_median_ms")
                per_temp = cell.get("prxteinmpnn_latency_per_temp_ms")
                total_s = f"{total:.2f}" if total else "—"
                per_s = f"{per_temp:.2f}" if per_temp else "—"
                lines.append(f"| {hw_label(hw)} | {m} | {per_s} | {total_s} |\n")

        lines.append("\n")

    # Scaling efficiency table
    lines.append("### M-Scaling Efficiency (ar_sample, seq_len=76, batch=1)\n\n")
    lines.append(
        "Ideal scaling: per-temp latency is constant as M increases "
        "(M temperatures compiled as a single JIT call).\n\n"
    )
    lines.append("| Hardware | M=1 (ms) | M=2 (ms) | M=4 (ms) | M=8 (ms) | Speedup per-temp at M=8 (ideal: 8.0×) |\n")
    lines.append("|---|---|---|---|---|---|\n")

    for hw in sorted_hardware(list(temp_files.keys())):
        data = load_json(temp_files[hw])
        cells = data.get("cells", [])
        hw_cells = [
            c for c in cells
            if c.get("task") == "ar_sample"
            and c.get("seq_len") == 76
            and c.get("batch_size") == 1
        ]
        row_vals: dict[int, float | None] = {}
        for m in [1, 2, 4, 8]:
            cell = next((c for c in hw_cells if c.get("m_value") == m), None)
            row_vals[m] = cell.get("prxteinmpnn_latency_per_temp_ms") if cell else None

        v1, v2, v4, v8 = row_vals.get(1), row_vals.get(2), row_vals.get(4), row_vals.get(8)
        # Speedup = per-temp M=1 / per-temp M=8; ideal is 8× (perfect linear scaling)
        speedup = f"{v1 / v8:.2f}×" if (v8 and v1 and v8 > 0) else "—"
        lines.append(
            f"| {hw_label(hw)} "
            f"| {f'{v1:.2f}' if v1 else '—'} "
            f"| {f'{v2:.2f}' if v2 else '—'} "
            f"| {f'{v4:.2f}' if v4 else '—'} "
            f"| {f'{v8:.2f}' if v8 else '—'} "
            f"| {speedup} |\n"
        )

    lines.append("\n")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Sprint 23 capability analysis report.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("outputs/results/benchmarks/"),
        help="Directory containing benchmark JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/"),
        help="Output directory.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(name)s: %(message)s",
    )

    results_dir = args.results_dir
    if not results_dir.is_dir():
        logger.error(f"results-dir not found: {results_dir}")
        return 1

    logger.info("Loading capability benchmark files...")

    temp_files = find_latest_temperature_files(results_dir)
    logger.info(f"Temperature files: {list(temp_files.keys())}")

    dedup_data = load_dedup_files(results_dir)
    logger.info(f"Dedup files: {list(dedup_data.keys())}")

    mixed_data = load_mixed_length_files(results_dir)
    logger.info(f"Mixed-length files: {list(mixed_data.keys())}")

    lines = []
    lines.append("# Sprint 23 Capability Benchmark Report\n\n")
    lines.append(
        "Sprint 23 introduced three new capability benchmarks demonstrating "
        "prxteinmpnn's architectural advantages over PyTorch and ColabDesign baselines.\n\n"
    )
    lines.append("---\n\n")

    lines.extend(generate_dedup_section(dedup_data))
    lines.append("---\n\n")
    lines.extend(generate_mixed_length_section(mixed_data))
    lines.append("---\n\n")
    lines.extend(generate_temperature_section(temp_files))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "sprint23_capability_report.md"
    with open(out_path, "w") as f:
        f.writelines(lines)

    logger.info(f"Written: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
