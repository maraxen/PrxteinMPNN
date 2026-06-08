#!/usr/bin/env python3
"""Benchmark reporting tool — generates Markdown tables and CSV from benchmark JSONs.

Reads one or more benchmark JSON files (produced by bench_*_*.py adapters) and produces:
  - benchmark_report.md: Formatted tables grouped by hardware/precision/ligand_conditioning
  - benchmark_report.csv: Flat table, one row per result

Usage:
    uv run python scripts/benchmarks/bench_report.py \
        --input-json outputs/results/benchmarks/A100_combined.json \
        --output-dir reports/

    uv run python scripts/benchmarks/bench_report.py \
        --input-json A100_combined.json H100_combined.json \
        --output-dir reports/ \
        --hardware A100

Exit codes:
    0: SUCCESS
    1: FAILURE (missing file, JSON parse error, or output write error)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def load_benchmark_json(json_path: Path) -> list[dict[str, Any]]:
    """Load benchmark results from a JSON file, return list of result dicts."""
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {json_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {json_path}: {e}")
        raise

    # Handle both wrapper ({"results": [...]}) and direct list format
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    elif isinstance(data, list):
        return data
    else:
        logger.error(f"Unexpected JSON structure in {json_path}")
        raise ValueError(f"Expected list or dict with 'results' key, got {type(data)}")


def merge_results(json_paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Merge results from multiple JSON files.

    Returns: (all_results, capability_results_dict)
    capability_results_dict has keys: "temperature", "dedup", "mixed_length"
    """
    all_results = []
    capability_results: dict[str, Any] = {}

    for json_path in json_paths:
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            logger.error(f"File not found: {json_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {json_path}: {e}")
            raise

        # Extract results
        if isinstance(data, dict) and "results" in data:
            all_results.extend(data["results"])
        elif isinstance(data, list):
            all_results.extend(data)
        else:
            logger.error(f"Unexpected JSON structure in {json_path}")
            raise ValueError(f"Expected list or dict with 'results' key, got {type(data)}")

        # Extract capability results if present
        if isinstance(data, dict) and "capability_results" in data:
            cap_results = data["capability_results"]
            if isinstance(cap_results, dict):
                capability_results.update(cap_results)

    return all_results, capability_results


def compute_speedup_vs_pytorch(
    results: list[dict[str, Any]], hardware: str, seq_len: int, batch_size: int,
    precision: str, ligand_conditioning: bool
) -> Optional[float]:
    """Find PyTorch latency and compute speedup factor.

    Returns: speedup = pytorch_latency / model_latency, or None if PyTorch not available.
    """
    pytorch_latency = None

    for result in results:
        if (result.get("hardware") == hardware and
            result.get("seq_len") == seq_len and
            result.get("batch_size") == batch_size and
            result.get("precision") == precision and
            result.get("ligand_conditioning") == ligand_conditioning and
            result.get("model") == "ligandmpnn_pytorch"):
            pytorch_latency = result.get("latency_median_ms")
            break

    return None  # PyTorch baseline not found


def generate_capability_comparison_section(
    capability_results: dict[str, Any], hardware: str
) -> list[str]:
    """Generate capability comparison section for markdown report.

    Reads capability_results dict with keys "temperature", "dedup", "mixed_length"
    and generates a table comparing prxteinmpnn vs baselines.

    Returns list of markdown lines.
    """
    lines = []
    lines.append("## Capability Comparison\n")
    lines.append("Capability-specific benchmarks comparing prxteinmpnn vs ColabDesign and PyTorch baselines.\n")

    # Temperature array capability
    if "temperature" in capability_results and capability_results["temperature"]:
        lines.append("### Temperature Array (M-temperature JIT sweep)\n")
        lines.append("| Capability | prxteinmpnn (ms) | ColabDesign (ms) | PyTorch (ms) | Speedup vs PT |\n")
        lines.append("|---|---|---|---|---|\n")

        temp_data = capability_results["temperature"]
        if isinstance(temp_data, dict) and "cells" in temp_data:
            cells = temp_data["cells"]
        elif isinstance(temp_data, list):
            cells = temp_data
        else:
            cells = []

        # Build representative rows for M=1, M=4, M=8
        for m_val in [1, 4, 8]:
            cell = next((c for c in cells if c.get("m") == m_val), None)
            if cell:
                prx = cell.get("prxteinmpnn_latency_ms")
                cola = cell.get("colabdesign_latency_ms")
                pt = cell.get("pytorch_latency_ms")
                speedup = (pt / prx) if (pt and prx and prx > 0) else None

                prx_str = f"{prx:.1f}" if prx else "—"
                cola_str = f"{cola:.1f}" if cola else "—"
                pt_str = f"{pt:.1f}" if pt else "—"
                speedup_str = f"{speedup:.2f}×" if speedup else "—"

                lines.append(f"| Temp Array M={m_val} | {prx_str} | {cola_str} | {pt_str} | {speedup_str} |\n")

        lines.append("\n")

    # DedupGather capability
    if "dedup" in capability_results and capability_results["dedup"]:
        lines.append("### DedupGather Heterogeneous Batch (K-unique proportional speedup)\n")
        lines.append("| K unique | N total | dedup_ratio | prxteinmpnn (ms) | PyTorch (ms) | Speedup vs PT |\n")
        lines.append("|---|---|---|---|---|---|\n")

        dedup_data = capability_results["dedup"]
        if isinstance(dedup_data, dict) and "cells" in dedup_data:
            cells = dedup_data["cells"]
        elif isinstance(dedup_data, list):
            cells = dedup_data
        else:
            cells = []

        # Show key K values
        for k_val in [1, 4, 8, 16, 32]:
            cell = next((c for c in cells if c.get("k") == k_val), None)
            if cell:
                k = cell.get("k")
                n = cell.get("n")
                ratio = cell.get("dedup_ratio")
                prx = cell.get("prxteinmpnn_latency_ms")
                pt = cell.get("pytorch_latency_ms")
                speedup = (pt / prx) if (pt and prx and prx > 0) else None

                prx_str = f"{prx:.1f}" if prx else "—"
                pt_str = f"{pt:.1f}" if pt else "—"
                speedup_str = f"{speedup:.2f}×" if speedup else "—"
                ratio_str = f"{ratio:.2f}" if ratio else "—"

                lines.append(f"| {k} | {n} | {ratio_str} | {prx_str} | {pt_str} | {speedup_str} |\n")

        lines.append("\n")

    # Mixed-length capability
    if "mixed_length" in capability_results and capability_results["mixed_length"]:
        lines.append("### Mixed-Length Heterogeneous Batch\n")
        lines.append("| Config | prxteinmpnn (ms) | ColabDesign (ms) | PyTorch (ms) | Speedup vs PT |\n")
        lines.append("|---|---|---|---|---|\n")

        mixed_data = capability_results["mixed_length"]
        if isinstance(mixed_data, dict) and "cells" in mixed_data:
            cells = mixed_data["cells"]
        elif isinstance(mixed_data, list):
            cells = mixed_data
        else:
            cells = []

        # Show first few representative results
        for cell in cells[:5]:
            config = cell.get("config", "unknown")
            prx = cell.get("prxteinmpnn_latency_ms")
            cola = cell.get("colabdesign_latency_ms")
            pt = cell.get("pytorch_latency_ms")
            speedup = (pt / prx) if (pt and prx and prx > 0) else None

            prx_str = f"{prx:.1f}" if prx else "—"
            cola_str = f"{cola:.1f}" if cola else "—"
            pt_str = f"{pt:.1f}" if pt else "—"
            speedup_str = f"{speedup:.2f}×" if speedup else "—"

            lines.append(f"| {config} | {prx_str} | {cola_str} | {pt_str} | {speedup_str} |\n")

        lines.append("\n")

    return lines


def generate_markdown_report(
    results: list[dict[str, Any]], output_path: Path, capability_results: dict[str, Any] | None = None
) -> None:
    """Generate Markdown report with tables grouped by hardware, precision, ligand_conditioning."""

    # Group results by (hardware, precision, ligand_conditioning)
    groups: dict[tuple[str, str, bool], list[dict[str, Any]]] = {}
    for result in results:
        key = (
            result.get("hardware", "unknown"),
            result.get("precision", "unknown"),
            result.get("ligand_conditioning", False),
        )
        if key not in groups:
            groups[key] = []
        groups[key].append(result)

    lines = []
    lines.append("# Benchmark Report\n")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")

    # Group by hardware first
    hardware_groups: dict[str, list[tuple[tuple[str, str, bool], list[dict[str, Any]]]]] = {}
    for key, group_results in groups.items():
        hardware = key[0]
        if hardware not in hardware_groups:
            hardware_groups[hardware] = []
        hardware_groups[hardware].append((key, group_results))

    for hardware in sorted(hardware_groups.keys()):
        lines.append(f"## Hardware: {hardware}\n")

        for (hardware, precision, ligand_conditioning), group_results in sorted(hardware_groups[hardware]):
            # Extract unique (seq_len, batch_size) combinations and models
            cells: dict[tuple[int, int], dict[str, Optional[float]]] = {}
            models = set()

            for result in group_results:
                seq_len = result.get("seq_len", 0)
                batch_size = result.get("batch_size", 0)
                model = result.get("model", "unknown")
                latency_ms = result.get("latency_median_ms")

                models.add(model)

                if (seq_len, batch_size) not in cells:
                    cells[(seq_len, batch_size)] = {}
                cells[(seq_len, batch_size)][model] = latency_ms

            models_sorted = sorted(models)

            ligand_str = "True" if ligand_conditioning else "False"
            lines.append(f"### Precision: {precision} | Ligand: {ligand_str}\n")

            # Latency table
            lines.append("#### Latency (ms)\n")
            header = ["seq_len", "batch"]
            header.extend(models_sorted)
            header.append("speedup vs PT")
            lines.append("| " + " | ".join(header) + " |\n")
            lines.append("|" + "|".join(["-" * (len(h) + 2) for h in header]) + "|\n")

            for (seq_len, batch_size) in sorted(cells.keys()):
                row = [str(seq_len), str(batch_size)]

                for model in models_sorted:
                    latency = cells[(seq_len, batch_size)].get(model)
                    if latency is not None:
                        row.append(f"{latency:.1f}")
                    else:
                        row.append("—")

                # Speedup vs PyTorch
                pytorch_latency = cells[(seq_len, batch_size)].get("ligandmpnn_pytorch")
                prxtein_latency = cells[(seq_len, batch_size)].get("prxteinmpnn_jax")

                if pytorch_latency is not None and prxtein_latency is not None and prxtein_latency > 0:
                    speedup = pytorch_latency / prxtein_latency
                    row.append(f"{speedup:.2f}×")
                else:
                    row.append("—")

                lines.append("| " + " | ".join(row) + " |\n")

            lines.append("\n*latency = median of warm calls; speedup = ligandmpnn_pytorch / prxteinmpnn_jax (higher is faster)*\n\n")

            # Compile time table
            lines.append("#### Compile Time Cold (s)\n")
            header = ["seq_len", "batch"]
            header.extend(models_sorted)
            lines.append("| " + " | ".join(header) + " |\n")
            lines.append("|" + "|".join(["-" * (len(h) + 2) for h in header]) + "|\n")

            compile_cells: dict[tuple[int, int], dict[str, Optional[float]]] = {}
            for result in group_results:
                seq_len = result.get("seq_len", 0)
                batch_size = result.get("batch_size", 0)
                model = result.get("model", "unknown")
                compile_s = result.get("compile_time_cold_s")

                if (seq_len, batch_size) not in compile_cells:
                    compile_cells[(seq_len, batch_size)] = {}
                compile_cells[(seq_len, batch_size)][model] = compile_s

            for (seq_len, batch_size) in sorted(compile_cells.keys()):
                row = [str(seq_len), str(batch_size)]
                for model in models_sorted:
                    compile_s = compile_cells[(seq_len, batch_size)].get(model)
                    if compile_s is not None:
                        row.append(f"{compile_s:.2f}")
                    else:
                        row.append("—")
                lines.append("| " + " | ".join(row) + " |\n")

            lines.append("\n")

            # Throughput table
            lines.append("#### Throughput (seq/s)\n")
            header = ["seq_len", "batch"]
            header.extend(models_sorted)
            lines.append("| " + " | ".join(header) + " |\n")
            lines.append("|" + "|".join(["-" * (len(h) + 2) for h in header]) + "|\n")

            throughput_cells: dict[tuple[int, int], dict[str, Optional[float]]] = {}
            for result in group_results:
                seq_len = result.get("seq_len", 0)
                batch_size = result.get("batch_size", 0)
                model = result.get("model", "unknown")
                throughput = result.get("throughput_seq_per_s")

                if (seq_len, batch_size) not in throughput_cells:
                    throughput_cells[(seq_len, batch_size)] = {}
                throughput_cells[(seq_len, batch_size)][model] = throughput

            for (seq_len, batch_size) in sorted(throughput_cells.keys()):
                row = [str(seq_len), str(batch_size)]
                for model in models_sorted:
                    throughput = throughput_cells[(seq_len, batch_size)].get(model)
                    if throughput is not None:
                        row.append(f"{throughput:.1f}")
                    else:
                        row.append("—")
                lines.append("| " + " | ".join(row) + " |\n")

            lines.append("\n")

            # Memory table
            lines.append("#### Peak GPU Memory (GB)\n")
            header = ["seq_len", "batch"]
            header.extend(models_sorted)
            lines.append("| " + " | ".join(header) + " |\n")
            lines.append("|" + "|".join(["-" * (len(h) + 2) for h in header]) + "|\n")

            memory_cells: dict[tuple[int, int], dict[str, Optional[float]]] = {}
            for result in group_results:
                seq_len = result.get("seq_len", 0)
                batch_size = result.get("batch_size", 0)
                model = result.get("model", "unknown")
                memory = result.get("peak_gpu_memory_gb")

                if (seq_len, batch_size) not in memory_cells:
                    memory_cells[(seq_len, batch_size)] = {}
                memory_cells[(seq_len, batch_size)][model] = memory

            for (seq_len, batch_size) in sorted(memory_cells.keys()):
                row = [str(seq_len), str(batch_size)]
                for model in models_sorted:
                    memory = memory_cells[(seq_len, batch_size)].get(model)
                    if memory is not None:
                        row.append(f"{memory:.2f}")
                    else:
                        row.append("—")
                lines.append("| " + " | ".join(row) + " |\n")

            lines.append("\n")

    # Add capability comparison section if available
    if capability_results:
        hardware = results[0].get("hardware", "unknown") if results else "unknown"
        capability_lines = generate_capability_comparison_section(capability_results, hardware)
        lines.extend(capability_lines)

    with open(output_path, "w") as f:
        f.writelines(lines)


def generate_csv_report(results: list[dict[str, Any]], output_path: Path) -> None:
    """Generate flat CSV with one row per result."""

    # Define field order
    fieldnames = [
        "hardware",
        "model",
        "seq_len",
        "batch_size",
        "precision",
        "ligand_conditioning",
        "compile_time_cold_s",
        "compile_time_warm_s",
        "latency_median_ms",
        "latency_p95_ms",
        "latency_per_residue_us",
        "throughput_seq_per_s",
        "peak_gpu_memory_gb",
        "n_timed",
        "speedup_vs_pytorch",
    ]

    # Pre-compute PyTorch latencies for speedup calculation
    pytorch_baseline: dict[tuple[str, int, int, str, bool], float] = {}
    for result in results:
        if result.get("model") == "ligandmpnn_pytorch":
            key = (
                result.get("hardware", "unknown"),
                result.get("seq_len", 0),
                result.get("batch_size", 0),
                result.get("precision", "unknown"),
                result.get("ligand_conditioning", False),
            )
            lat = result.get("latency_median_ms")
            if lat is not None:
                pytorch_baseline[key] = lat

    rows = []
    for result in results:
        row = {}

        # Copy standard fields
        for field in fieldnames[:-1]:  # All except speedup_vs_pytorch
            row[field] = result.get(field, "")

        # Compute speedup
        key = (
            result.get("hardware", "unknown"),
            result.get("seq_len", 0),
            result.get("batch_size", 0),
            result.get("precision", "unknown"),
            result.get("ligand_conditioning", False),
        )

        model_latency = result.get("latency_median_ms")

        if model_latency is not None and model_latency > 0 and key in pytorch_baseline:
            pytorch_latency = pytorch_baseline[key]
            if pytorch_latency > 0:
                speedup = pytorch_latency / model_latency
                row["speedup_vs_pytorch"] = f"{speedup:.2f}"
            else:
                row["speedup_vs_pytorch"] = ""
        else:
            row["speedup_vs_pytorch"] = ""

        # For PyTorch itself, speedup = 1.0
        if result.get("model") == "ligandmpnn_pytorch" and model_latency is not None and model_latency > 0:
            row["speedup_vs_pytorch"] = "1.0"

        rows.append(row)

    # Sort by hardware, model, seq_len, batch_size, precision, ligand_conditioning
    rows.sort(key=lambda r: (
        r.get("hardware", ""),
        r.get("model", ""),
        int(r.get("seq_len", 0)),
        int(r.get("batch_size", 0)),
        r.get("precision", ""),
        str(r.get("ligand_conditioning", False)),
    ))

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate benchmark reports from JSON files."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        nargs="+",
        required=True,
        help="One or more benchmark JSON files to process.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Output directory for reports (default: reports/).",
    )
    parser.add_argument(
        "--hardware",
        type=str,
        default=None,
        help="Optional hardware label override.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(name)s: %(message)s",
    )

    try:
        # Load and merge results
        logger.info(f"Loading {len(args.input_json)} JSON file(s)...")
        results, capability_results = merge_results(args.input_json)
        logger.info(f"Loaded {len(results)} benchmark results.")
        if capability_results:
            logger.info(f"Loaded capability results: {', '.join(capability_results.keys())}")

        if not results:
            logger.warning("No results loaded, skipping report generation.")
            return 0

        # Create output directory
        args.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate reports
        md_path = args.output_dir / "benchmark_report.md"
        csv_path = args.output_dir / "benchmark_report.csv"

        logger.info(f"Generating Markdown report: {md_path}")
        generate_markdown_report(results, md_path, capability_results=capability_results)

        logger.info(f"Generating CSV report: {csv_path}")
        generate_csv_report(results, csv_path)

        # Summary
        hardware_targets = set(r.get("hardware") for r in results)
        models = set(r.get("model") for r in results)
        cells_per_model = {}
        for result in results:
            model = result.get("model", "unknown")
            cells_per_model[model] = cells_per_model.get(model, 0) + 1

        logger.info("")
        logger.info("=" * 60)
        logger.info("BENCHMARK REPORT SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Hardware targets: {', '.join(sorted(hardware_targets))}")
        logger.info(f"Models: {', '.join(sorted(models))}")
        for model in sorted(models):
            logger.info(f"  {model}: {cells_per_model[model]} cells")
        logger.info(f"Total results: {len(results)}")
        logger.info("")
        logger.info(f"Output files:")
        logger.info(f"  Markdown: {md_path}")
        logger.info(f"  CSV:      {csv_path}")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
