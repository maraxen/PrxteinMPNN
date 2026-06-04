#!/usr/bin/env python3
"""Test bench_temperature_array.py: temperature sweep benchmark."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).parents[2]
BENCH_SCRIPT = PROJECT_ROOT / "scripts" / "benchmarks" / "bench_temperature_array.py"


class TestBenchTemperatureArray:
    """Verify bench_temperature_array.py exists and runs."""

    def test_script_exists(self) -> None:
        """Temperature array script must exist."""
        assert BENCH_SCRIPT.exists(), f"Script not found: {BENCH_SCRIPT}"

    def test_dry_run_succeeds(self) -> None:
        """--dry-run should exit 0 without running benchmarks."""
        result = subprocess.run(
            ["uv", "run", "python", str(BENCH_SCRIPT), "--dry-run"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"--dry-run failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    def test_smoke_succeeds(self) -> None:
        """--smoke should run minimal benchmark and exit 0."""
        result = subprocess.run(
            ["uv", "run", "python", str(BENCH_SCRIPT), "--smoke"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"--smoke failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    def test_help_shows_temperature_args(self) -> None:
        """Help text should document temperature options."""
        result = subprocess.run(
            ["uv", "run", "python", str(BENCH_SCRIPT), "--help"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--m-values" in result.stdout or "--temperature" in result.stdout, \
            "Help should document temperature sweep options"


class TestBenchAminxJaxTemperatures:
    """Verify bench_aminx_jax.py accepts --temperatures flag."""

    JAX_BENCH_SCRIPT = PROJECT_ROOT / "scripts" / "benchmarks" / "bench_aminx_jax.py"

    def test_temperatures_flag_dry_run(self) -> None:
        """--temperatures flag should work with --dry-run."""
        result = subprocess.run(
            [
                "uv", "run", "python", str(self.JAX_BENCH_SCRIPT),
                "--dry-run",
                "--temperatures", "0.1,1.0,2.0",
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, \
            f"--temperatures flag failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
