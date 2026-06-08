"""Generate deterministic golden fixtures for fast parity tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _build_linear_fixture(seed: int) -> dict[str, np.ndarray]:
  rng = np.random.default_rng(seed)
  in_size = 16
  out_size = 8
  batch_size = 6
  weight = rng.standard_normal((out_size, in_size), dtype=np.float32)
  bias = rng.standard_normal((out_size,), dtype=np.float32)
  inputs = rng.standard_normal((batch_size, in_size), dtype=np.float32)
  expected = inputs @ weight.T + bias
  return {
    "inputs": inputs.astype(np.float32),
    "weight": weight.astype(np.float32),
    "bias": bias.astype(np.float32),
    "expected": expected.astype(np.float32),
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("tests/golden/parity"),
    help="Directory where parity golden fixtures are written.",
  )
  parser.add_argument(
    "--seed",
    type=int,
    default=20260409,
    help="Deterministic seed for fixture generation.",
  )
  args = parser.parse_args()

  args.output_dir.mkdir(parents=True, exist_ok=True)
  fixture = _build_linear_fixture(seed=args.seed)
  np.savez(args.output_dir / "linear_layer_fixture.npz", **fixture)

  metadata = {
    "fixture_name": "linear_layer_fixture",
    "seed": args.seed,
    "dtype": "float32",
    "tolerances": {"rtol": 1e-6, "atol": 1e-6},
    "reference_source": "LigandMPNN parity fast check",
    "reference_commit": "3870631",
    "notes": "Fixture validates deterministic linear layer parity contract used by JAX/PyTorch conversions.",
  }
  (args.output_dir / "metadata.json").write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )


if __name__ == "__main__":
  main()
