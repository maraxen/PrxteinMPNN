#!/usr/bin/env python3
"""Regenerate bundled ``*.eqx.zst`` from a Dauparas-format ``.pt`` checkpoint.

Used when packaged Equinox blobs were exported with mismatched topology
(e.g. wrong ``num_context_layers`` on LigandMPNN).

Pipeline: ``convert_weights.py`` (uncompressed ``.eqx``) → zstd level 19 (bundle format).

Examples::

  uv run python scripts/regenerate_packaged_eqx_zst.py \\
    --input ~/reference_ligandmpnn_clone/model_weights/ligandmpnn_v_32_020_25.pt

After replacing ``src/prxteinmpnn/model_params/<stem>.eqx.zst``, run::

  uv run pytest tests/model/test_ligandmpnn_equivalence.py -q
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import zstandard as zstd


def _project_root() -> Path:
  return Path(__file__).resolve().parents[1]


def _run_convert(*, convert_script: Path, pt_path: Path, eqx_out: Path, side_chain: str) -> None:
  cmd = [
    sys.executable,
    str(convert_script),
    "--input",
    str(pt_path),
    "--output",
    str(eqx_out),
    "--ligand-side-chain-context",
    side_chain,
  ]
  subprocess.run(cmd, check=True)


def _compress_eqx(*, eqx_path: Path, zst_path: Path, level: int) -> None:
  raw = eqx_path.read_bytes()
  compressor = zstd.ZstdCompressor(level=level)
  zst_path.write_bytes(compressor.compress(raw))


def main() -> int:
  parser = argparse.ArgumentParser(description="Rebuild packaged .eqx.zst from a PyTorch checkpoint.")
  parser.add_argument("--input", type=Path, required=True, help="Source .pt checkpoint")
  parser.add_argument(
    "--out-dir",
    type=Path,
    default=None,
    help="Directory for output <stem>.eqx.zst (default: src/prxteinmpnn/model_params)",
  )
  parser.add_argument(
    "--zstd-level",
    type=int,
    default=19,
    help="Zstandard compression level (default: 19, matches bundled assets).",
  )
  parser.add_argument(
    "--ligand-side-chain-context",
    type=str,
    choices=["auto", "on", "off"],
    default="auto",
    help="Forwarded to scripts/convert_weights.py",
  )
  parser.add_argument("--verify", action="store_true", help="Try load_model(local_path=out) after writing.")
  args = parser.parse_args()

  root = _project_root()
  out_dir = args.out_dir if args.out_dir is not None else root / "src" / "prxteinmpnn" / "model_params"
  out_dir.mkdir(parents=True, exist_ok=True)

  stem = args.input.stem
  zst_path = out_dir / f"{stem}.eqx.zst"
  convert_script = root / "scripts" / "convert_weights.py"
  if not convert_script.is_file():
    print(f"Missing {convert_script}", file=sys.stderr)
    return 1

  with tempfile.TemporaryDirectory(prefix="eqx_rebuild_") as tmp:
    tmp_eqx = Path(tmp) / f"{stem}.eqx"
    print(f"Converting {args.input} -> {tmp_eqx}")
    _run_convert(
      convert_script=convert_script,
      pt_path=args.input.resolve(),
      eqx_out=tmp_eqx,
      side_chain=args.ligand_side_chain_context,
    )
    print(f"Compressing (zstd {args.zstd_level}) -> {zst_path}")
    _compress_eqx(eqx_path=tmp_eqx, zst_path=zst_path, level=args.zstd_level)

  if args.verify:
    import jax

    from prxteinmpnn.io.weights import load_model

    key = jax.random.PRNGKey(0)
    out_abs = zst_path.resolve()
    load_model(checkpoint_id=stem, local_path=str(out_abs), key=key)
    print("verify: load_model(local_path=...) OK")

  print(f"Wrote {zst_path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
