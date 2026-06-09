"""Command-line interface for Potts model operations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from aminx.potts.spec import PottsRunSpec

potts_app = typer.Typer(
  name="potts",
  help="Potts model operations and specification management.",
  no_args_is_help=True,
)

_OPT = typer.Option


def _emit_spec_json(spec: PottsRunSpec, compact: bool, out: Path | None) -> None:  # noqa: FBT001
  """Serialize PottsRunSpec to JSON and write to out or stdout."""
  # PottsRunSpec has .to_json() which returns a JSON string
  blob = spec.to_json()
  if not compact:
    # Pretty-print if not compact
    data = json.loads(blob)
    blob = json.dumps(data, indent=2, sort_keys=True)
  if out is not None:
    out.write_text(blob, encoding="utf-8")
  else:
    typer.echo(blob)


@potts_app.command("emit")
def potts_emit(
  weights_path: Annotated[
    str,
    _OPT(help="Path to Potts model weights checkpoint (required)"),
  ],
  k_neighbors: Annotated[
    int,
    _OPT(help="Graph connectivity (required)"),
  ],
  out: Annotated[
    Path,
    _OPT(help="Write JSON specification to this file (required)"),
  ],
  n_backbones: Annotated[
    int,
    _OPT(help="Number of backbones in ensemble"),
  ] = 1,
  caliby_path: Annotated[
    str | None,
    _OPT(help="Optional path to learned calibration model"),
  ] = None,
  trw_backend: Annotated[  # noqa: ARG001
    str,
    _OPT(help="TRW rho backend (reserved for future use)"),
  ] = "dense_pinv",
  trw_iters: Annotated[  # noqa: ARG001
    int,
    _OPT(help="TRW iterations (reserved for future use)"),
  ] = 10,
) -> None:
  """Emit PottsRunSpec as JSON specification.

  Creates a specification file for Potts model inference with configurable
  TRW numerics backend and optional post-hoc calibration.

  Validates that weights_path exists and constructs a PottsRunSpec with
  default TRW configuration. The output JSON can be round-tripped via
  PottsRunSpec.from_json().

  Args:
      weights_path: Path to Potts model checkpoint (must exist).
      k_neighbors: Graph connectivity from checkpoint metadata.
      out: Output JSON file path (required).
      n_backbones: Number of backbones (default 1, must be >= 1).
      caliby_path: Optional calibration model path (None is valid identity default).
      trw_backend: TRW rho backend name (default "dense_pinv", reserved for future use).
      trw_iters: TRW iterations (default 10, reserved for future use).

  Raises:
      typer.Exit: If weights_path does not exist or validation fails.
  """
  # Validate weights_path exists
  weights_file = Path(weights_path)
  if not weights_file.exists():
    typer.echo(f"Error: weights_path '{weights_path}' does not exist", err=True)
    raise typer.Exit(code=1)

  # Construct spec
  try:
    spec = PottsRunSpec(
      n_backbones=n_backbones,
      weights_path=weights_path,
      caliby_path=caliby_path,
      trw_spec=None,  # Will use default in __post_init__
      k_neighbors=k_neighbors,
      training=False,
    )
  except (ValueError, TypeError) as exc:
    typer.echo(f"Invalid specification: {exc}", err=True)
    raise typer.Exit(code=1) from exc

  # Emit JSON
  _emit_spec_json(spec, compact=False, out=out)
