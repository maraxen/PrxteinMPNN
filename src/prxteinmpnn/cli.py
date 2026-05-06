"""Command-line interface (Typer) for PrxteinMPNN."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, cast

import typer

from prxteinmpnn.run.run_spec_portable_json import (
  run_spec_portable_from_dict,
  run_spec_portable_to_dict,
)
from prxteinmpnn.run.spec_json import run_specification_from_json, run_specification_to_json

app = typer.Typer(
  name="prxteinmpnn",
  help="PrxteinMPNN CLI — configuration and run helpers.",
  no_args_is_help=True,
)

spec_app = typer.Typer(
  name="spec",
  help="Run specification JSON (see prxteinmpnn.run.spec_json).",
  no_args_is_help=True,
)
app.add_typer(spec_app, name="spec")


@spec_app.command("validate")
def spec_validate(
  path: Annotated[Path, typer.Argument(exists=True, readable=True, help="JSON file from run_specification_to_json")],
) -> None:
  """Load a JSON specification and exit 0 if it constructs a valid spec."""
  text = path.read_text(encoding="utf-8")
  loaded = run_specification_from_json(text)
  typer.secho(f"OK: {type(loaded).__name__}", fg=typer.colors.GREEN)


@spec_app.command("roundtrip")
def spec_roundtrip(
  path: Annotated[Path, typer.Argument(exists=True, readable=True, help="JSON spec file")],
  out: Annotated[Path | None, typer.Option(help="Write re-serialized JSON here")] = None,
  *,
  compact: Annotated[bool, typer.Option(help="Single-line JSON")] = False,
) -> None:
  """Load JSON, re-encode through the same codec (checks round-trip)."""
  spec = run_specification_from_json(path.read_text(encoding="utf-8"))
  blob = run_specification_to_json(spec, indent=None if compact else 2)
  if out is not None:
    out.write_text(blob, encoding="utf-8")
    typer.secho(f"Wrote {out}", fg=typer.colors.GREEN)
  else:
    typer.echo(blob)


@spec_app.command("portable-roundtrip")
def spec_portable_roundtrip(
  path: Annotated[
    Path,
    typer.Argument(exists=True, readable=True, help="JSON dict: portable RunSpec subset"),
  ],
  *,
  compact: Annotated[bool, typer.Option(help="Single-line JSON")] = False,
) -> None:
  """Load portable subset JSON, rebuild :class:`~prxteinmpnn.run.spec.RunSpec`, re-encode subset."""
  raw = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(raw, dict):
    typer.secho("Top-level JSON must be an object", fg=typer.colors.RED)
    raise typer.Exit(code=1)
  rs = run_spec_portable_from_dict(cast("dict[str, Any]", raw))
  out = run_spec_portable_to_dict(rs)
  typer.echo(json.dumps(out, indent=None if compact else 2))


def main() -> None:
  """Console script entrypoint."""
  app()


if __name__ == "__main__":
  main()
