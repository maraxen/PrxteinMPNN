"""Command-line interface for Potts model operations."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Annotated

import equinox as eqx
import jax.numpy as jnp
import typer
import zstandard as zstd

from aminx.potts.spec import PottsRunSpec
from aminx.types.bundles import GeometryBundle

potts_app = typer.Typer(
  name="potts",
  help="Potts model operations and specification management.",
  no_args_is_help=True,
)

_OPT = typer.Option


def _geometry_skeleton() -> GeometryBundle:
  """Create a skeleton GeometryBundle for deserialization.

  The skeleton structure is used by eqx.tree_deserialise_leaves to restore
  a serialized GeometryBundle with the correct shapes and dtypes.
  """
  return GeometryBundle(
    coords=jnp.zeros((1, 1, 4, 3), dtype=jnp.float32),
    mask=jnp.zeros((1, 1), dtype=jnp.float32),
    residue_index=jnp.zeros((1, 1), dtype=jnp.int32),
    chain_index=jnp.zeros((1, 1), dtype=jnp.int32),
    n_states=1,
    n_canonical=20,
    n_flat=1,
    structure_mapping=None,
    physics_features=None,
  )


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
    _OPT(
      help=(
        "Number of nearest neighbours used during training (required). "
        "Must match the checkpoint exactly — value is logged during recapture "
        "(search pottsmpnn_to_eqx.py logs for 'k_neighbors=NN'). "
        "Wrong value produces incorrect graph topology. "
        "Cannot be recovered from checkpoint file."
      ),
    ),
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

  IMPORTANT: k_neighbors is a static hyperparameter not stored in the checkpoint.
  The user must supply the correct value from pottsmpnn_to_eqx.py recapture logs.
  Using the wrong value produces silent misconfiguration (incorrect graph topology).

  Args:
      weights_path: Path to Potts model checkpoint (must exist).
      k_neighbors: Graph connectivity from training (non-recoverable from checkpoint).
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


@potts_app.command("run")
def potts_run(
  spec_path: Annotated[
    Path,
    _OPT(help="Path to PottsRunSpec JSON file (required)"),
  ],
  geometry_path: Annotated[
    Path,
    _OPT(help="Path to GeometryBundle (eqx or eqx.zst format) (required)"),
  ],
  out: Annotated[
    Path,
    _OPT(help="Write PottsResult JSON to this file (required)"),
  ],
) -> None:
  """Run Potts inference from a specification and geometry.

  Loads a PottsRunSpec from JSON, loads geometry from equinox serialized file,
  executes run_potts orchestration, and writes the result to JSON.

  Args:
      spec_path: Path to PottsRunSpec JSON specification.
      geometry_path: Path to GeometryBundle file (.eqx or .eqx.zst format).
      out: Output file path for PottsResult JSON.

  Raises:
      typer.Exit: If spec or geometry cannot be loaded or run fails.
  """
  import jax  # noqa: PLC0415

  from aminx.potts.runner import run_potts  # noqa: PLC0415

  try:
    # Load specification
    spec_json = spec_path.read_text(encoding="utf-8")
    spec = PottsRunSpec.from_json(spec_json)

    # Load geometry from equinox serialized file (eqx or eqx.zst format)
    geometry_path_obj = Path(geometry_path)
    skeleton = _geometry_skeleton()
    with geometry_path_obj.open("rb") as f:
      header = f.read(4)
      f.seek(0)
      if header == b"\x28\xb5\x2f\xfd":  # zstd magic number
        # Decompress zstd and deserialize
        dctx = zstd.ZstdDecompressor()
        stream = io.BytesIO(dctx.decompress(f.read()))  # ty: ignore[unresolved-attribute]
        geometry = eqx.tree_deserialise_leaves(stream, skeleton)
      else:
        # Direct eqx deserialization (uncompressed)
        geometry = eqx.tree_deserialise_leaves(geometry_path_obj, skeleton)

    # Run Potts inference
    key = jax.random.PRNGKey(0)
    result = run_potts(spec, geometry, key)

    # Serialize result to JSON
    result_dict = {
      "marginals": result.marginals.tolist(),
      "h": result.h.tolist(),
      "J": result.J.tolist(),
      "rho": result.rho.tolist(),
      "calibrated_marginals": result.calibrated_marginals.tolist(),
      "n_backbones": result.n_backbones,
    }
    result_json = json.dumps(result_dict, indent=2)
    out.write_text(result_json, encoding="utf-8")

  except (ValueError, TypeError, FileNotFoundError, zstd.ZstdError) as exc:
    typer.echo(f"Error running Potts inference: {exc}", err=True)
    raise typer.Exit(code=1) from exc
