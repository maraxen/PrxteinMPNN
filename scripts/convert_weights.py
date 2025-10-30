"""Convert ProteinMPNN weights from functional format to Equinox format.

This script loads model parameters from the original .pkl files and converts them
to Equinox module format, saving them as .eqx files for easier loading.

Usage:
  uv run python scripts/convert_weights.py

Example:
  >>> uv run python scripts/convert_weights.py
  Converting v_48_020.pkl (original weights)...
  Saved to: src/prxteinmpnn/model/original/v_48_020.eqx
  Done!

"""

from __future__ import annotations

import argparse
import pathlib
from typing import Literal

import equinox as eqx
import jax

from prxteinmpnn.conversion import create_prxteinmpnn
from prxteinmpnn.functional import get_functional_model

ModelVersion = Literal[
  "v_48_002.pkl",
  "v_48_010.pkl",
  "v_48_020.pkl",
  "v_48_030.pkl",
]

ModelWeights = Literal["original", "soluble"]


def convert_model(
  model_version: ModelVersion = "v_48_020.pkl",
  model_weights: ModelWeights = "original",
  num_encoder_layers: int = 3,
  num_decoder_layers: int = 3,
  output_dir: pathlib.Path | None = None,
) -> pathlib.Path:
  """Convert a functional model to Equinox format.

  Args:
    model_version: Version of the model to convert.
    model_weights: Type of weights (original or soluble).
    num_encoder_layers: Number of encoder layers.
    num_decoder_layers: Number of decoder layers.
    output_dir: Directory to save the converted model. If None, uses default location.

  Returns:
    Path to the saved .eqx file.

  Example:
    >>> output_path = convert_model("v_48_020.pkl", "original")
    >>> print(f"Saved to: {output_path}")

  """
  print(f"Converting {model_version} ({model_weights} weights)...")  # noqa: T201

  # Load functional model parameters
  model_params = get_functional_model(
    model_version=model_version,
    model_weights=model_weights,
  )

  # Create Equinox model
  key = jax.random.PRNGKey(0)
  eqx_model = create_prxteinmpnn(
    model_params,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    key=key,
  )

  # Determine output path
  if output_dir is None:
    base_dir = pathlib.Path(__file__).parent.parent / "src" / "prxteinmpnn"
    output_dir = base_dir / "model" / model_weights

  output_dir.mkdir(parents=True, exist_ok=True)
  output_path = output_dir / model_version.replace(".pkl", ".eqx")

  # Save the model
  eqx.tree_serialise_leaves(output_path, eqx_model)

  print(f"Saved to: {output_path}")  # noqa: T201
  return output_path


def main() -> None:
  """Run the conversion script to convert ProteinMPNN weights to Equinox format."""
  parser = argparse.ArgumentParser(
    description="Convert ProteinMPNN weights to Equinox format",
  )
  parser.add_argument(
    "--version",
    type=str,
    default="v_48_020.pkl",
    choices=["v_48_002.pkl", "v_48_010.pkl", "v_48_020.pkl", "v_48_030.pkl"],
    help="Model version to convert",
  )
  parser.add_argument(
    "--weights",
    type=str,
    default="original",
    choices=["original", "soluble"],
    help="Weight type to convert",
  )
  parser.add_argument(
    "--encoder-layers",
    type=int,
    default=3,
    help="Number of encoder layers",
  )
  parser.add_argument(
    "--decoder-layers",
    type=int,
    default=3,
    help="Number of decoder layers",
  )
  parser.add_argument(
    "--output-dir",
    type=str,
    default=None,
    help="Output directory for converted models",
  )
  parser.add_argument(
    "--all",
    action="store_true",
    help="Convert all model versions",
  )

  args = parser.parse_args()

  output_dir = pathlib.Path(args.output_dir) if args.output_dir else None

  if args.all:
    versions = ["v_48_002.pkl", "v_48_010.pkl", "v_48_020.pkl", "v_48_030.pkl"]
    for version in versions:
      convert_model(
        model_version=version,  # type: ignore[arg-type]
        model_weights=args.weights,  # type: ignore[arg-type]
        num_encoder_layers=args.encoder_layers,
        num_decoder_layers=args.decoder_layers,
        output_dir=output_dir,
      )
  else:
    convert_model(
      model_version=args.version,  # type: ignore[arg-type]
      model_weights=args.weights,  # type: ignore[arg-type]
      num_encoder_layers=args.encoder_layers,
      num_decoder_layers=args.decoder_layers,
      output_dir=output_dir,
    )

  print("Done!")  # noqa: T201


if __name__ == "__main__":
  main()
