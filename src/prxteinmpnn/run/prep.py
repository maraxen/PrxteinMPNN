"""Core user interface for the PrxteinMPNN package."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from io import StringIO

  from grain.python import IterDataset

  from prxteinmpnn.run.specs import Specs
  from prxteinmpnn.utils.types import Model

from prxteinmpnn.io import loaders
from prxteinmpnn.io.weights import load_model


def _loader_inputs(inputs: Sequence[str | StringIO] | str | StringIO) -> Sequence[str | StringIO]:
  return (inputs,) if not isinstance(inputs, Sequence) else inputs  # type: ignore[invalid-return-type]


def prep_protein_stream_and_model(
  spec: Specs,
  *,
  use_new_architecture: bool = True,
) -> tuple[IterDataset, Model]:
  """Prepare the protein data stream and model parameters.

  Args:
      spec: A RunSpecification object containing configuration options.
      use_new_architecture: If True (default), return a PrxteinMPNN Equinox module.
                            If False, return legacy PyTree parameters.

  Returns:
      A tuple containing the protein data iterator and model (PyTree or PrxteinMPNN).

  """
  parse_kwargs = {
    "chain_id": spec.chain_id,
    "model": spec.model,
    "altloc": spec.altloc,
    "topology": spec.topology,
  }
  protein_iterator = loaders.create_protein_dataset(
    _loader_inputs(spec.inputs),
    batch_size=spec.batch_size,
    foldcomp_database=spec.foldcomp_database,
    parse_kwargs=parse_kwargs,
    pass_mode=spec.pass_mode,
    use_preprocessed=spec.use_preprocessed,
    preprocessed_index_path=spec.preprocessed_index_path,
    split=spec.split,
    use_electrostatics=spec.use_electrostatics,
    estat_noise=spec.estat_noise,
    estat_noise_mode=spec.estat_noise_mode,
    use_vdw=spec.use_vdw,
    vdw_noise=spec.vdw_noise,
    vdw_noise_mode=spec.vdw_noise_mode,
  )
  # use_new_architecture parameter is deprecated; always use Equinox model now
  del use_new_architecture
  model = load_model(
    model_version=spec.model_version,
    model_weights=spec.model_weights,
  )
  return protein_iterator, model
