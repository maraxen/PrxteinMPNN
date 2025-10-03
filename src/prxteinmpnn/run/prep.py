"""Core user interface for the PrxteinMPNN package."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from io import StringIO

  from grain.python import IterDataset

  from prxteinmpnn.run.specs import Specs
  from prxteinmpnn.utils.types import (
    ModelParameters,
  )


from prxteinmpnn.io import loaders
from prxteinmpnn.mpnn import get_mpnn_model


def _loader_inputs(inputs: Sequence[str | StringIO] | str | StringIO) -> Sequence[str | StringIO]:
  return (inputs,) if not isinstance(inputs, Sequence) else inputs


def prep_protein_stream_and_model(spec: Specs) -> tuple[IterDataset, ModelParameters]:
  """Prepare the protein data stream and model parameters.

  Args:
      spec: A RunSpecification object containing configuration options.

  Returns:
      A tuple containing the protein data iterator and model parameters.

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
    cache_path=spec.cache_path,
  )
  model_parameters = get_mpnn_model(
    model_version=spec.model_version,
    model_weights=spec.model_weights,
  )
  return protein_iterator, model_parameters
