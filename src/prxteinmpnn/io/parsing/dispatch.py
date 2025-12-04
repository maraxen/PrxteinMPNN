"""Utilities for processing structure and trajectory files.

prxteinmpnn.io.parsing

This module has been refactored to contain only synchronous parsing logic,
making it suitable for use in parallel worker processes managed by Grain.
All async operations and direct I/O handling have been moved to the
`sources.py` and `operations.py` modules.
"""

import logging
import pathlib
import tempfile
import warnings
from collections.abc import Sequence
from io import StringIO
from typing import IO, Any

import h5py
import mdtraj as md

from prxteinmpnn.utils.data_structures import ProteinStream

from priox.io.parsing.biotite import _parse_biotite
from priox.io.streaming.mdcath import parse_mdcath_to_processed_structure
from priox.io.parsing.mdtraj import parse_mdtraj_to_processed_structure
from priox.io.parsing.pqr import parse_pqr_to_processed_structure
from priox.io.parsing.utils import processed_structure_to_protein_tuples

logger = logging.getLogger(__name__)


def _determine_h5_structure(source: str | StringIO | pathlib.Path) -> str:
  """Determine the structure of an HDF5 file."""
  logger.debug("Attempting to determine HDF5 file structure for source: %s", source)
  try:
    with h5py.File(source, "r") as f:
      keys = list(f.keys())
      if "coordinates" in f:
        logger.info("HDF5 structure determined as 'mdtraj' (found 'coordinates' key).")
        return "mdtraj"
      if len(keys) == 1:
        logger.info(
          "HDF5 structure determined as 'mdcath' (found single top-level key: %s).",
          keys[0],
        )
        return "mdcath"
      msg = f"Could not determine HDF5 structure: Unrecognized format. Top-level keys: {keys}"
      logger.warning(msg)
      warnings.warn(msg, stacklevel=2)  # Retain original warning behavior
      return "unknown"
  except Exception as e:
    msg = f"Could not determine HDF5 structure: {type(e).__name__}: {e}"
    logger.exception(msg)
    warnings.warn(msg, stacklevel=2)  # Retain original warning behavior
    return "unknown"


def parse_input(  # noqa: C901, PLR0912, PLR0915
  source: str | IO[str] | pathlib.Path,
  *,
  model: int | None = None,
  altloc: str | None = None,
  chain_id: Sequence[str] | str | None = None,
  topology: str | pathlib.Path | None = None,
  extract_dihedrals: bool = False,
  **kwargs: Any,  # noqa: ANN401
) -> ProteinStream:
  """Parse a structure file or string into a generator of ProteinTuples.

  This is a synchronous, CPU-bound function intended to be run in a worker process.
  It dispatches to the correct low-level parser based on file type.

  Args:
      source: A file path (str or Path) or a file-like object (StringIO).
      model: The model number to load. If None, all models are loaded.
      altloc: The alternate location identifier to use.
      chain_id: Specific chain(s) to parse from the structure.
      topology: Optional topology file path for formats requiring separate topology.
      extract_dihedrals: Whether to compute and include backbone dihedral angles.
      **kwargs: Additional keyword arguments to pass to the structure loader.

  Yields:
      A tuple containing a `ProteinTuple` and the source identifier string.

  """
  logger.info("Starting input parsing for source: %s", source)
  temp_path = None
  tmp_top_path = None

  if not isinstance(source, (pathlib.Path | str)):
    logger.debug("Source is IO. Creating temporary PDB file.")
    with tempfile.NamedTemporaryFile(
      mode="w",
      delete=False,
      suffix=".pdb",
    ) as tmp:
      tmp.write(source.read())
      temp_path = pathlib.Path(tmp.name)
      logger.info("Content written to temporary file: %s", temp_path)
      source = temp_path

  try:
    if isinstance(source, (str | pathlib.Path)):
      path = pathlib.Path(source)
      if path.suffix.lower() == ".pqr":
        logger.info("Dispatching to PQR parser.")
        processed = parse_pqr_to_processed_structure(path, chain_id)
        yield from processed_structure_to_protein_tuples(
          processed,
          str(path),
          extract_dihedrals=extract_dihedrals,
        )
        return
      if path.suffix.lower() in {".h5", ".hdf5"}:
        h5_structure = _determine_h5_structure(path)
        add_hydrogens = kwargs.get("add_hydrogens", True)

        if h5_structure == "mdcath":
          logger.info("Dispatching to mdCATH HDF5 parser.")
          for processed in parse_mdcath_to_processed_structure(
            path,
            chain_id,
            add_hydrogens=add_hydrogens,
          ):
            yield from processed_structure_to_protein_tuples(
              processed,
              str(path),
              extract_dihedrals=extract_dihedrals,
            )
        elif h5_structure == "mdtraj":
          logger.info("Dispatching to MDTraj HDF5 parser.")
          for processed in parse_mdtraj_to_processed_structure(
            path,
            chain_id,
            topology=topology,
            add_hydrogens=add_hydrogens,
          ):
            yield from processed_structure_to_protein_tuples(
              processed,
              str(path),
              extract_dihedrals=extract_dihedrals,
            )
        else:
          logger.warning("Unknown HDF5 structure, returning early.")
        return

    logger.info(
      "Dispatching to general Biotite parser for file type: %s",
      pathlib.Path(source).suffix,
    )
    topology = pathlib.Path(topology) if topology is not None else None
    if topology is not None and topology.suffix.lower() not in {".pdb", ".cif"}:
      logger.info(
        "Topology file %s has unsupported extension for Biotite. Using MDTraj to load topology.",
        topology,
      )
      try:
        traj_holder = md.load_frame(source, 0, top=topology)
        logger.info("Successfully loaded mdtraj with topology. Now attempting to convert.")
        with tempfile.NamedTemporaryFile(
          mode="w",
          delete=False,
          suffix=".pdb",
        ) as tmp_top:
          traj_holder.save_pdb(tmp_top.name)
          tmp_top_path = pathlib.Path(tmp_top.name)
          logger.info("Converted topology saved to temporary file: %s", tmp_top_path)
          topology = tmp_top_path
      except Exception as e:
        logger.exception("Exception encountered in topology processing...", exc_info=e)

    yield from _parse_biotite(
      source,
      model,
      altloc=altloc,
      chain_id=chain_id,
      extract_dihedrals=extract_dihedrals,
      topology=topology,
      **kwargs,
    )

  finally:
    if temp_path is not None:
      try:
        temp_path.unlink()
        logger.debug("Cleaned up temporary file: %s", temp_path)
      except OSError as e:
        logger.warning("Could not delete temporary file %s: %s", temp_path, e)
    if tmp_top_path is not None:
      try:
        tmp_top_path.unlink()
        logger.debug("Cleaned up temporary topology file: %s", tmp_top_path)
      except OSError as e:
        logger.warning("Could not delete temporary topology file %s: %s", tmp_top_path, e)
