"""PQR file parsing utilities.

prxteinmpnn.io.parsing.pqr
"""

import logging
import pathlib
import tempfile
from collections.abc import Sequence
from typing import IO

import numpy as np

from prxteinmpnn.utils.data_structures import EstatInfo

logger = logging.getLogger(__name__)

n_index: np.ndarray


def _parse_pqr(
  pqr_file: IO[str] | str | pathlib.Path,
  chain_id: Sequence[str] | str | None = None,
) -> tuple[
  pathlib.Path,
  EstatInfo,
]:
  """Parse a PQR file to extract atom array, electrostatics data, and masks.

  Args:
      pqr_file: The path to the PQR file or a file-like object.
      chain_id: The specific chain(s) to parse from the structure.

  Returns:
      A tuple containing:
        - temp_path: Path to a temporary PDB file with atom records.
        - (charges, radii): Tuple of numpy arrays for charges and radii.
        - estat_backbone_mask: Boolean numpy array, True for backbone atoms.
        - estat_resid: Integer numpy array of residue numbers.
        - estat_chain_id: Integer numpy array of chain IDs (ord value).

  """
  if isinstance(pqr_file, (str, pathlib.Path)):
    path = pathlib.Path(pqr_file)
    with path.open() as f:
      lines = f.readlines()
  else:
    lines = pqr_file.readlines()

  atom_lines = [line for line in lines if line.startswith(("ATOM", "HETATM"))]
  charge_array, radius_array, estat_backbone_mask, estat_resid, estat_chain_id = [], [], [], [], []
  backbone_names = {"N", "CA", "C", "O"}

  # Normalize chain_id to a set for filtering
  chain_id_set = (
    {chain_id} if isinstance(chain_id, str) else set(chain_id) if chain_id is not None else None
  )

  pdb_lines = []
  for line in atom_lines:
    fields = line.split()
    try:
      charge = float(fields[-2])
      radius = float(fields[-1])
      atom_name = fields[2]
      res_name = fields[3]
      chain = fields[4]
      res_seq = fields[5]
      x = float(fields[6])
      y = float(fields[7])
      z = float(fields[8])
      # Optional: atom serial number
      serial = int(fields[1]) if fields[1].isdigit() else 0
      occupancy = 1.00
      bfactor = 0.00
    except (IndexError, ValueError) as e:
      logger.warning("Failed to parse charge/radius from line: %s; error: %s", line.strip(), e)
      continue

    # Filter by chain_id if specified
    if chain_id_set is not None and chain not in chain_id_set:
      continue

    charge_array.append(charge)
    radius_array.append(radius)
    estat_backbone_mask.append(atom_name in backbone_names)
    try:
      estat_resid.append(int(res_seq))
    except ValueError:
      estat_resid.append(-1)
    estat_chain_id.append(ord(chain) if chain else -1)

    # Compose a PDB-formatted line (columns aligned)
    pdb_line = (
      f"{fields[0]:<6}{serial:>5} {atom_name:^4}{' '}{res_name:>3} {chain:>1}{int(res_seq):>4}    "
      f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{bfactor:6.2f}          \n"
    )
    pdb_lines.append(pdb_line)

  with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".pdb") as tmp:
    tmp.writelines(pdb_lines)
    temp_path = pathlib.Path(tmp.name)

  return (
    temp_path,
    EstatInfo(
      np.array(charge_array, dtype=np.float32),
      np.array(radius_array, dtype=np.float32),
      np.array(estat_backbone_mask, dtype=bool),
      np.array(estat_resid, dtype=np.int32),
      np.array(estat_chain_id, dtype=np.int32),
    ),
  )
