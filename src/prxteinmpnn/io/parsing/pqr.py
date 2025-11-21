"""PQR file parsing utilities.

prxteinmpnn.io.parsing.pqr
"""

import logging
import pathlib
from collections.abc import Sequence
from typing import IO, Any

import numpy as np
from biotite.structure import AtomArray

from prxteinmpnn.io.parsing.structures import ProcessedStructure
from prxteinmpnn.utils.residue_constants import van_der_waals_epsilon

logger = logging.getLogger(__name__)

n_index: np.ndarray
RECORD_NAME_MAX_LEN = 6


def _parse_atom_line(line: str) -> dict[str, Any] | None:
  """Parse a single ATOM/HETATM line from a PQR file."""
  fields = line.split()
  try:
    charge = float(fields[-2])
    radius = float(fields[-1])

    # Handle cases where serial number runs into record name
    if len(fields[0]) > RECORD_NAME_MAX_LEN:
      atom_name = fields[1]
      res_name = fields[2]
      chain = fields[3]
      res_seq = fields[4]
      x_idx, y_idx, z_idx = 5, 6, 7
    else:
      atom_name = fields[2]
      res_name = fields[3]
      chain = fields[4]
      res_seq = fields[5]
      x_idx, y_idx, z_idx = 6, 7, 8

    # Skip water molecules
    if res_name in ("HOH", "H2O", "WAT"):
      return None

    x = float(fields[x_idx])
    y = float(fields[y_idx])
    z = float(fields[z_idx])

    # Lookup epsilon
    element = atom_name[0]
    epsilon = van_der_waals_epsilon.get(element, 0.15)

    # Parse res_seq
    res_num_str = "".join(c for c in res_seq if c.isdigit() or c == "-")
    res_id = int(res_num_str) if res_num_str else -1

  except (IndexError, ValueError) as e:
    logger.warning("Failed to parse line: %s; error: %s", line.strip(), e)
    return None

  return {
    "coord": [x, y, z],
    "atom_name": atom_name,
    "res_name": res_name,
    "chain_id": chain,
    "res_id": res_id,
    "element": element,
    "charge": charge,
    "radius": radius,
    "epsilon": epsilon,
  }


def parse_pqr_to_processed_structure(
  pqr_file: IO[str] | str | pathlib.Path,
  chain_id: Sequence[str] | str | None = None,
) -> ProcessedStructure:
  """Parse a PQR file directly into a ProcessedStructure."""
  if isinstance(pqr_file, (str, pathlib.Path)):
    path = pathlib.Path(pqr_file)
    with path.open() as f:
      lines = f.readlines()
  else:
    lines = pqr_file.readlines()

  atom_lines = [line for line in lines if line.startswith(("ATOM", "HETATM"))]

  # Pre-allocate lists
  coords = []
  atom_names = []
  res_names = []
  chain_ids = []
  res_ids = []
  elements = []
  charges = []
  radii = []
  epsilons = []

  # Normalize chain_id to a set for filtering
  chain_id_set = (
    {chain_id} if isinstance(chain_id, str) else set(chain_id) if chain_id is not None else None
  )

  for line in atom_lines:
    parsed = _parse_atom_line(line)
    if parsed is None:
      continue

    if chain_id_set is not None and parsed["chain_id"] not in chain_id_set:
      continue

    coords.append(parsed["coord"])
    atom_names.append(parsed["atom_name"])
    res_names.append(parsed["res_name"])
    chain_ids.append(parsed["chain_id"])
    res_ids.append(parsed["res_id"])
    elements.append(parsed["element"])
    charges.append(parsed["charge"])
    radii.append(parsed["radius"])
    epsilons.append(parsed["epsilon"])

  num_atoms = len(coords)
  if num_atoms == 0:
    msg = "No atoms found in PQR file."
    raise ValueError(msg)

  # Create AtomArray
  atom_array = AtomArray(num_atoms)
  atom_array.coord = np.array(coords, dtype=np.float32)
  atom_array.atom_name = np.array(atom_names, dtype="U6")
  atom_array.res_name = np.array(res_names, dtype="U3")
  atom_array.chain_id = np.array(chain_ids, dtype="U3")
  atom_array.res_id = np.array(res_ids, dtype=int)
  atom_array.element = np.array(elements, dtype="U2")

  # Add charge annotation for consistency
  atom_array.set_annotation(
    "charge", np.array(charges, dtype=int),
  )

  return ProcessedStructure(
    atom_array=atom_array,
    r_indices=atom_array.res_id,
    chain_ids=np.zeros(num_atoms, dtype=np.int32),  # Placeholder
    charges=np.array(charges, dtype=np.float32),
    radii=np.array(radii, dtype=np.float32),
    epsilons=np.array(epsilons, dtype=np.float32),
  )
