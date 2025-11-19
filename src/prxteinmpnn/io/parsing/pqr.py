"""PQR file parsing utilities.

prxteinmpnn.io.parsing.pqr
"""

import logging
import pathlib
import tempfile
from collections.abc import Sequence
from typing import IO

import numpy as np
from biotite.structure import AtomArray

from prxteinmpnn.utils.data_structures import (
  EstatInfo,
)
from prxteinmpnn.io.parsing.structures import ProcessedStructure
from prxteinmpnn.utils.residue_constants import van_der_waals_epsilon

logger = logging.getLogger(__name__)

n_index: np.ndarray


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
    fields = line.split()
    try:
      charge = float(fields[-2])
      radius = float(fields[-1])

      # Handle cases where serial number runs into record name
      if len(fields[0]) > 6:
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
        continue

      # Filter by chain_id
      if chain_id_set is not None and chain not in chain_id_set:
        continue

      x = float(fields[x_idx])
      y = float(fields[y_idx])
      z = float(fields[z_idx])
      
      # Lookup epsilon
      element = atom_name[0]
      epsilon = van_der_waals_epsilon.get(element, 0.15)
      
      coords.append([x, y, z])
      atom_names.append(atom_name)
      res_names.append(res_name)
      chain_ids.append(chain)
      
      # Parse res_seq (remove insertion code for now or keep it?)
      # Biotite res_id is integer.
      res_num_str = "".join(c for c in res_seq if c.isdigit() or c == "-")
      res_ids.append(int(res_num_str) if res_num_str else -1)
      
      elements.append(element)
      charges.append(charge)
      radii.append(radius)
      epsilons.append(epsilon)

    except (IndexError, ValueError) as e:
      logger.warning("Failed to parse line: %s; error: %s", line.strip(), e)
      continue

  num_atoms = len(coords)
  if num_atoms == 0:
      raise ValueError("No atoms found in PQR file.")

  # Create AtomArray
  atom_array = AtomArray(num_atoms)
  atom_array.coord = np.array(coords, dtype=np.float32)
  atom_array.atom_name = np.array(atom_names, dtype="U6")
  atom_array.res_name = np.array(res_names, dtype="U3")
  atom_array.chain_id = np.array(chain_ids, dtype="U3")
  atom_array.res_id = np.array(res_ids, dtype=int)
  atom_array.element = np.array(elements, dtype="U2")
  
  # Add charge annotation for consistency (though we store it in ProcessedStructure)
  atom_array.set_annotation("charge", np.array(charges, dtype=int)) # Biotite expects int? No, usually float but PDB is weird.
  # Actually Biotite's charge annotation is usually integer formal charge.
  # Partial charges are not standard in AtomArray unless we add a custom annotation.
  # But we return charges separately in ProcessedStructure.
  
  return ProcessedStructure(
      atom_array=atom_array,
      r_indices=atom_array.res_id,
      chain_ids=np.zeros(num_atoms, dtype=np.int32), # Placeholder
      charges=np.array(charges, dtype=np.float32),
      radii=np.array(radii, dtype=np.float32),
      epsilons=np.array(epsilons, dtype=np.float32),
  )
