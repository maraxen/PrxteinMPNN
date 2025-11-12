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


def _parse_pqr(  # noqa: PLR0915
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

      # Handle cases where serial number runs into record name (e.g., "HETATM14004")
      # If fields[0] is longer than 6 chars, the serial is concatenated
      if len(fields[0]) > 6:  # noqa: PLR2004
        # Serial is smashed with record name, shift indices
        atom_name = fields[1]
        res_name = fields[2]
        chain = fields[3]
        res_seq = fields[4]
        x_idx, y_idx, z_idx = 5, 6, 7
        serial = 0  # Can't reliably extract, use 0
      else:
        # Normal case with separate fields
        atom_name = fields[2]
        res_name = fields[3]
        chain = fields[4]
        res_seq = fields[5]
        x_idx, y_idx, z_idx = 6, 7, 8
        serial = int(fields[1]) if fields[1].isdigit() else 0

      # Skip water molecules
      if res_name in ("HOH", "H2O", "WAT"):
        continue

      x = float(fields[x_idx])
      y = float(fields[y_idx])
      z = float(fields[z_idx])
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

    # Parse residue sequence number, handling insertion codes (e.g., '52A')
    # Separate numeric part from insertion code
    res_num_str = "".join(c for c in res_seq if c.isdigit() or c == "-")
    insertion_code = "".join(c for c in res_seq if c.isalpha())

    try:
      estat_resid.append(int(res_num_str) if res_num_str else -1)
    except ValueError:
      estat_resid.append(-1)
    # Chain ID might be multiple characters, use first char or -1 if empty
    estat_chain_id.append(ord(chain[0]) if chain else -1)

    # Format for PDB: residue number (4 chars) + insertion code (1 char)
    # PDB columns: 23-26 for resSeq (right-aligned), 27 for iCode
    # Handle edge case: if residue number is too long, truncate to fit 4 columns
    if res_num_str and len(res_num_str) > 4:  # noqa: PLR2004
      res_num_str = res_num_str[-4:]  # Keep last 4 digits
    res_num_formatted = f"{res_num_str:>4}" if res_num_str else "    "
    icode = insertion_code[:1] if insertion_code else " "
    chain_char = chain[0] if chain else " "  # PDB format uses single character chain ID

    # PDB format with strict column alignment
    # ATOM records: cols 1-6, 7-11, 12, 13-16, 17, 18-20, 21, 22, 23-26, 27,
    # 28-30, 31-38, 39-46, 47-54, 55-60, 61-66
    pdb_line = (
      f"{fields[0]:<6}"  # 1-6: Record name
      f"{serial:>5}"  # 7-11: Serial
      f" "  # 12: blank
      f"{atom_name:^4}"  # 13-16: Atom name
      f" "  # 17: Alt loc
      f"{res_name:>3}"  # 18-20: Res name
      f" "  # 21: blank
      f"{chain_char}"  # 22: Chain
      f"{res_num_formatted}"  # 23-26: Res seq
      f"{icode}"  # 27: iCode
      f"   "  # 28-30: blank
      f"{x:8.3f}"  # 31-38: X
      f"{y:8.3f}"  # 39-46: Y
      f"{z:8.3f}"  # 47-54: Z
      f"{occupancy:6.2f}"  # 55-60: Occupancy
      f"{bfactor:6.2f}"  # 61-66: B-factor
      f"          \n"  # 67+: element, charge, etc.
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
