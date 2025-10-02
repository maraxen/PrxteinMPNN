"""Utilities for processing and manipulating protein structures from foldcomp.

prxteinmpnn.utils.foldcomp_utils
"""

import logging
import warnings
from collections.abc import Sequence
from functools import cache
from typing import Literal

import foldcomp
import numpy as np

from prxteinmpnn.io.conversion import (
  string_to_protein_sequence,
)
from prxteinmpnn.utils.data_structures import ProteinStream, ProteinTuple

logger = logging.getLogger(__name__)

FoldCompDatabase = Literal[
  "esmatlas",
  "esmatlas_v2023_02",
  "highquality_clust30",
  "afdb_uniprot_v4",
  "afdb_swissprot_v4",
  "afdb_rep_v4",
  "afdb_rep_dark_v4",
  "afdb_h_sapiens",
  "a_thaliana",
  "c_albicans",
  "c_elegans",
  "d_discoideum",
  "d_melanogaster",
  "d_rerio",
  "e_coli",
  "g_max",
  "m_jannaschii",
  "m_musculus",
  "o_sativa",
  "r_norvegicus",
  "s_cerevisiae",
  "s_pombe",
  "z_mays",
]


@cache
def _setup_foldcomp_database(database: FoldCompDatabase) -> None:
  """Set up the FoldComp database synchronously.

  This is designed to be called from within a synchronous worker process.
  """
  foldcomp.setup(database)


def get_protein_structures(
  protein_ids: Sequence[str],
  database: FoldCompDatabase | None = None,
) -> ProteinStream:
  """Retrieve protein structures from the FoldComp database and return them as a list of ensembles.

  This is a synchronous, blocking function designed to be run in an executor.

  Args:
    protein_ids: A sequence of protein IDs to retrieve.
    database: The FoldCompDatabase to use.

  Returns:
    A list of ProteinEnsemble objects. Each ensemble contains the structure(s)
    for one of the requested protein IDs.

  """
  if database is None:
    warnings.warn(
      "No FoldComp database specified. Defaulting to 'afdb_rep_v4'.",
      stacklevel=2,
    )
    database = "afdb_rep_v4"
  _setup_foldcomp_database(database)
  with foldcomp.open(database, ids=protein_ids, decompress=False) as proteins:  # type: ignore[attr-access]
    for name, fcz in proteins:
      try:
        fcz_data = foldcomp.get_data(fcz)  # type: ignore[attr-access]

        phi = np.array(fcz_data["phi"], dtype=np.float64)
        psi = np.array(fcz_data["psi"], dtype=np.float64)
        omega = np.array(fcz_data["omega"], dtype=np.float64)
        dihedrals = np.stack([phi, psi, omega], axis=-1)

        coordinates = np.array(fcz_data["coordinates"], dtype=np.float32)
        sequence = string_to_protein_sequence(fcz_data["residues"])
        num_res = len(sequence)

        yield ProteinTuple(
          coordinates=coordinates,
          aatype=sequence,
          atom_mask=np.ones((coordinates.shape[0], 37), dtype=np.bool_),
          residue_index=np.arange(num_res),
          chain_index=np.zeros(num_res, dtype=np.int32),
          dihedrals=dihedrals,
          source=str(name),
        )
      except Exception as e:  # noqa: BLE001
        msg = f"Failed to process a FoldComp entry. Error: {e}"
        logger.warning(msg)
        continue
