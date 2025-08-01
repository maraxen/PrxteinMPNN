"""Utilities for processing and manipulating protein structures from foldcomp."""

import asyncio
import enum
from collections.abc import Iterator, Sequence
from functools import cache

import foldcomp
import nest_asyncio

from prxteinmpnn.io import from_string, protein_structure_to_model_inputs
from prxteinmpnn.mpnn import ModelWeights, ProteinMPNNModelVersion, get_mpnn_model
from prxteinmpnn.utils.data_structures import ModelInputs, ProteinStructure
from prxteinmpnn.utils.types import ModelParameters


class FoldCompDatabaseEnum(enum.Enum):
  """Enum for FoldComp databases."""

  ESMATLAS_FULL = "esmatlas"
  ESMATLAS_v2023_02 = "esmatlas_v2023_02"
  ESMATLAS_HIGH_QUALITY = "highquality_clust30"
  AFDB_UNIPROT_V4 = "afdb_uniprot_v4"
  AFDB_SWISSPROT_V4 = "afdb_swissprot_v4"
  AFDB_REP_V4 = "afdb_rep_v4"
  AFDB_REP_DARK_V4 = "afdb_rep_dark_v4"
  AFDB_H_SAPIENS = "afdb_h_sapiens"
  AFDB_A_THALIANA = "a_thaliana"
  AFDB_C_ALBICANS = "c_albicans"
  AFDB_C_ELEGANS = "c_elegans"
  AFDB_D_DISCOIDEUM = "d_discoideum"
  AFDB_D_MELANOGASTER = "d_melanogaster"
  AFDB_D_RERIO = "d_rerio"
  AFDB_E_COLI = "e_coli"
  AFDB_G_MAX = "g_max"
  AFDB_M_JANNASCHII = "m_jannaschii"
  AFDB_M_MUSCULUS = "m_musculus"
  AFDB_O_SATIVA = "o_sativa"
  AFDB_R_NORVEGICUS = "r_norvegicus"
  AFDB_S_CEREVISIAE = "s_cerevisiae"
  AFDB_S_POMBE = "s_pombe"
  AFDB_Z_MAYS = "z_mays"


@cache
def _setup_foldcomp_database(database: FoldCompDatabaseEnum) -> None:
  """Set up the FoldComp database, handling sync and async contexts.

  Args:
    database: The FoldCompDatabase enum value specifying which database to set up.

  Returns:
    None

  Example:
    >>> _setup_foldcomp_database(FoldCompDatabase.ESMATLAS_FULL)

  """
  try:
    loop = asyncio.get_running_loop()
    # If we're here, we're in an async context (e.g., Jupyter)

    nest_asyncio.apply()
    coro = foldcomp.setup_async(database.value)
    loop.run_until_complete(coro)
  except RuntimeError:
    foldcomp.setup(database.value)


def _get_protein_structures_from_database(
  proteins: foldcomp.FoldcompDatabase,  # type: ignore[attr-access]
) -> Iterator[ProteinStructure]:
  """Retrieve protein structures from the FoldComp database.

  Args:
    proteins: The FoldComp protein database object.

  Returns:
    An iterator over ProteinStructure objects containing the structure data
    for the specified protein IDs.

  """
  for _, pdb in proteins:
    protein_structure = from_string(pdb)
    yield protein_structure


def get_protein_structures(
  protein_ids: Sequence[str],
  database: FoldCompDatabaseEnum = FoldCompDatabaseEnum.AFDB_REP_V4,
) -> Iterator[ProteinStructure]:
  """Retrieve protein structures from the FoldComp database.

  Args:
    protein_ids: A sequence of protein IDs to retrieve.
    database: The FoldCompDatabase enum value specifying which database to use.

  Returns:
    An iterator over ProteinStructure objects containing the structure data
    for the specified protein IDs.

  Example:
    >>> ids = ["P12345", "Q67890"]
    >>> structures = get_protein_structures(ids)
    >>> for struct in structures:
    ...     print(struct)

  """
  _setup_foldcomp_database(database)
  with foldcomp.open(database.value, ids=protein_ids) as proteins:  # type: ignore[attr-access]
    yield from _get_protein_structures_from_database(proteins)


def model_from_id(
  protein_ids: str | Sequence[str],
  model_weights: ModelWeights | None = None,
  model_version: ProteinMPNNModelVersion | None = None,
) -> tuple[ModelParameters, Iterator[ModelInputs]]:
  """Get the MPNN model and inputs for specific protein IDs.

  Args:
    protein_ids: The ID(s) of the protein(s) to retrieve the model for.
    model_weights: The weights to use for the model.
    model_version: The model version to use.

  Returns:
    A tuple containing the MPNN model parameters and model inputs.

  Raises:
    ValueError: If no protein structures are found for the given IDs.

  Example:
    >>> model, inputs = model_from_id("P12345")
    >>> # Use model and inputs for inference

  """
  base_model = get_mpnn_model(
    model_version=model_version or ProteinMPNNModelVersion.V_48_002,
    model_weights=model_weights or ModelWeights.DEFAULT,
  )

  if isinstance(protein_ids, str):
    protein_ids = [protein_ids]
  structures = list(get_protein_structures(protein_ids=protein_ids))
  if not structures:
    msg = f"No protein structures found for IDs: {protein_ids}"
    raise ValueError(msg)

  model_inputs = (protein_structure_to_model_inputs(structure) for structure in structures)
  # Check if at least one model input is generated
  first_input = next(model_inputs, None)
  if first_input is None:
    msg = f"No model inputs generated for protein structures: {protein_ids}"
    raise ValueError(msg)

  def model_inputs_with_first() -> Iterator[ModelInputs]:
    yield first_input
    yield from model_inputs

  return base_model, model_inputs_with_first()
