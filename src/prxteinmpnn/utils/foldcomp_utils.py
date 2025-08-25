"""Utilities for processing and manipulating protein structures from foldcomp."""

import asyncio
from collections.abc import Iterator, Sequence
from functools import cache
from typing import Literal

import foldcomp
import jax.numpy as jnp
import nest_asyncio

from prxteinmpnn.io import (
  protein_structure_to_model_inputs,
  string_to_protein_sequence,
)
from prxteinmpnn.mpnn import ModelVersion, ModelWeights, get_mpnn_model
from prxteinmpnn.utils.data_structures import ModelInputs, ProteinStructure
from prxteinmpnn.utils.types import ModelParameters

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
  """Set up the FoldComp database, handling sync and async contexts.

  Args:
    database: The FoldCompDatabase enum value specifying which database to set up.

  Returns:
    None

  Example:
    >>> _setup_foldcomp_database("esmatlas")

  """
  try:
    loop = asyncio.get_running_loop()
    nest_asyncio.apply()
    coro = foldcomp.setup_async(database)
    loop.run_until_complete(coro)
  except RuntimeError:
    foldcomp.setup(database)


def _from_fcz(
  proteins: foldcomp.FoldcompDatabase,  # type: ignore[attr-access]
) -> Iterator[ProteinStructure]:
  """Retrieve protein dihedral structures from the FoldComp database.

  Args:
    proteins: The FoldComp protein database object.

  Returns:
    An iterator over DihedralStructure objects containing the dihedral angle data
    for the specified protein IDs.

  """
  for _, fcz in proteins:
    fcz_data = foldcomp.get_data(fcz)  # type: ignore[attr-access]
    phi_angles = jnp.array(fcz_data["phi"], dtype=jnp.float64)
    psi_angles = jnp.array(fcz_data["psi"], dtype=jnp.float64)
    omega_angles = jnp.array(fcz_data["omega"], dtype=jnp.float64)
    dihedrals = jnp.stack(
      [phi_angles, psi_angles, omega_angles],
      axis=-1,
    )
    coordinates = jnp.array(fcz_data["coordinates"], dtype=jnp.float64)
    residue_sequence = string_to_protein_sequence(fcz_data["residues"])
    yield ProteinStructure(
      coordinates=coordinates,
      dihedrals=dihedrals,
      aatype=residue_sequence,
      atom_mask=jnp.ones(len(residue_sequence)),
      residue_index=jnp.arange(len(residue_sequence)),
      chain_index=jnp.zeros(len(residue_sequence), dtype=jnp.int32),
    )


def get_protein_structures(
  protein_ids: Sequence[str],
  database: FoldCompDatabase = "afdb_rep_v4",
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
  with foldcomp.open(database, ids=protein_ids, decompress=False) as proteins:  # type: ignore[attr-access]
    yield from _from_fcz(proteins)


def model_from_id(
  protein_ids: str | Sequence[str],
  model_weights: ModelWeights | None = None,
  model_version: ModelVersion | None = None,
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
    model_version=model_version or "v_48_020.pkl",
    model_weights=model_weights or "original",
  )

  if isinstance(protein_ids, str):
    protein_ids = [protein_ids]
  structures = list(get_protein_structures(protein_ids=protein_ids))
  if not structures:
    msg = f"No protein structures found for IDs: {protein_ids}"
    raise ValueError(msg)

  model_inputs = (protein_structure_to_model_inputs(structure) for structure in structures)
  first_input = next(model_inputs, None)
  if first_input is None:
    msg = f"No model inputs generated for protein structures: {protein_ids}"
    raise ValueError(msg)

  def model_inputs_with_first() -> Iterator[ModelInputs]:
    yield first_input
    yield from model_inputs

  return base_model, model_inputs_with_first()
