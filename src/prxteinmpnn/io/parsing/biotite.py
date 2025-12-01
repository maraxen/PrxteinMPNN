"""Parsing utilities for Biotite."""

import logging
import pathlib
from collections.abc import Sequence
from typing import Any

import hydride
import jax.numpy as jnp
import numpy as np
from biotite import structure
from biotite.structure import AtomArray, AtomArrayStack
from biotite.structure import io as structure_io

from prxteinmpnn.io.parsing.structures import ProcessedStructure
from prxteinmpnn.io.parsing.utils import (
  _process_chain_id,
  _validate_atom_array_type,
  processed_structure_to_protein_tuples,
)
from prxteinmpnn.utils.data_structures import (
  EstatInfo,
  ProteinStream,
)

logger = logging.getLogger(__name__)


def _remove_solvent_from_structure(
  atom_array: AtomArray | AtomArrayStack,
) -> AtomArray | AtomArrayStack:
  """Remove solvent atoms from the structure."""
  solvent_mask = structure.filter_solvent(atom_array)
  if np.any(solvent_mask):
    n_solvent = np.sum(solvent_mask)
    logger.info("Removing %d solvent atoms", n_solvent)

    if isinstance(atom_array, AtomArrayStack):
      atom_array = atom_array[:, ~solvent_mask]
    else:
      atom_array = atom_array[~solvent_mask]

    logger.debug("Structure after solvent removal: %d atoms", atom_array.array_length())
  return atom_array


def _add_hydrogens_to_structure(
  atom_array: AtomArray | AtomArrayStack,
) -> AtomArray | AtomArrayStack:
  """Add hydrogens to the structure using Hydride."""
  # Check if hydrogens exist
  has_hydrogens = (atom_array.element == "H").any()
  if has_hydrogens:
    logger.debug("Structure already has hydrogens.")
    return atom_array

  logger.info("No hydrogens found. Adding hydrogens using Hydride.")

  if isinstance(atom_array, AtomArrayStack):
    logger.warning(
      "Hydride integration with AtomArrayStack is experimental. "
      "Processing frame 0 only for now or iterating?",
    )
    # Hydride returns a single AtomArray.
    # For now, let's just return the stack as is if we can't easily protonate it,
    # or warn.
    # Ideally we should iterate and protonate each frame, but topology might change?
    # Let's just support AtomArray for now for hydride.
    return atom_array

  # Hydride requires a BondList. If not present, we must infer it.
  if not atom_array.bonds:
    logger.info("No BondList found. Inferring bonds via residue names.")
    try:
      atom_array.bonds = structure.connect_via_residue_names(atom_array)  # type: ignore[unresolved-attribute]
    except Exception as e:  # noqa: BLE001
      logger.warning("Failed to connect via residue names: %s. Falling back to distances.", e)
      atom_array.bonds = structure.connect_via_distances(atom_array)  # type: ignore[unresolved-attribute]

  # Hydride also requires a 'charge' annotation, even if 0.
  if "charge" not in atom_array.get_annotation_categories():
    logger.info("No charge annotation found. Adding zero charges for Hydride compatibility.")
    charges = np.zeros(atom_array.array_length(), dtype=int)
    
    # Heuristic: Set N-terminal Nitrogen charge to +1 to get NH3+
    if atom_array.array_length() > 0:
        res_ids = atom_array.res_id
        res_names = atom_array.res_name
        atom_names = atom_array.atom_name
        
        # 1. N-term
        if len(res_ids) > 0:
            first_res_id = res_ids[0]
            mask_n = (res_ids == first_res_id) & (atom_names == "N")
            charges[mask_n] = 1
            
            # C-term
            last_res_id = res_ids[-1]
            # Check for OXT
            mask_oxt = (res_ids == last_res_id) & (atom_names == "OXT")
            if np.any(mask_oxt):
                charges[mask_oxt] = -1
            
        # 2. Standard Ionizable Residues (pH 7)
        # LYS: NZ +1
        mask_lys = (res_names == "LYS") & (atom_names == "NZ")
        charges[mask_lys] = 1
        
        # ARG: CZ +1 failed. NH1 +1 caused asymmetry.
        # We will handle ARG protonation manually after Hydride.
        # So do NOT set charge here.
        pass
        
        # ASP: OD2 -1 (Arbitrary choice between OD1/OD2)
        mask_asp = (res_names == "ASP") & (atom_names == "OD2")
        charges[mask_asp] = -1
        
        # GLU: OE2 -1
        mask_glu = (res_names == "GLU") & (atom_names == "OE2")
        charges[mask_glu] = -1
        
    atom_array.set_annotation("charge", charges)

  atom_array, _ = hydride.add_hydrogen(atom_array)
  
  # Post-processing: Fix ARG protonation (ensure 5 sidechain H)
  atom_array = _fix_arg_protonation(atom_array)
  
  logger.info("Hydrogens added. New atom count: %d", atom_array.array_length())

  return atom_array


def _fix_arg_protonation(atom_array: AtomArray) -> AtomArray:
    """Ensure Arginine residues have correct protonation (HE, HH11, HH12, HH21, HH22)."""
    # Iterate ARGs
    # This is slow but safe.
    # We need to rebuild the AtomArray if we add atoms.
    # Biotite AtomArray is not easily resizeable in place?
    # We can append.
    
    # Identify ARGs
    arg_mask = atom_array.res_name == "ARG"
    if not np.any(arg_mask):
        return atom_array
        
    res_ids = atom_array.res_id
    arg_res_ids = np.unique(res_ids[arg_mask])
    
    atoms_to_add = [] # List of atoms to append
    
    for rid in arg_res_ids:
        # Get atoms for this residue
        mask = res_ids == rid
        res_atoms = atom_array[mask]
        
        # Check for HE
        has_he = np.any(res_atoms.atom_name == "HE")
        
        # Check NH1 hydrogens
        nh1_h = [name for name in res_atoms.atom_name if name.startswith("HH1")]
        # Check NH2 hydrogens
        nh2_h = [name for name in res_atoms.atom_name if name.startswith("HH2")]
        
        # We expect HE, HH11, HH12, HH21, HH22
        # If Hydride produced neutral, we might have HE, HH11, HH21, HH22 (missing HH12)
        # or HE, HH11, HH12, HH21 (missing HH22)
        # or HE, HH11, HH21 (missing one on each?)
        
        # Standardize names if needed? Hydride might use H, H2...
        # But let's assume we need to ADD missing ones.
        
        # Find parent atoms
        try:
            nh1_idx = np.where(res_atoms.atom_name == "NH1")[0][0]
            nh2_idx = np.where(res_atoms.atom_name == "NH2")[0][0]
            cz_idx = np.where(res_atoms.atom_name == "CZ")[0][0]
            
            nh1_coord = res_atoms.coord[nh1_idx]
            nh2_coord = res_atoms.coord[nh2_idx]
            cz_coord = res_atoms.coord[cz_idx]
        except IndexError:
            continue # Missing heavy atoms?
            
        # Helper to add H
        def add_h(parent_name, h_name, parent_coord, grand_parent_coord, existing_h_coords):
            # Calculate position
            # Vector CZ -> NH
            v_bond = parent_coord - grand_parent_coord
            v_bond = v_bond / np.linalg.norm(v_bond)
            
            # We want trigonal planar geometry.
            # If we have existing H, place new H to balance.
            # If no existing H, place two H symmetric.
            
            # For simplicity, let's just place it at parent + offset vector
            # that is roughly correct.
            # v_bond is direction of bond. H should be ~120 deg from it.
            # We need a vector perpendicular to v_bond.
            # Arbitrary axis?
            # Better: use cross product with random vector (or Z axis) to get normal.
            
            # Simple heuristic: Just place it and let minimization fix it?
            # But we need to avoid clashes.
            
            # Let's try to place it along v_bond rotated.
            # Or just take v_bond, add random noise, normalize, scale to 1.0 A.
            # But direction matters.
            # v_bond points TO parent.
            # H should point AWAY from parent, roughly along v_bond direction?
            # No, CZ-NH1-H angle is 120.
            # So H is roughly extension of CZ-NH1 bond.
            # v_bond = NH1 - CZ.
            # H pos = NH1 + v_bond * 1.0 (linear).
            # This is 180 deg. 120 deg is better.
            # But 180 is a good starting guess that won't clash with CZ.
            # It might clash with other things.
            
            # If we have existing H, we should place new H away from it.
            
            new_pos = parent_coord + v_bond * 1.01 # 1.01 A bond length
            
            # Create atom
            new_atom = structure.Atom(
                coord=new_pos,
                chain_id=res_atoms.chain_id[0],
                res_id=rid,
                res_name="ARG",
                atom_name=h_name,
                element="H"
            )
            atoms_to_add.append(new_atom)

        # Check NH1
        if len(nh1_h) < 2:
            # Add missing
            # Identify which one is missing? HH11 or HH12?
            # Just add HH12 if HH11 exists, or both.
            existing = [name for name in res_atoms.atom_name if name.startswith("HH1")]
            needed = ["HH11", "HH12"]
            for n in needed:
                if n not in existing:
                    # Check if "H" or "H1" exists and map it?
                    # Hydride might name them differently.
                    # But if count is < 2, we need to add.
                    add_h("NH1", n, nh1_coord, cz_coord, [])
                    
        # Check NH2
        if len(nh2_h) < 2:
            existing = [name for name in res_atoms.atom_name if name.startswith("HH2")]
            needed = ["HH21", "HH22"]
            for n in needed:
                if n not in existing:
                    add_h("NH2", n, nh2_coord, cz_coord, [])
                    
    if atoms_to_add:
        # Append atoms using concatenation
        added_array = structure.array(atoms_to_add)
        combined = atom_array + added_array
        
        # Sort by res_id to keep residues together
        # Note: stable sort is preferred to keep atom order within residue, 
        # but new atoms will be at end of residue block if we sort by res_id.
        indices = np.argsort(combined.res_id, kind='stable')
        combined = combined[indices]
        
        return combined
        
    return atom_array


def load_structure_with_hydride(
  source: str | pathlib.Path,
  model: int | None = None,
  altloc: str | None = None,
  topology: str | pathlib.Path | None = None,
  *,
  add_hydrogens: bool = True,
  remove_solvent: bool = True,
  **kwargs: Any,  # noqa: ANN401
) -> AtomArray | AtomArrayStack:
  """Load a structure and optionally add hydrogens using Hydride.

  Args:
      source: Path to structure file
      model: Model number to load (for multi-model files)
      altloc: Alternate location identifier
      topology: Optional topology file
      add_hydrogens: Whether to add hydrogens using hydride
      remove_solvent: Whether to remove solvent molecules (water, ions)
      **kwargs: Additional arguments for structure loading

  Returns:
      AtomArray or AtomArrayStack with processed structure

  """
  altloc = altloc if altloc is not None else "first"
  topology_array = None

  if topology is not None:
    topology_array = structure_io.load_structure(topology)

  if pathlib.Path(source).suffix.lower() in [".xtc"]:
    atom_array = structure_io.load_structure(
      source,
      template=topology_array,
      **kwargs,
    )
  else:
    atom_array = structure_io.load_structure(
      source,
      model=model,
      altloc=altloc,
      template=topology_array,
      **kwargs,
    )

  logger.info(
    "Loading structure from %s (add_hydrogens=%s, remove_solvent=%s)",
    source,
    add_hydrogens,
    remove_solvent,
  )
  _validate_atom_array_type(atom_array)

  if remove_solvent:
    atom_array = _remove_solvent_from_structure(atom_array)

  if add_hydrogens:
    atom_array = _add_hydrogens_to_structure(atom_array)

  return atom_array


def _parse_biotite(
  source: str | pathlib.Path,
  model: int | None,
  altloc: str | None,
  chain_id: Sequence[str] | str | None,
  topology: str | pathlib.Path | None = None,
  estat_info: EstatInfo | None = None,
  *,
  extract_dihedrals: bool = False,
  add_hydrogens: bool = True,
  **kwargs: Any,  # noqa: ANN401
) -> ProteinStream:
  """Parse standard structure files using biotite."""
  logger.info(
    "Starting Biotite parsing for source: %s (model: %s, altloc: %s)",
    source,
    model,
    altloc,
  )
  try:
    # Load structure with hydride (adds H if missing)
    atom_array = load_structure_with_hydride(
      source,
      model=model,
      altloc=altloc,
      topology=topology,
      add_hydrogens=add_hydrogens,
      **kwargs,
    )

    if chain_id is not None:
      atom_array, _ = _process_chain_id(atom_array, chain_id)

    # Create ProcessedStructure
    charges = None
    radii = None
    epsilons = None

    if estat_info:
      charges = estat_info.charges
      radii = estat_info.radii
      epsilons = estat_info.epsilons

    processed = ProcessedStructure(
      atom_array=atom_array,
      r_indices=atom_array.res_id,
      chain_ids=np.zeros(atom_array.array_length(), dtype=np.int32),  # Placeholder
      charges=charges,
      radii=radii,
      epsilons=epsilons,
    )

    yield from processed_structure_to_protein_tuples(
      processed,
      source_name=str(source),
      extract_dihedrals=extract_dihedrals,
    )

  except Exception as e:
    msg = f"Failed to parse structure from source: {source}. {type(e).__name__}: {e}"
    logger.exception(msg)
    raise RuntimeError(msg) from e


def biotite_to_jax_md_system(
  atom_array: AtomArray | AtomArrayStack,
  force_field: Any,  # noqa: ANN401
) -> tuple[dict[str, Any], jnp.ndarray]:
  """Converts a Biotite AtomArray to JAX MD SystemParams and coordinates.

  Args:
      atom_array: The structure to parameterize.
      force_field: The loaded FullForceField object.

  Returns:
      Tuple of (SystemParams, coordinates).

  """
  from prxteinmpnn.physics import jax_md_bridge

  # Ensure single frame
  if isinstance(atom_array, AtomArrayStack):
    atom_array = atom_array[0]

  # Extract data
  res_names = []
  atom_names = []
  atom_counts = []

  for res_atoms in structure.residue_iter(atom_array):
    res_names.append(res_atoms[0].res_name)
    atom_counts.append(len(res_atoms))
    for atom in res_atoms:
      atom_names.append(atom.atom_name)

  params = jax_md_bridge.parameterize_system(force_field, res_names, atom_names, atom_counts)
  coords = jnp.array(atom_array.coord)

  return params, coords
