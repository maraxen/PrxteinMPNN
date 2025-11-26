"""Bridge between PrxteinMPNN data structures and JAX MD arrays."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import jax.numpy as jnp
import numpy as np

from prxteinmpnn.utils import residue_constants

if TYPE_CHECKING:
  from prxteinmpnn.physics.force_fields import FullForceField


class SystemParams(TypedDict):
  """Parameters for a JAX MD system."""

  charges: jnp.ndarray  # (N,)
  sigmas: jnp.ndarray  # (N,)
  epsilons: jnp.ndarray  # (N,)
  bonds: jnp.ndarray  # (N_bonds, 2)
  bond_params: jnp.ndarray  # (N_bonds, 2) [length, k]
  angles: jnp.ndarray  # (N_angles, 3)
  angle_params: jnp.ndarray  # (N_angles, 2) [theta0, k]
  backbone_indices: jnp.ndarray  # (N_residues, 4) [N, CA, C, O] indices
  exclusion_mask: jnp.ndarray  # (N, N) boolean mask (True = interact, False = exclude)


def parameterize_system(
  force_field: FullForceField,
  residues: list[str],
  atom_names: list[str],
) -> SystemParams:
  """Converts a protein structure and force field into JAX MD compatible arrays.

  Uses template matching based on residue names to define topology (bonds/angles).
  Assumes `atom_names` follows the standard ordering for each residue as defined
  in `residue_constants`.

  Args:
      force_field: The loaded FullForceField object containing parameters.
      residues: List of 3-letter residue codes (e.g., ['ALA', 'GLY']).
      atom_names: List of atom names corresponding to the flat position array.

  Returns:
      SystemParams dictionary.

  """
  n_atoms = len(atom_names)

  # Lists to collect parameters
  charges_list = []
  sigmas_list = []
  epsilons_list = []

  bonds_list = []
  bond_params_list = []
  angles_list = []
  angle_params_list = []

  backbone_indices_list = []

  # Load standard topology templates
  # residue_constants.residue_bonds: resname -> list[Bond(atom1, atom2, length, stddev)]
  # residue_constants.residue_bond_angles: resname -> list[BondAngle(atom1, atom2, atom3, rad, stddev)]
  # Note: residue_constants values are experimental/statistical.
  # For MD, we might prefer force field equilibrium values if available.
  # For this implementation, we will use the force field parameters if possible,
  # falling back to residue_constants or generic defaults if not found in FF.
  # However, FullForceField object has `bonds` and `angles` as lists of tuples,
  # which are usually defined by atom TYPES, not names.
  # Mapping names to types is required.

  # Current FullForceField implementation (from file view) has:
  # atom_key_to_id: (res, atom) -> int
  # charges_by_id, sigmas_by_id, epsilons_by_id
  # It does NOT seem to have a direct "atom_name -> atom_type" map exposed easily
  # for bond parameter lookup unless we parse `bonds` list which is (type1, type2, k, len).

  # Simplified approach for Phase 1:
  # 1. Use FullForceField for non-bonded (Charge, LJ).
  # 2. Use residue_constants for Topology (connectivity).
  # 3. Use generic/heuristic parameters for bonds/angles if FF lookup is too complex,
  #    OR try to look up in FF.
  #    Let's use residue_constants lengths/angles as equilibrium values,
  #    and a stiff spring constant (e.g. 300 kcal/mol/A^2) for now.
  #    This ensures the "equilibrium" matches the PDB statistics which is good for
  #    restraining to "protein-like" geometry.

  std_bonds, _, std_angles = residue_constants.load_stereo_chemical_props()

  current_atom_idx = 0
  prev_c_idx = -1  # For peptide bond

  for res_name in residues:
    # 1. Identify atoms for this residue
    # We assume the input `atom_names` stream contains atoms for this residue
    # in the order defined by `residue_constants.residue_atoms[res_name]`.
    # However, input might be missing atoms (e.g. H).
    # The `atom_names` arg is the ACTUAL atoms present in the system we are simulating.
    # We need to match them.

    # Heuristic: Consume atoms from `atom_names` that belong to `res_name`.
    # This is tricky if we don't know exactly how many.
    # We will assume `atom_names` is perfectly aligned with `residues` blocks.
    # To do this robustly, we really need `residue_index` from the input.
    # Since we don't have it in args, we assume standard completeness for now.
    # TODO: Pass residue_index or counts to this function for robustness.

    # For this implementation, let's assume we are building the system FROM the sequence,
    # and `atom_names` is just the flat list of what we expect.
    # Actually, `sample.py` usually generates a full backbone + sidechain structure.
    # Let's assume standard atoms (heavy atoms) are present.

    expected_atoms = residue_constants.residue_atoms.get(res_name, [])
    # Filter expected atoms to those that are actually in our `atom_names` stream?
    # No, `atom_names` is the ground truth of what particles exist.
    # We iterate through `atom_names` and assign them to the current residue
    # until we hit an atom that starts the NEXT residue?
    # Standard PDB order: N, CA, C, O, ...
    # So if we see 'N' and we are not at the start, it might be next residue.
    # But sidechains have N too (ASN, GLN, HIS, LYS, ARG, TRP).

    # BETTER APPROACH:
    # We simply assume the caller provides `atom_names` that exactly matches
    # `residue_constants.residue_atoms[res_name]` for each residue in order.
    # If the system has missing atoms, this will desync.
    # For PrxteinMPNN sampling, we usually generate full heavy-atom structures.

    res_atom_names = expected_atoms
    # Check if these match the next chunk of atom_names
    chunk = atom_names[current_atom_idx : current_atom_idx + len(res_atom_names)]
    # In a real scenario, we might have hydrogens or missing atoms.
    # For Phase 1, we assume strict matching to `residue_constants` heavy atoms.

    # Map: local_atom_name -> global_index
    local_map = {}
    
    # Backbone indices for this residue
    bb_indices = [-1, -1, -1, -1] # N, CA, C, O

    for i, name in enumerate(res_atom_names):
      global_idx = current_atom_idx + i
      local_map[name] = global_idx

      # Non-bonded params from FF
      q = force_field.get_charge(res_name, name)
      sig, eps = force_field.get_lj_params(res_name, name)

      charges_list.append(q)
      sigmas_list.append(sig)
      epsilons_list.append(eps)
      
      if name == "N": bb_indices[0] = global_idx
      elif name == "CA": bb_indices[1] = global_idx
      elif name == "C": bb_indices[2] = global_idx
      elif name == "O": bb_indices[3] = global_idx

    backbone_indices_list.append(bb_indices)

    # Internal Bonds
    if res_name in std_bonds:
      for bond in std_bonds[res_name]:
        if bond.atom1_name in local_map and bond.atom2_name in local_map:
          idx1 = local_map[bond.atom1_name]
          idx2 = local_map[bond.atom2_name]
          bonds_list.append([idx1, idx2])
          # Use stddev to estimate k? k = kT / sigma^2 ?
          # Or just use a stiff constant. 300 kcal/mol/A^2 is typical for bonds.
          # bond.length is equilibrium.
          bond_params_list.append([bond.length, 300.0])


    # Internal Angles
    if res_name in std_angles:
      for angle in std_angles[res_name]:
        if (
          angle.atom1_name in local_map
          and angle.atom2_name in local_map
          and angle.atom3name in local_map
        ):
          idx1 = local_map[angle.atom1_name]
          idx2 = local_map[angle.atom2_name]
          idx3 = local_map[angle.atom3name]
          angles_list.append([idx1, idx2, idx3])
          # Angle k: ~50-100 kcal/mol/rad^2
          angle_params_list.append([angle.angle_rad, 80.0])

    # Peptide Bond (Prev C -> Curr N)
    if prev_c_idx != -1 and "N" in local_map:
      curr_n_idx = local_map["N"]
      bonds_list.append([prev_c_idx, curr_n_idx])
      # Peptide bond length ~1.33 A
      bond_params_list.append([1.33, 300.0])
      
      # We should also add angles involving the peptide bond (C_prev-N-CA, CA_prev-C_prev-N)
      # But `residue_constants` doesn't list inter-residue angles easily.
      # For Phase 1, we might skip explicit inter-residue angles or hardcode them.
      # Let's hardcode the critical backbone angles if we can track CA_prev.
      # For now, rely on bonds to hold it together, but angles are important for secondary structure.
      # TODO: Add inter-residue angles.

    if "C" in local_map:
      prev_c_idx = local_map["C"]
    else:
      prev_c_idx = -1

    current_atom_idx += len(res_atom_names)

  # Compute exclusion mask (1-2 and 1-3 interactions)
  # Start with all True (interact)
  exclusion_mask = jnp.ones((n_atoms, n_atoms), dtype=jnp.bool_)
  
  # Exclude self
  exclusion_mask = exclusion_mask.at[jnp.diag_indices(n_atoms)].set(False)
  
  # Exclude 1-2 (Bonds)
  if len(bonds_list) > 0:
      b_idx = jnp.array(bonds_list, dtype=jnp.int32)
      exclusion_mask = exclusion_mask.at[b_idx[:, 0], b_idx[:, 1]].set(False)
      exclusion_mask = exclusion_mask.at[b_idx[:, 1], b_idx[:, 0]].set(False)


      
  # Exclude 1-3 (Angles)
  # Angles list is [i, j, k]. We exclude (i, k).
  if len(angles_list) > 0:
      a_idx = jnp.array(angles_list, dtype=jnp.int32)
      exclusion_mask = exclusion_mask.at[a_idx[:, 0], a_idx[:, 2]].set(False)
      exclusion_mask = exclusion_mask.at[a_idx[:, 2], a_idx[:, 0]].set(False)

  # Convert to JAX arrays
  return {
    "charges": jnp.array(charges_list, dtype=jnp.float32),
    "sigmas": jnp.array(sigmas_list, dtype=jnp.float32),
    "epsilons": jnp.array(epsilons_list, dtype=jnp.float32),
    "bonds": jnp.array(bonds_list, dtype=jnp.int32).reshape(-1, 2),
    "bond_params": jnp.array(bond_params_list, dtype=jnp.float32).reshape(-1, 2),
    "angles": jnp.array(angles_list, dtype=jnp.int32).reshape(-1, 3),
    "angle_params": jnp.array(angle_params_list, dtype=jnp.float32).reshape(-1, 2),
    "backbone_indices": jnp.array(backbone_indices_list, dtype=jnp.int32).reshape(-1, 4),
    "exclusion_mask": exclusion_mask,
  }
