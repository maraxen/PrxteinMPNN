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
  gb_radii: jnp.ndarray  # (N,) Implicit solvent radii (mbondi2)
  bonds: jnp.ndarray  # (N_bonds, 2)
  bond_params: jnp.ndarray  # (N_bonds, 2) [length, k]
  angles: jnp.ndarray  # (N_angles, 3)
  angle_params: jnp.ndarray  # (N_angles, 2) [theta0, k]
  backbone_indices: jnp.ndarray  # (N_residues, 4) [N, CA, C, O] indices
  exclusion_mask: jnp.ndarray  # (N, N) boolean mask (True = interact, False = exclude)
  dihedrals: jnp.ndarray  # (N_dihedrals, 4)
  dihedral_params: jnp.ndarray  # (N_dihedrals, 3) [periodicity, phase, k]
  impropers: jnp.ndarray  # (N_impropers, 4)
  improper_params: jnp.ndarray  # (N_impropers, 3) [periodicity, phase, k]


def parameterize_system(
  force_field: FullForceField,
  residues: list[str],
  atom_names: list[str],
  atom_counts: list[int] | None = None,
) -> SystemParams:
  """Converts a protein structure and force field into JAX MD compatible arrays.

  Uses template matching based on residue names to define topology (bonds/angles).
  Assumes `atom_names` follows the standard ordering for each residue as defined
  in `residue_constants`.

  Args:
      force_field: The loaded FullForceField object containing parameters.
      residues: List of 3-letter residue codes (e.g., ['ALA', 'GLY']).
      atom_names: List of atom names corresponding to the flat position array.
      atom_counts: Optional list of atom counts per residue. If provided, allows
                   handling missing atoms by slicing `atom_names` explicitly.

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

  for r_i, res_name in enumerate(residues):
    # 1. Identify atoms for this residue
    # We assume the input `atom_names` stream contains atoms for this residue
    # in the order defined by `residue_constants.residue_atoms[res_name]`.
    # However, input might be missing atoms (e.g. H).
    # The `atom_names` arg is the ACTUAL atoms present in the system we are simulating.
    # We need to match them.

    if atom_counts is not None:
        count = atom_counts[r_i]
        res_atom_names = atom_names[current_atom_idx : current_atom_idx + count]
    else:
        # Legacy/Strict mode: Assume full residue
        expected_atoms = residue_constants.residue_atoms.get(res_name, [])
        res_atom_names = expected_atoms
        
    # Check if these match the next chunk of atom_names (sanity check)
    # chunk = atom_names[current_atom_idx : current_atom_idx + len(res_atom_names)]
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

  # -----------------------------------------------------------------------
  # Torsions (Dihedrals)
  # -----------------------------------------------------------------------
  # We need to find all paths i-j-k-l of length 3.
  # 1. Build adjacency
  adj = {i: [] for i in range(n_atoms)}
  for b in bonds_list:
      adj[b[0]].append(b[1])
      adj[b[1]].append(b[0])

  dihedrals_list = []
  dihedral_params_list = []

  # Helper to get atom class
  # We need to reconstruct which residue each atom belongs to
  # We can do this by creating a map: global_idx -> (res_name, atom_name)
  # Since we iterated sequentially, we can rebuild this map or just store it during the loop.
  # Rebuilding is safer given the current structure.
  
  atom_info_map = {} # global_idx -> (res_name, atom_name)
  curr_idx = 0
  for r_i, res_name in enumerate(residues):
      if atom_counts is not None:
          count = atom_counts[r_i]
          r_atoms = atom_names[curr_idx : curr_idx + count]
      else:
          r_atoms = residue_constants.residue_atoms.get(res_name, [])
      
      for i, name in enumerate(r_atoms):
          atom_info_map[curr_idx + i] = (res_name, name)
      curr_idx += len(r_atoms)

  def get_class(idx):
      if idx not in atom_info_map: return ""
      r_name, a_name = atom_info_map[idx]
      key = f"{r_name}_{a_name}"
      return force_field.atom_class_map.get(key, "")

  # 2. Find Dihedrals
  # Iterate over central bonds j-k
  seen_dihedrals = set()

  for b in bonds_list:
      j, k = b[0], b[1]
      
      # Neighbors of j (excluding k)
      neighbors_j = [n for n in adj[j] if n != k]
      # Neighbors of k (excluding j)
      neighbors_k = [n for n in adj[k] if n != j]
      
      for i in neighbors_j:
          for l in neighbors_k:
              # Candidate i-j-k-l
              # Check if already added (reverse)
              if (l, k, j, i) in seen_dihedrals: continue
              seen_dihedrals.add((i, j, k, l))
              
              # Lookup parameters
              c_i = get_class(i)
              c_j = get_class(j)
              c_k = get_class(k)
              c_l = get_class(l)
              
              # Try to match in force_field.propers
              # propers is list of dicts: {'classes': [c1, c2, c3, c4], 'terms': [[n, phase, k], ...]}
              # We need to match [c_i, c_j, c_k, c_l] or [c_l, c_k, c_j, c_i]
              # Wildcards: '' matches anything.
              
              best_match_score = -1
              best_terms = []
              
              for proper in force_field.propers:
                  pc = proper['classes']
                  score = sum(1 for x in pc if x != '')
                  
                  # Forward
                  match_fwd = True
                  if pc[0] != '' and pc[0] != c_i: match_fwd = False
                  elif pc[1] != '' and pc[1] != c_j: match_fwd = False
                  elif pc[2] != '' and pc[2] != c_k: match_fwd = False
                  elif pc[3] != '' and pc[3] != c_l: match_fwd = False
                  
                  if match_fwd:
                      if score > best_match_score:
                          best_match_score = score
                          best_terms = proper['terms']
                      elif score == best_match_score:
                          best_terms = proper['terms']
                      continue

                  # Reverse
                  match_rev = True
                  if pc[0] != '' and pc[0] != c_l: match_rev = False
                  elif pc[1] != '' and pc[1] != c_k: match_rev = False
                  elif pc[2] != '' and pc[2] != c_j: match_rev = False
                  elif pc[3] != '' and pc[3] != c_i: match_rev = False
                  
                  if match_rev:
                      if score > best_match_score:
                          best_match_score = score
                          best_terms = proper['terms']
                      elif score == best_match_score:
                          best_terms = proper['terms']

              if best_terms:
                  for term in best_terms:
                      # term is [periodicity, phase, k]
                      dihedrals_list.append([i, j, k, l])
                      dihedral_params_list.append(term)

  # -----------------------------------------------------------------------
  # Improper Torsions
  # -----------------------------------------------------------------------
  # Amber definition: Central atom is the 3rd atom (k).
  # i-j-k-l where k is bonded to i, j, l.
  # We iterate over all atoms k, check if they have >= 3 neighbors.
  # If so, we check all permutations of 3 neighbors (i, j, l).
  # But usually the FF defines specific ordering or wildcards.
  # Amber FF usually defines impropers for planar centers (e.g. C in C=O).
  
  impropers_list = []
  improper_params_list = []
  
  for k in range(n_atoms):
      neighbors = adj[k]
      if len(neighbors) < 3: continue
      
      # We need to check triplets of neighbors (i, j, l) around k
      # For a planar center with exactly 3 neighbors, there is 1 group (with permutations).
      # Amber impropers are often defined with wildcards like X-X-C-O.
      # The central atom is the 3rd one.
      
      # Let's iterate over all triplets of neighbors
      import itertools
      for i, j, l in itertools.permutations(neighbors, 3):
          # Candidate i-j-k-l
          
          c_i = get_class(i)
          c_j = get_class(j)
          c_k = get_class(k)
          c_l = get_class(l)
          
          # Lookup in force_field.impropers
          # Same logic as propers, but for impropers list
          
          best_match_score = -1
          best_terms = []
          
          for improper in force_field.impropers:
              pc = improper['classes']
              score = sum(1 for x in pc if x != '')
              
              # Forward
              match_fwd = True
              if pc[0] != '' and pc[0] != c_i: match_fwd = False
              elif pc[1] != '' and pc[1] != c_j: match_fwd = False
              elif pc[2] != '' and pc[2] != c_k: match_fwd = False
              elif pc[3] != '' and pc[3] != c_l: match_fwd = False
              
              if match_fwd:
                  if score > best_match_score:
                      best_match_score = score
                      best_terms = improper['terms']
                  elif score == best_match_score:
                      best_terms = improper['terms']
                  continue
                  
              # Reverse? Improper definition usually implies specific order (central is 3rd).
              # But sometimes they are symmetric?
              # Amber standard: "The first and last atoms are interchangeable, as are the second and third." -> No, that's not right.
              # "The central atom is listed third."
              # "The other three atoms are bound to the central atom."
              # So i, j, l are interchangeable?
              # Actually, Amber impropers are torsion terms applied to i-j-k-l.
              # The ordering in the FF file matters for matching.
              # If the FF says A-B-C-D, it matches an improper where atoms have those types.
              # Since i,j,l are all bonded to k, the "path" i-j-k-l is valid (j-k is bond, k-l is bond, i-j is NOT necessarily bond).
              # Wait, improper i-j-k-l means i-j, j-k, k-l bonds? NO.
              # Improper i-j-k-l usually means k is bonded to i, j, l.
              # The angle is defined as the angle between planes (i,j,k) and (j,k,l).
              # So it treats it *as if* it were a dihedral i-j-k-l.
              # So we need to ensure we pass indices (i, j, k, l) such that the dihedral angle computed
              # corresponds to what the FF expects.
              # If the FF defines X-X-C-O (with C central), it usually means the O is the 4th atom (or 1st?).
              # Amber convention: Central atom is 3rd.
              # So if we have C bonded to CA, N, O. And FF has CA-N-C-O.
              # Then i=CA, j=N, k=C, l=O.
              # We need to find permutations of neighbors that match the FF.
              
              # So, we are already iterating permutations of neighbors (i, j, l).
              # So (i, j, k, l) covers all orderings.
              # We just need to check if it matches the FF entry.
              # And we assume the FF entry defines the order.
              
              # Reverse match?
              # If FF has A-B-C-D, does it match D-C-B-A?
              # For impropers, usually NO, because C is central (3rd).
              # D-C-B-A would imply B is central.
              # So we ONLY check forward match.
              
              pass 

          if best_terms:
              for term in best_terms:
                  impropers_list.append([i, j, k, l])
                  improper_params_list.append(term)

  # Convert to JAX arrays
  return {
    "charges": jnp.array(charges_list, dtype=jnp.float32),
    "sigmas": jnp.array(sigmas_list, dtype=jnp.float32),
    "epsilons": jnp.array(epsilons_list, dtype=jnp.float32),
    "gb_radii": jnp.array(
      assign_mbondi2_radii(atom_names, residues, bonds_list), dtype=jnp.float32
    ),
    "bonds": jnp.array(bonds_list, dtype=jnp.int32).reshape(-1, 2),
    "bond_params": jnp.array(bond_params_list, dtype=jnp.float32).reshape(-1, 2),
    "angles": jnp.array(angles_list, dtype=jnp.int32).reshape(-1, 3),
    "angle_params": jnp.array(angle_params_list, dtype=jnp.float32).reshape(-1, 2),
    "backbone_indices": jnp.array(backbone_indices_list, dtype=jnp.int32).reshape(-1, 4),
    "exclusion_mask": exclusion_mask,
    "dihedrals": jnp.array(dihedrals_list, dtype=jnp.int32).reshape(-1, 4),
    "dihedral_params": jnp.array(dihedral_params_list, dtype=jnp.float32).reshape(-1, 3),
    "impropers": jnp.array(impropers_list, dtype=jnp.int32).reshape(-1, 4),
    "improper_params": jnp.array(improper_params_list, dtype=jnp.float32).reshape(-1, 3),
  }


def assign_mbondi2_radii(
  atom_names: list[str],
  residue_names: list[str],
  bonds: list[list[int]],
) -> list[float]:
  """Assign intrinsic radii for Generalized Born calculations using mbondi2 scheme.

  Reference:
      Onufriev, Bashford, Case, "Exploring native states and large-scale dynamics with the generalized born model",
      Proteins 55, 383-394 (2004).

  Rules (MBondi2):
      C: 1.70 Å
      N: 1.55 Å
      O: 1.50 Å
      S: 1.80 Å
      H (generic): 1.20 Å
      H (bound to N): 1.30 Å
      C (C1/C2/C3 > 13.0 mass): 2.20 Å (Not fully implemented, using 1.70 default for C)

  Args:
      atom_names: List of atom names.
      residue_names: List of residue names (aligned with atom blocks).
      bonds: List of [i, j] bond indices.

  Returns:
      List of radii.
  """
  n_atoms = len(atom_names)
  radii = [0.0] * n_atoms

  # Build adjacency for H-bonding check
  adj = {i: [] for i in range(n_atoms)}
  for i, j in bonds:
    adj[i].append(j)
    adj[j].append(i)

  # Heuristic to map flat atom list to residues for context if needed
  # For mbondi2, we mostly need element type and neighbors.

  for i, name in enumerate(atom_names):
    element = name[0]  # Simple element inference

    if element == "H":
      # Check if bonded to Nitrogen
      is_bound_to_N = False
      for neighbor in adj[i]:
        if atom_names[neighbor].startswith("N"):
          is_bound_to_N = True
          break
      
      if is_bound_to_N:
        radii[i] = 1.30
      else:
        radii[i] = 1.20

    elif element == "C":
      # Simplified C radius (1.70). 
      # Full mbondi2 distinguishes C types by mass/hybridization which is hard to infer here without mass.
      # Most protein C are 1.70 except maybe some sidechain terminals?
      # The reference implementation uses 1.70 as default for C.
      radii[i] = 1.70
    
    elif element == "N":
      radii[i] = 1.55
    
    elif element == "O":
      radii[i] = 1.50
    
    elif element == "S":
      radii[i] = 1.80
    
    elif element == "P":
      radii[i] = 1.85
      
    elif element == "F":
      radii[i] = 1.50
      
    elif element == "Cl":
      radii[i] = 1.70
      
    else:
      # Default fallback
      radii[i] = 1.50

  return radii
