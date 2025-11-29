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
  scale_matrix_vdw: jnp.ndarray # (N, N) scaling factors for VDW
  scale_matrix_elec: jnp.ndarray # (N, N) scaling factors for Electrostatics
  dihedrals: jnp.ndarray  # (N_dihedrals, 4)
  dihedral_params: jnp.ndarray  # (N_dihedrals, 3) [periodicity, phase, k]
  impropers: jnp.ndarray  # (N_impropers, 4)
  improper_params: jnp.ndarray  # (N_impropers, 3) [periodicity, phase, k]
  cmap_energy_grids: jnp.ndarray # (N_maps, Grid, Grid)
  cmap_indices: jnp.ndarray # (N_torsions,) map index for each torsion
  cmap_torsions: jnp.ndarray # (N_torsions, 5) [i, j, k, l, map_idx] - wait, we need indices of atoms. 
                             # Actually, CMAP is usually defined on phi/psi pairs.
                             # But general CMAP is a torsion-torsion map.
                             # In Amber, it's usually 5 atoms defining 2 torsions? No, it's 5 atoms A-B-C-D-E.
                             # Torsions are A-B-C-D and B-C-D-E.
                             # So we need to store the 5 atom indices.


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
  std_bonds, _, std_angles = residue_constants.load_stereo_chemical_props()

  current_atom_idx = 0
  prev_c_idx = -1  # For peptide bond

  # Map: global_idx -> (res_name, atom_name)
  atom_info_map = {} 
  
  # Helper: Get atom class
  def get_class(idx):
      if idx not in atom_info_map: return ""
      r_name, a_name = atom_info_map[idx]
      key = f"{r_name}_{a_name}"
      return force_field.atom_class_map.get(key, "")

  # Pre-process FF lookups
  bond_lookup = {}
  for b in force_field.bonds:
      # b is (class1, class2, length, k)
      c1, c2, l, k = b
      key = tuple(sorted((c1, c2)))
      bond_lookup[key] = (l, k)
      
  angle_lookup = {}
  for a in force_field.angles:
      # a is (class1, class2, class3, theta, k)
      c1, c2, c3, theta, k = a
      angle_lookup[(c1, c2, c3)] = (theta, k)
      # Also store reverse? Or check reverse on lookup?
      # Let's check reverse on lookup to save memory/time, or just store both.
      # Storing both is safer.
      angle_lookup[(c3, c2, c1)] = (theta, k) 

  for r_i, res_name in enumerate(residues):
    if atom_counts is not None:
        count = atom_counts[r_i]
        res_atom_names = atom_names[current_atom_idx : current_atom_idx + count]
    else:
        # Legacy/Strict mode: Assume full residue
        expected_atoms = residue_constants.residue_atoms.get(res_name, [])
        res_atom_names = expected_atoms
        
    # Map: local_atom_name -> global_index
    local_map = {}
    
    # Backbone indices for this residue
    bb_indices = [-1, -1, -1, -1] # N, CA, C, O

    for i, name in enumerate(res_atom_names):
      global_idx = current_atom_idx + i
      local_map[name] = global_idx
      atom_info_map[global_idx] = (res_name, name)

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
          
          # Lookup params
          c1 = get_class(idx1)
          c2 = get_class(idx2)
          
          # Try direct match or reverse
          key = tuple(sorted((c1, c2)))
          if key in bond_lookup:
              l_nm, k_nm = bond_lookup[key]
              # Convert units:
              # Length: nm -> Angstrom (x10)
              # Stiffness: kJ/mol/nm^2 -> kcal/mol/A^2
              # 1 kJ = 1/4.184 kcal
              # 1 nm = 10 A => 1 nm^2 = 100 A^2
              # k_A = k_nm * (1/4.184) / 100
              # AMBER uses E = k(r-r0)^2. JAX MD uses E = 1/2 k(r-r0)^2.
              # So we need to multiply k by 2.
              length = l_nm * 10.0
              k = (k_nm / 418.4) * 2.0
              bond_params_list.append([length, k])
          else:
              # Fallback (should not happen for valid FF)
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
          
          # Lookup params
          c1 = get_class(idx1)
          c2 = get_class(idx2)
          c3 = get_class(idx3)
          
          # Try match (forward or reverse)
          # Angle is defined by (c1, c2, c3). c2 is central.
          # Match (c1, c2, c3) or (c3, c2, c1)
          
          params = None
          if (c1, c2, c3) in angle_lookup:
              params = angle_lookup[(c1, c2, c3)]
          elif (c3, c2, c1) in angle_lookup:
              params = angle_lookup[(c3, c2, c1)]
              
          if params:
              theta, k_kj = params
              # Convert units:
              # Stiffness: kJ/mol/rad^2 -> kcal/mol/rad^2
              # AMBER uses E = k(theta-theta0)^2. Bonded.py uses E = 1/2 k(theta-theta0)^2.
              # So we need to multiply k by 2.
              k = (k_kj / 4.184) * 2.0
              angle_params_list.append([theta, k])
          else:
              angle_params_list.append([angle.angle_rad, 80.0])

    # Peptide Bond (Prev C -> Curr N)
    if prev_c_idx != -1 and "N" in local_map:
      curr_n_idx = local_map["N"]
      bonds_list.append([prev_c_idx, curr_n_idx])
      
      c1 = get_class(prev_c_idx)
      c2 = get_class(curr_n_idx)
      key = tuple(sorted((c1, c2)))
      
      if key in bond_lookup:
          l_nm, k_nm = bond_lookup[key]
          length = l_nm * 10.0
          k = (k_nm / 418.4) * 2.0
          bond_params_list.append([length, k])
      else:
          bond_params_list.append([1.33, 300.0])
      
    if "C" in local_map:
      prev_c_idx = local_map["C"]
    else:
      prev_c_idx = -1

    current_atom_idx += len(res_atom_names)

  # -----------------------------------------------------------------------
  # Scaling Matrices (1-2, 1-3, 1-4)
  # -----------------------------------------------------------------------
  # Initialize with 1.0
  scale_matrix_vdw = jnp.ones((n_atoms, n_atoms), dtype=jnp.float32)
  scale_matrix_elec = jnp.ones((n_atoms, n_atoms), dtype=jnp.float32)
  
  # Mask self
  diag_indices = jnp.diag_indices(n_atoms)
  scale_matrix_vdw = scale_matrix_vdw.at[diag_indices].set(0.0)
  scale_matrix_elec = scale_matrix_elec.at[diag_indices].set(0.0)
  
  # 1-2 (Bonds) -> 0.0
  if len(bonds_list) > 0:
      b_idx = jnp.array(bonds_list, dtype=jnp.int32)
      scale_matrix_vdw = scale_matrix_vdw.at[b_idx[:, 0], b_idx[:, 1]].set(0.0)
      scale_matrix_vdw = scale_matrix_vdw.at[b_idx[:, 1], b_idx[:, 0]].set(0.0)
      scale_matrix_elec = scale_matrix_elec.at[b_idx[:, 0], b_idx[:, 1]].set(0.0)
      scale_matrix_elec = scale_matrix_elec.at[b_idx[:, 1], b_idx[:, 0]].set(0.0)

  # 1-3 (Angles) -> 0.0
  if len(angles_list) > 0:
      a_idx = jnp.array(angles_list, dtype=jnp.int32)
      scale_matrix_vdw = scale_matrix_vdw.at[a_idx[:, 0], a_idx[:, 2]].set(0.0)
      scale_matrix_vdw = scale_matrix_vdw.at[a_idx[:, 2], a_idx[:, 0]].set(0.0)
      scale_matrix_elec = scale_matrix_elec.at[a_idx[:, 0], a_idx[:, 2]].set(0.0)
      scale_matrix_elec = scale_matrix_elec.at[a_idx[:, 2], a_idx[:, 0]].set(0.0)

  # Legacy exclusion mask (for backward compat or if needed)
  exclusion_mask = (scale_matrix_vdw > 0.0) # Roughly correct, but 1-4 are > 0.
  # Wait, exclusion_mask in system.py was used to MASK interactions.
  # If 1-4 are scaled, they are NOT excluded.
  # So exclusion_mask should only be False for 1-2 and 1-3.
  # Which corresponds to scale == 0.0.
  
  # -----------------------------------------------------------------------
  # Torsions (Dihedrals) & 1-4 Scaling
  # -----------------------------------------------------------------------
  adj = {i: [] for i in range(n_atoms)}
  for b in bonds_list:
      adj[b[0]].append(b[1])
      adj[b[1]].append(b[0])

  dihedrals_list = []
  dihedral_params_list = []



  # Find Dihedrals
  seen_dihedrals = set()
  
  # 1-4 pairs set to avoid duplicates
  pairs_14 = set()

  for b in bonds_list:
      j, k = b[0], b[1]
      neighbors_j = [n for n in adj[j] if n != k]
      neighbors_k = [n for n in adj[k] if n != j]
      
      for i in neighbors_j:
          for l in neighbors_k:
              # 1-4 Pair: i and l
              if i < l: pairs_14.add((i, l))
              else: pairs_14.add((l, i))

              # Dihedral i-j-k-l
              if (l, k, j, i) in seen_dihedrals: continue
              seen_dihedrals.add((i, j, k, l))
              
              c_i, c_j, c_k, c_l = get_class(i), get_class(j), get_class(k), get_class(l)
              
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
                      dihedrals_list.append([i, j, k, l])
                      dihedral_params_list.append(term)

  # Apply 1-4 Scaling
  # Note: 1-4 scaling applies to ALL 1-4 pairs, even if no dihedral term exists in FF.
  # But usually they go together.
  # We need to ensure we don't overwrite 1-2 or 1-3 if they happen to be 1-4 (e.g. rings).
  # But for proteins (linear), 1-4 is distinct from 1-2/1-3 except in small rings (PRO).
  # In PRO, 1-4 might be 1-3? No, 5-mem ring.
  # Standard practice: If it's 1-2 or 1-3, it's 0.0. If it's 1-4 AND NOT 1-2/1-3, it's scaled.
  
  if pairs_14:
      p14 = jnp.array(list(pairs_14), dtype=jnp.int32)
      # Check if currently 0.0 (meaning 1-2 or 1-3)
      current_vals = scale_matrix_vdw[p14[:, 0], p14[:, 1]]
      
      # Only update if not already 0.0
      # We can do this by mask
      mask_update = current_vals > 0.0
      
      p14_update = p14[mask_update]
      if len(p14_update) > 0:
          # 1-4 interactions: Scale instead of exclude
          # Electrostatics: 1/1.2 ~= 0.8333
          # Van der Waals: 1/2.0 = 0.5
          
          # Note: We only scale if the value is currently > 0.0 (meaning it wasn't excluded as 1-2 or 1-3)
          # The check `current_vals > 0.0` above ensures this.
          
          scale_matrix_vdw = scale_matrix_vdw.at[p14_update[:, 0], p14_update[:, 1]].set(0.5)
          scale_matrix_vdw = scale_matrix_vdw.at[p14_update[:, 1], p14_update[:, 0]].set(0.5)
          
          scale_matrix_elec = scale_matrix_elec.at[p14_update[:, 0], p14_update[:, 1]].set(1.0 / 1.2)
          scale_matrix_elec = scale_matrix_elec.at[p14_update[:, 1], p14_update[:, 0]].set(1.0 / 1.2)

  # -----------------------------------------------------------------------
  # CMAP
  # -----------------------------------------------------------------------
  # CMAP terms are 5-atom sequences: i-j-k-l-m.
  # They define two torsions: phi(i-j-k-l) and psi(j-k-l-m).
  # We need to find all such 5-atom chains and match against force_field.cmap_torsions.
  
  cmap_torsions_list = [] # [i, j, k, l, m]
  cmap_indices_list = [] # map_index
  
  # Iterate over central atoms k
  # We need path of length 4 edges (5 atoms).
  # i - j - k - l - m
  # Iterate over j-k and k-l bonds?
  # Let's iterate over all 5-atom paths.
  # Optimization: CMAP is usually only on backbone.
  # C_prev - N - CA - C - N_next
  # We can just look for backbone atoms if we want to be fast, but generic is better.
  
  # Iterate over middle atom k.
  # Find neighbors j, l.
  # Find neighbors of j (i).
  # Find neighbors of l (m).
  
  # This is O(N * degree^4). For proteins degree ~ 3-4. Fast enough.
  
  for k in range(n_atoms):
      neighbors_k = adj[k]
      for j in neighbors_k:
          for l in neighbors_k:
              if j == l: continue # Distinct neighbors
              if j > l: continue # Avoid double counting central pair direction? 
              # Actually CMAP is directional usually? No, maps are usually symmetric or defined specifically.
              # But let's check all paths.
              
              # Neighbors of j
              for i in adj[j]:
                  if i == k: continue
                  
                  # Neighbors of l
                  for m in adj[l]:
                      if m == k: continue
                      
                      # Path i-j-k-l-m
                      # Check against FF
                      c_i, c_j, c_k, c_l, c_m = get_class(i), get_class(j), get_class(k), get_class(l), get_class(m)
                      classes = [c_i, c_j, c_k, c_l, c_m]
                      
                      # Match
                      for cmap_def in force_field.cmap_torsions:
                          # cmap_def['classes'] is tuple of 5
                          # Check match (wildcards?)
                          # Usually CMAP is very specific, no wildcards.
                          
                          # Check forward
                          if cmap_def['classes'] == tuple(classes):
                              cmap_torsions_list.append([i, j, k, l, m])
                              cmap_indices_list.append(cmap_def['map_index'])
                              break
                          
                          # Check reverse
                          # If the FF defines A-B-C-D-E, and we found E-D-C-B-A (which is valid chemically)
                          # We should add it as E-D-C-B-A.
                          # Note: The map index assumes the torsion order of the definition.
                          # If we match reverse, we are adding the reverse sequence.
                          # The energy function calculates phi/psi on the sequence passed.
                          # So if we pass E-D-C-B-A, it calculates phi(E-D-C-B) and psi(D-C-B-A).
                          # This corresponds to the reverse traversal.
                          # Is the map symmetric? Usually yes for backbone if it's just phi/psi.
                          # But strictly, we should match the definition.
                          if cmap_def['classes'] == tuple(classes[::-1]):
                              # We matched the reverse of the definition.
                              # So the sequence i-j-k-l-m corresponds to the reverse of the FF definition.
                              # This means we should probably add it as m-l-k-j-i to match the FF definition order?
                              # Yes, let's add it as m-l-k-j-i so the phi/psi calculation aligns with the map.
                              cmap_torsions_list.append([m, l, k, j, i])
                              cmap_indices_list.append(cmap_def['map_index'])
                              break
                          
  # -----------------------------------------------------------------------
  # Impropers
  # -----------------------------------------------------------------------
  impropers_list = []
  improper_params_list = []
  
  for k in range(n_atoms):
      neighbors = adj[k]
      if len(neighbors) < 3: continue
      
      import itertools
      for i, j, l in itertools.permutations(neighbors, 3):
          c_i = get_class(i)
          c_j = get_class(j)
          c_k = get_class(k)
          c_l = get_class(l)
          
          best_match_score = -1
          best_terms = []
          
          for improper in force_field.impropers:
              pc = improper['classes']
              score = sum(1 for x in pc if x != '')
              
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
    "scale_matrix_vdw": scale_matrix_vdw,
    "scale_matrix_elec": scale_matrix_elec,
    "dihedrals": jnp.array(dihedrals_list, dtype=jnp.int32).reshape(-1, 4),
    "dihedral_params": jnp.array(dihedral_params_list, dtype=jnp.float32).reshape(-1, 3),
    "impropers": jnp.array(impropers_list, dtype=jnp.int32).reshape(-1, 4),
    "improper_params": jnp.array(improper_params_list, dtype=jnp.float32).reshape(-1, 3),
    "cmap_energy_grids": force_field.cmap_energy_grids,
    "cmap_indices": jnp.array(cmap_indices_list, dtype=jnp.int32),
    "cmap_torsions": jnp.array(cmap_torsions_list, dtype=jnp.int32).reshape(-1, 5),
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
