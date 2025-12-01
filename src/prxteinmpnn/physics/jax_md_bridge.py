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
  masses: jnp.ndarray # (N,) Atomic masses (amu)
  sigmas: jnp.ndarray  # (N,)
  epsilons: jnp.ndarray  # (N,)
  gb_radii: jnp.ndarray  # (N,) Implicit solvent radii (mbondi2)
  scaled_radii: jnp.ndarray # (N,) Scaled radii for descreening (OBC2)
  bonds: jnp.ndarray  # (N_bonds, 2)
  bond_params: jnp.ndarray  # (N_bonds, 2) [length, k]
  constrained_bonds: jnp.ndarray # (N_constraints, 2)
  constrained_bond_lengths: jnp.ndarray # (N_constraints,)
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
  cmap_torsions: jnp.ndarray # (N_torsions, 5) [i, j, k, l, map_idx]


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

  # Load standard topology templates (fallback)
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

  # Helper: Get atom type
  def get_type(idx):
      if idx not in atom_info_map: return ""
      r_name, a_name = atom_info_map[idx]
      key = f"{r_name}_{a_name}"
      return force_field.atom_type_map.get(key, "")

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
      angle_lookup[(c3, c2, c1)] = (theta, k) 

  # Pre-calculate available residues in FF
  ff_residues = set(r for r, a in force_field.atom_key_to_id.keys())

  for r_i, res_name in enumerate(residues):
    if atom_counts is not None:
        count = atom_counts[r_i]
        res_atom_names = atom_names[current_atom_idx : current_atom_idx + count]
    else:
        # Legacy/Strict mode: Assume full residue
        expected_atoms = residue_constants.residue_atoms.get(res_name, [])
        res_atom_names = expected_atoms
        
    # --- Residue Mapping Logic ---
    # 1. Terminal Mapping
    mapped_res_name = res_name
    is_n_term = (r_i == 0)
    is_c_term = (r_i == len(residues) - 1)
    
    # Check for specific terminal atoms to confirm
    has_oxt = "OXT" in res_atom_names
    
    if is_c_term or has_oxt:
        # Generic C-terminal mapping (e.g. ALA -> CALA)
        # We assume standard Amber naming: C + ResName
        # Check if C+ResName exists in force field
        c_term_name = f"C{res_name}"
        if c_term_name in ff_residues:
             mapped_res_name = c_term_name
             
    elif is_n_term:
        # Generic N-terminal mapping (e.g. ALA -> NALA)
        # We assume standard Amber naming: N + ResName
        # Check if N+ResName exists in force field
        n_term_name = f"N{res_name}"
        if n_term_name in ff_residues:
            mapped_res_name = n_term_name

    # 2. Histidine Mapping (HIS -> HIE/HID/HIP)
    if res_name == "HIS":
        has_hd1 = "HD1" in res_atom_names
        has_he2 = "HE2" in res_atom_names
        
        if has_hd1 and has_he2:
            mapped_res_name = "HIP" # Both protonated
        elif has_hd1:
            mapped_res_name = "HID" # Delta protonated
        elif has_he2:
            mapped_res_name = "HIE" # Epsilon protonated (default neutral)
        else:
            mapped_res_name = "HIE" # Default fallback
            
    # Apply mapping
    res_name = mapped_res_name
    # -----------------------------

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
    # Priority: FF Templates > residue_constants
    
    bonds_found = False
    
    # 1. Try Force Field Templates (includes Hydrogens!)
    if hasattr(force_field, 'residue_templates') and res_name in force_field.residue_templates:
        bonds_found = True
        for atom1, atom2 in force_field.residue_templates[res_name]:
            if atom1 in local_map and atom2 in local_map:
                idx1 = local_map[atom1]
                idx2 = local_map[atom2]
                bonds_list.append([idx1, idx2])
                
                # Lookup params
                c1 = get_class(idx1)
                c2 = get_class(idx2)
                key = tuple(sorted((c1, c2)))
                
                if key in bond_lookup:
                    l_nm, k_nm = bond_lookup[key]
                    length = l_nm * 10.0
                    k = (k_nm / 418.4) # OpenMM k is already 2*k_amber, and we use 0.5*k*(r-r0)^2
                    bond_params_list.append([length, k])
                else:
                    bond_params_list.append([1.0, 300.0])
            else:
                 pass 

    # 2. Fallback to residue_constants (only if no template found)
    if not bonds_found and res_name in std_bonds:
      for bond in std_bonds[res_name]:
        if bond.atom1_name in local_map and bond.atom2_name in local_map:
          idx1 = local_map[bond.atom1_name]
          idx2 = local_map[bond.atom2_name]
          bonds_list.append([idx1, idx2])
          
          # Lookup params
          c1 = get_class(idx1)
          c2 = get_class(idx2)
          key = tuple(sorted((c1, c2)))
          
          if key in bond_lookup:
              l_nm, k_nm = bond_lookup[key]
              length = l_nm * 10.0
              k = (k_nm / 418.4)
              bond_params_list.append([length, k])
          else:
              bond_params_list.append([bond.length, 300.0])

    # Internal Angles - Generate from Bonds
    # Build local adjacency for this residue's atoms
    # We need to consider that bonds might connect to previous residue (Peptide Bond)
    # But here we are iterating residues.
    # Actually, it's better to generate ALL angles after ALL bonds are found for the whole system.
    # But parameterize_system iterates residues.
    # If we do it here, we only find intra-residue angles (and maybe peptide bond angles if we handle prev_c).
    
    # BETTER APPROACH:
    # Collect ALL bonds first (including peptide bonds), then generate ALL angles/dihedrals at the end.
    # Currently, bonds are collected per residue + peptide bond.
    # So `bonds_list` grows.
    # We can move Angle generation to AFTER the residue loop.
    # -----------------------------------------------------------------------
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
          k = (k_nm / 418.4)
          bond_params_list.append([length, k])
      else:
          bond_params_list.append([1.33, 300.0])
      
    # Update prev_c
    if "C" in local_map:
      prev_c_idx = local_map["C"]
    else:
      prev_c_idx = -1

    current_atom_idx += len(res_atom_names)

  # -----------------------------------------------------------------------
  # Generate Angles from Bonds (After all bonds are collected)
  # -----------------------------------------------------------------------
  # Build adjacency for angles
  adj_angles = {i: [] for i in range(n_atoms)}
  for b in bonds_list:
      adj_angles[b[0]].append(b[1])
      adj_angles[b[1]].append(b[0])

  angles_list = []
  angle_params_list = []
  seen_angles = set()

  for j in range(n_atoms):
      neighbors = adj_angles[j]
      if len(neighbors) < 2: continue
      
      import itertools
      for i, k in itertools.combinations(neighbors, 2):
          # Angle i-j-k
          # Sort i, k to avoid duplicates (i-j-k vs k-j-i)
          if i > k: i, k = k, i
          
          if (i, j, k) in seen_angles: continue
          seen_angles.add((i, j, k))
          
          angles_list.append([i, j, k])
          
          # Lookup params
          c1 = get_class(i)
          c2 = get_class(j)
          c3 = get_class(k)
          
          params = None
          if (c1, c2, c3) in angle_lookup:
              params = angle_lookup[(c1, c2, c3)]
          elif (c3, c2, c1) in angle_lookup:
              params = angle_lookup[(c3, c2, c1)]
              
          if params:
              theta, k_kj = params
              k = (k_kj / 4.184) # OpenMM k is already 2*k_amber
              angle_params_list.append([theta, k])
          else:
              # Fallback: 109.5 degrees (1.91 rad), k=100.0 (50 kcal/mol/rad^2)
              angle_params_list.append([1.91, 100.0])

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

  # Legacy exclusion mask
  exclusion_mask = (scale_matrix_vdw > 0.0)
  
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
  if pairs_14:
      p14 = jnp.array(list(pairs_14), dtype=jnp.int32)
      # Check if currently 0.0 (meaning 1-2 or 1-3)
      current_vals = scale_matrix_vdw[p14[:, 0], p14[:, 1]]
      
      # Only update if not already 0.0
      mask_update = current_vals > 0.0
      
      p14_update = p14[mask_update]
      if len(p14_update) > 0:
          scale_matrix_vdw = scale_matrix_vdw.at[p14_update[:, 0], p14_update[:, 1]].set(0.5)
          scale_matrix_vdw = scale_matrix_vdw.at[p14_update[:, 1], p14_update[:, 0]].set(0.5)
          
          scale_matrix_elec = scale_matrix_elec.at[p14_update[:, 0], p14_update[:, 1]].set(1.0 / 1.2)
          scale_matrix_elec = scale_matrix_elec.at[p14_update[:, 1], p14_update[:, 0]].set(1.0 / 1.2)

  # -----------------------------------------------------------------------
  # CMAP
  # -----------------------------------------------------------------------
  cmap_torsions_list = [] # [i, j, k, l, m]
  cmap_indices_list = [] # map_index
  
  for k in range(n_atoms):
      neighbors_k = adj[k]
      for j in neighbors_k:
          for l in neighbors_k:
              if j == l: continue 
              if j > l: continue 
              
              # Neighbors of j
              for i in adj[j]:
                  if i == k: continue
                  
                  # Neighbors of l
                  for m in adj[l]:
                      if m == k: continue
                      
                      # Path i-j-k-l-m
                      c_i, c_j, c_k, c_l, c_m = get_class(i), get_class(j), get_class(k), get_class(l), get_class(m)
                      t_i, t_j, t_k, t_l, t_m = get_type(i), get_type(j), get_type(k), get_type(l), get_type(m)
                      
                      for cmap_def in force_field.cmap_torsions:
                          def_classes = cmap_def['classes']
                          match = True
                          # Check each atom against class OR type
                          if def_classes[0] != c_i and def_classes[0] != t_i: match = False
                          elif def_classes[1] != c_j and def_classes[1] != t_j: match = False
                          elif def_classes[2] != c_k and def_classes[2] != t_k: match = False
                          elif def_classes[3] != c_l and def_classes[3] != t_l: match = False
                          elif def_classes[4] != c_m and def_classes[4] != t_m: match = False
                          
                          if match:
                              cmap_torsions_list.append([i, j, k, l, m])
                              cmap_indices_list.append(cmap_def['map_index'])
                              break
  
  # -----------------------------------------------------------------------
  # Impropers
  # -----------------------------------------------------------------------
  # -----------------------------------------------------------------------
  # Impropers
  # -----------------------------------------------------------------------
  impropers_list = []
  improper_params_list = []
  
  for k in range(n_atoms):
      neighbors = adj[k]
      if len(neighbors) != 3: continue
      
      c_k = get_class(k)
      
      best_match_score = -1
      best_terms = []
      best_perm = None
      
      import itertools
      for perm in itertools.permutations(neighbors, 3):
          i, j, l = perm
          c_i = get_class(i)
          c_j = get_class(j)
          c_l = get_class(l)
          
          # Check against all impropers
          # OpenMM/Amber XML usually defines Improper as Center-N1-N2-N3 (class1-class2-class3-class4)
          # where class1 is Center.
          # Or sometimes class3 is Center (Amber standard i-j-k-l).
          # We need to support both or infer from XML.
          # Based on protein.ff19SB.xml analysis, class1 seems to be Center.
          # So we match: class1=k, class2=i, class3=j, class4=l
          
          for improper in force_field.impropers:
              pc = improper['classes']
              
              # Match assuming class1 is Center (k)
              # And class2,3,4 are neighbors i,j,l
              match = True
              if pc[0] != '' and pc[0] != c_k: match = False
              elif pc[1] != '' and pc[1] != c_i: match = False
              elif pc[2] != '' and pc[2] != c_j: match = False
              elif pc[3] != '' and pc[3] != c_l: match = False
              
              if match:
                  score = sum(1 for x in pc if x != '')
                  if score > best_match_score:
                      best_match_score = score
                      best_terms = improper['terms']
                      best_perm = (i, j, k, l)
                  # If scores equal, we keep the first one found? Or last?
                  # Amber precedence usually says specific overrides general.
                  # If same specificity, order in file matters.
                  # We iterate propers in order. So we should update if score >= best?
                  # But permutations loop complicates this.
                  # Let's stick to strict > for now, or >= if we want last in file.
                  # But we are inside permutation loop.
                  elif score == best_match_score:
                       # If same score, maybe this permutation is better?
                       # Or maybe it's the same definition matched by different permutation?
                       # If terms are same, doesn't matter.
                       pass

      if best_terms and best_perm:
          i, j, k, l = best_perm
          # Amber improper geometry is usually i-j-k-l (k is center).
          # So we add [i, j, k, l]
          for term in best_terms:
              impropers_list.append([i, j, k, l])
              improper_params_list.append(term)

  # Convert to JAX arrays
  return {
    "charges": jnp.array(charges_list, dtype=jnp.float32),
    "masses": jnp.array(assign_masses(atom_names), dtype=jnp.float32),
    "sigmas": jnp.array(sigmas_list, dtype=jnp.float32),
    "epsilons": jnp.array(epsilons_list, dtype=jnp.float32),
    "gb_radii": jnp.array(
      assign_mbondi2_radii(atom_names, residues, bonds_list), dtype=jnp.float32
    ),
    "scaled_radii": jnp.array(
      assign_mbondi2_radii(atom_names, residues, bonds_list), dtype=jnp.float32
    ) * jnp.array(assign_obc2_scaling_factors(atom_names), dtype=jnp.float32),
    "bonds": jnp.array(bonds_list, dtype=jnp.int32).reshape(-1, 2),
    "bond_params": jnp.array(bond_params_list, dtype=jnp.float32).reshape(-1, 2),
    "constrained_bonds": jnp.array(
        [b for i, b in enumerate(bonds_list) if atom_names[b[0]].startswith("H") or atom_names[b[1]].startswith("H")],
        dtype=jnp.int32
    ).reshape(-1, 2),
    "constrained_bond_lengths": jnp.array(
        [bond_params_list[i][0] for i, b in enumerate(bonds_list) if atom_names[b[0]].startswith("H") or atom_names[b[1]].startswith("H")],
        dtype=jnp.float32
    ),
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


def assign_obc2_scaling_factors(atom_names: list[str]) -> list[float]:
  """Assign scaling factors for OBC2 GBSA calculation.
  
  Reference:
      Onufriev, Bashford, Case, Proteins 55, 383-394 (2004).
      
  Factors:
      H: 0.85
      C: 0.72
      N: 0.79
      O: 0.85
      F: 0.88
      P: 0.86
      S: 0.96
      Other: 0.80
  """
  factors = []
  for name in atom_names:
      element = name[0]
      if element == "H": factors.append(0.85)
      elif element == "C": factors.append(0.72)
      elif element == "N": factors.append(0.79)
      elif element == "O": factors.append(0.85)
      elif element == "F": factors.append(0.88)
      elif element == "P": factors.append(0.86)
      elif element == "S": factors.append(0.96)
      else: factors.append(0.80)
  return factors


def assign_masses(atom_names: list[str]) -> list[float]:
  """Assign atomic masses based on element type.
  
  Args:
      atom_names: List of atom names.
      
  Returns:
      List of masses in amu.
  """
  masses = []
  for name in atom_names:
      element = name[0]
      if element == "H": masses.append(1.008)
      elif element == "C": masses.append(12.011)
      elif element == "N": masses.append(14.007)
      elif element == "O": masses.append(15.999)
      elif element == "S": masses.append(32.06)
      elif element == "P": masses.append(30.97)
      elif element == "F": masses.append(18.998)
      else: masses.append(12.0) # Default
  return masses
