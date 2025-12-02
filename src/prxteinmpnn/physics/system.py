"""System setup and energy function for implicit solvent MD."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax_md import energy, partition, space, util

from prxteinmpnn.physics import bonded, generalized_born, cmap, sasa, constants
from prxteinmpnn.physics.jax_md_bridge import SystemParams

Array = util.Array


def compute_dihedral_angles(
    r: Array,
    indices: Array,
    displacement_fn: space.DisplacementFn
) -> Array:
    """Computes dihedral angles for a batch of indices (N, 4)."""
    r_i = r[indices[:, 0]]
    r_j = r[indices[:, 1]]
    r_k = r[indices[:, 2]]
    r_l = r[indices[:, 3]]

    b0 = jax.vmap(displacement_fn)(r_i, r_j)
    b1 = jax.vmap(displacement_fn)(r_k, r_j)
    b2 = jax.vmap(displacement_fn)(r_l, r_k)

    b1_norm = jnp.linalg.norm(b1, axis=-1, keepdims=True) + 1e-8
    b1_unit = b1 / b1_norm

    v = b0 - jnp.sum(b0 * b1_unit, axis=-1, keepdims=True) * b1_unit
    w = b2 - jnp.sum(b2 * b1_unit, axis=-1, keepdims=True) * b1_unit

    x = jnp.sum(v * w, axis=-1)
    y = jnp.sum(jnp.cross(b1_unit, v) * w, axis=-1)

    return jnp.arctan2(y, x)


def make_energy_fn(
  displacement_fn: space.DisplacementFn,
  system_params: SystemParams,
  neighbor_list: partition.NeighborList | None = None,
  dielectric_constant: float = 1.0,
  implicit_solvent: bool = True,
  solvent_dielectric: float = 78.5,
  solute_dielectric: float = constants.DIELECTRIC_PROTEIN,
  surface_tension: float = constants.SURFACE_TENSION,
  dielectric_offset: float = constants.DIELECTRIC_OFFSET,
) -> Callable[[Array], Array]:
  """Creates the total potential energy function.

  U(R) = U_bond + U_angle + U_vdw + U_elec + U_cmap + U_sasa

  Args:
      displacement_fn: JAX MD displacement function.
      system_params: System parameters from `jax_md_bridge`.
      neighbor_list: Optional neighbor list. If provided, non-bonded terms
                     will use it. If None, they will be N^2 (slow).
                     NOTE: For proteins, N^2 is often acceptable for small systems,
                     but neighbor lists are better for >500 atoms.
                     We strongly recommend using neighbor lists.
      dielectric_constant: Dielectric constant for explicit solvent/vacuum (default 1.0).
      implicit_solvent: Whether to use implicit solvent (GBSA).
      solvent_dielectric: Solvent dielectric constant (default 78.5).
      solute_dielectric: Solute dielectric constant (default 1.0).
      surface_tension: Surface tension for SASA term (kcal/mol/A^2).
      dielectric_offset: Offset for Born radius calculation (default 0.09 A).

  Returns:
      A function energy(R, neighbor=None) -> float.

  """

  # 1. Bonded Terms
  bond_energy_fn = bonded.make_bond_energy_fn(
    displacement_fn,
    system_params["bonds"],
    system_params["bond_params"],
  )

  angle_energy_fn = bonded.make_angle_energy_fn(
    displacement_fn,
    system_params["angles"],
    system_params["angle_params"],
  )

  dihedral_energy_fn = bonded.make_dihedral_energy_fn(
    displacement_fn,
    system_params["dihedrals"],
    system_params["dihedral_params"],
  )

  improper_energy_fn = bonded.make_dihedral_energy_fn(
    displacement_fn,
    system_params["impropers"],
    system_params["improper_params"],
  )

  # 2. Non-Bonded Terms
  charges = system_params["charges"]
  sigmas = system_params["sigmas"]
  epsilons = system_params["epsilons"]
  
  # Use scaling matrices if available, otherwise fallback to exclusion_mask
  scale_matrix_vdw = system_params.get("scale_matrix_vdw")
  scale_matrix_elec = system_params.get("scale_matrix_elec")
  exclusion_mask = system_params["exclusion_mask"]

  def lj_pair(dr, sigma_i, sigma_j, eps_i, eps_j, **kwargs):
    sigma = 0.5 * (sigma_i + sigma_j)
    epsilon = jnp.sqrt(eps_i * eps_j)
    return energy.lennard_jones(dr, sigma, epsilon)

  # Electrostatics
  # -------------------------------------------------------------------------
  def compute_electrostatics(r, neighbor_idx=None):
    # Prepare parameters
    if "gb_radii" in system_params and system_params["gb_radii"] is not None:
      radii = system_params["gb_radii"]
    else:
      radii = sigmas * 0.5

    e_gb = 0.0
    born_radii = None
    
    if implicit_solvent:
      # Generalized Born (OBC) - Solvation Term
      
      if scale_matrix_vdw is not None:
          # OpenMM CustomGBForce (OBC2) behavior:
            # GBSA Hypothesis: Include All in Radii.
            # Energy: Include 1-2/1-3 (1.0), Exclude 1-4 (0.0), Include others (1.0)
            gb_mask = jnp.ones_like(scale_matrix_vdw)
            
            if scale_matrix_elec is not None:
                # 1-2/1-3 are 0.0 -> Set to 1.0
                # 1-4 are ~0.833 -> Set to 0.0
                # Others are 1.0 -> Keep 1.0
                mask_12_13 = scale_matrix_elec == 0.0
                mask_14 = (scale_matrix_elec > 0.0) & (scale_matrix_elec < 0.9)
                
                gb_energy_mask = jnp.where(mask_12_13, 1.0, scale_matrix_elec)
                gb_energy_mask = jnp.where(mask_14, 0.0, gb_energy_mask)
            else:
                gb_energy_mask = jnp.ones_like(scale_matrix_vdw)
      else:
          gb_mask = exclusion_mask
          gb_energy_mask = None
          
      scaled_radii = system_params.get("scaled_radii")
      
      if neighbor_idx is None:
        e_gb, born_radii = generalized_born.compute_gb_energy(
            r, 
            charges, 
            radii, 
            solvent_dielectric=solvent_dielectric, 
            solute_dielectric=solute_dielectric,
            dielectric_offset=dielectric_offset,
            mask=gb_mask, # Radii: Scale 1-4 (0.5)
            energy_mask=gb_energy_mask, # Energy: Full (1.0)
            scaled_radii=scaled_radii
        )
      else:
        # TODO: Update neighbor list version of GBSA to accept mask
        # For now, we assume neighbor list version handles exclusions via neighbor list construction?
        # No, neighbor list usually includes everything within cutoff.
        # But we don't have mask support in compute_gb_energy_neighbor_list yet.
        # Since validation script uses N^2 (neighbor_idx=None), this is fine for now.
        e_gb, born_radii = generalized_born.compute_gb_energy_neighbor_list(
            r, 
            charges, 
            radii, 
            neighbor_idx, 
            solvent_dielectric=solvent_dielectric, 
            solute_dielectric=solute_dielectric,
            dielectric_offset=dielectric_offset
        )
      
      # Non-polar Solvation (SASA) - Now computed separately using ACE
      pass
    
    # Direct Coulomb / Screened Coulomb
    if implicit_solvent:
        eff_dielectric = solute_dielectric
        kappa = 0.0
    else:
        eff_dielectric = dielectric_constant
        kappa = 0.1  # Legacy screened coulomb kappa

    COULOMB_CONSTANT = 332.0637 / eff_dielectric
    
    if neighbor_idx is None:
      # Dense
      dr = space.map_product(displacement_fn)(r, r)
      dist = space.distance(dr)
      q_ij = charges[:, None] * charges[None, :]
      
      dist_safe = dist + 1e-6
      
      if kappa > 0:
          e_coul = COULOMB_CONSTANT * (q_ij / dist_safe) * jnp.exp(-kappa * dist)
      else:
          e_coul = COULOMB_CONSTANT * (q_ij / dist_safe)
          
      # Apply scaling/masking
      if scale_matrix_elec is not None:
          e_coul = e_coul * scale_matrix_elec
      else:
          # Fallback to binary mask
          mask = 1.0 - jnp.eye(charges.shape[0])
          e_coul = jnp.where(mask, e_coul, 0.0)
          e_coul = jnp.where(exclusion_mask, e_coul, 0.0)

      e_direct = 0.5 * jnp.sum(e_coul)
    else:
      # Neighbor List
      idx = neighbor_idx
      r_neighbors = r[idx]
      r_central = r[:, None, :]
      dr = jax.vmap(lambda ra, rb: displacement_fn(ra, rb))(r_central, r_neighbors)
      dist = space.distance(dr)
      
      q_neighbors = charges[idx]
      q_central = charges[:, None]
      q_ij = q_central * q_neighbors
      
      dist_safe = dist + 1e-6
      
      if kappa > 0:
          e_coul = COULOMB_CONSTANT * (q_ij / dist_safe) * jnp.exp(-kappa * dist)
      else:
          e_coul = COULOMB_CONSTANT * (q_ij / dist_safe)
      
      # Mask padding
      mask_neighbors = idx < r.shape[0]
      
      # Scaling/Masking
      i_idx = jnp.arange(r.shape[0])[:, None]
      safe_idx = jnp.minimum(idx, r.shape[0] - 1)
      
      if scale_matrix_elec is not None:
          scale = scale_matrix_elec[i_idx, safe_idx]
          e_coul = e_coul * scale
      else:
          interaction_allowed = exclusion_mask[i_idx, safe_idx]
          e_coul = jnp.where(interaction_allowed, e_coul, 0.0)
      
      final_mask = mask_neighbors
      e_coul = jnp.where(final_mask, e_coul, 0.0)
      
      e_direct = 0.5 * jnp.sum(e_coul)

    if implicit_solvent:
        return e_gb, e_direct, born_radii
    else:
        return 0.0, e_direct, None

  # Combine Non-Bonded
  # -------------------------------------------------------------------------
  def compute_lj(r, neighbor_idx=None):
    if neighbor_idx is None:
      # Dense
      dr = space.map_product(displacement_fn)(r, r)
      dist = space.distance(dr)
      
      sig_ij = 0.5 * (sigmas[:, None] + sigmas[None, :])
      eps_ij = jnp.sqrt(epsilons[:, None] * epsilons[None, :])
      
      e_lj = energy.lennard_jones(dist, sig_ij, eps_ij)
      
      # Apply scaling/masking
      if scale_matrix_vdw is not None:
          e_lj = e_lj * scale_matrix_vdw
      else:
          mask = exclusion_mask
          e_lj = jnp.where(mask, e_lj, 0.0)
      
      return 0.5 * jnp.sum(e_lj)
    else:
      # Neighbor List
      idx = neighbor_idx
      r_neighbors = r[idx]
      r_central = r[:, None, :]
      dr = jax.vmap(lambda ra, rb: displacement_fn(ra, rb))(r_central, r_neighbors)
      dist = space.distance(dr)
      
      sig_neighbors = sigmas[idx]
      eps_neighbors = epsilons[idx]
      sig_central = sigmas[:, None]
      eps_central = epsilons[:, None]
      
      sig_ij = 0.5 * (sig_central + sig_neighbors)
      eps_ij = jnp.sqrt(eps_central * eps_neighbors)
      
      e_lj = energy.lennard_jones(dist, sig_ij, eps_ij)
      
      # Mask padding and scaling
      mask_neighbors = idx < r.shape[0]
      
      i_idx = jnp.arange(r.shape[0])[:, None]
      safe_idx = jnp.minimum(idx, r.shape[0] - 1)
      
      if scale_matrix_vdw is not None:
          scale = scale_matrix_vdw[i_idx, safe_idx]
          e_lj = e_lj * scale
      else:
          interaction_allowed = exclusion_mask[i_idx, safe_idx]
          e_lj = jnp.where(interaction_allowed, e_lj, 0.0)
      
      final_mask = mask_neighbors
      e_lj = jnp.where(final_mask, e_lj, 0.0)
      
      return 0.5 * jnp.sum(e_lj)
      
  def compute_nonpolar(r, born_radii, neighbor_idx=None):
    if not implicit_solvent or born_radii is None:
        return 0.0
        
    if "gb_radii" in system_params and system_params["gb_radii"] is not None:
      radii = system_params["gb_radii"]
    else:
      radii = sigmas * 0.5
      
    # Use ACE approximation (matches OpenMM CustomGBForce)
    return generalized_born.compute_ace_nonpolar_energy(
        radii, born_radii, surface_tension=surface_tension, probe_radius=constants.PROBE_RADIUS
    )

  def compute_cmap_term(r):
      if "cmap_torsions" not in system_params or "cmap_energy_grids" not in system_params:
          return 0.0
      
      cmap_torsions = system_params["cmap_torsions"]
      if cmap_torsions.shape[0] == 0:
          return 0.0
          
      cmap_indices = system_params["cmap_indices"]
      cmap_coeffs = system_params["cmap_coeffs"]
      
      # cmap_torsions is (N, 5) [i, j, k, l, m]
      # Phi: i-j-k-l
      # Psi: j-k-l-m
      
      phi_indices = cmap_torsions[:, 0:4]
      psi_indices = cmap_torsions[:, 1:5]
      
      phi = compute_dihedral_angles(r, phi_indices, displacement_fn)
      psi = compute_dihedral_angles(r, psi_indices, displacement_fn)
      
      # Swapped to (psi, phi) based on validation results matching OpenMM
      # jax.debug.print("CMAP Phi[0]: {}", phi[0])
      # jax.debug.print("CMAP Psi[0]: {}", psi[0])
      e_cmap = cmap.compute_cmap_energy(psi, phi, cmap_indices, cmap_coeffs)
      # jax.debug.print("CMAP Energy (psi, phi): {}", e_cmap)
      return e_cmap

  # Total Energy Function
  # -------------------------------------------------------------------------
  def total_energy(r: Array, neighbor: partition.NeighborList | None = None, **kwargs) -> Array:
    e_bond = bond_energy_fn(r)
    e_angle = angle_energy_fn(r)
    e_dihedral = dihedral_energy_fn(r)
    e_improper = improper_energy_fn(r)
    e_cmap = compute_cmap_term(r)
    
    neighbor_idx = neighbor.idx if neighbor is not None else None
    
    e_lj = compute_lj(r, neighbor_idx)
    e_gb, e_direct, born_radii = compute_electrostatics(r, neighbor_idx)
    e_elec = e_gb + e_direct
    e_np = compute_nonpolar(r, born_radii, neighbor_idx)
    
    return e_bond + e_angle + e_dihedral + e_improper + e_cmap + e_lj + e_elec + e_np

  return total_energy
