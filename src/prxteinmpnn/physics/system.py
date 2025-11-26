"""System setup and energy function for implicit solvent MD."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax_md import energy, partition, space, util

from prxteinmpnn.physics import bonded, generalized_born
from prxteinmpnn.physics.jax_md_bridge import SystemParams

Array = util.Array


def make_energy_fn(
  displacement_fn: space.DisplacementFn,
  system_params: SystemParams,
  neighbor_list: partition.NeighborList | None = None,
  dielectric_constant: float = 1.0,
  implicit_solvent: bool = True,
  solvent_dielectric: float = 78.5,
  solute_dielectric: float = 1.0,
) -> Callable[[Array], Array]:
  """Creates the total potential energy function.

  U(R) = U_bond + U_angle + U_vdw + U_elec

  Args:
      displacement_fn: JAX MD displacement function.
      system_params: System parameters from `jax_md_bridge`.
      neighbor_list: Optional neighbor list. If provided, non-bonded terms
                     will use it. If None, they will be N^2 (slow).
                     NOTE: For proteins, N^2 is often acceptable for small systems,
                     but neighbor lists are better for >500 atoms.
                     We strongly recommend using neighbor lists.

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

  # 2. Non-Bonded Terms
  # We need to handle exclusions.
  # The energy functions below need to be wrapped to apply the mask.
  charges = system_params["charges"]
  sigmas = system_params["sigmas"]
  epsilons = system_params["epsilons"]
  exclusion_mask = system_params["exclusion_mask"]

  # Let's use `energy.lennard_jones_neighbor_list` but we need to pre-combine parameters
  # or use a custom energy function passed to `neighbor_list_energy`.
  # `jax_md` provides `energy.lennard_jones` which works on distances.

  def lj_pair(dr, sigma_i, sigma_j, eps_i, eps_j, **kwargs):
    sigma = 0.5 * (sigma_i + sigma_j)
    epsilon = jnp.sqrt(eps_i * eps_j)
    return energy.lennard_jones(dr, sigma, epsilon)

  # Electrostatics
  # -------------------------------------------------------------------------
  # We support two modes:
  # 1. Implicit Solvent (Generalized Born) - Default
  # 2. Vacuum + Screened Coulomb (Legacy)

  def compute_electrostatics(r, neighbor_idx=None):
    # Prepare parameters
    # Intrinsic radii for GB: R ~ sigma / 2
    radii = sigmas * 0.5

    if implicit_solvent:
      # Generalized Born (OBC)
      if neighbor_idx is None:
        # Dense O(N^2)
        return generalized_born.compute_gb_energy(
          r, charges, radii,
          solvent_dielectric=solvent_dielectric,
          solute_dielectric=solute_dielectric
        )
      else:
        # Neighbor List
        return generalized_born.compute_gb_energy_neighbor_list(
          r, charges, radii, neighbor_idx,
          solvent_dielectric=solvent_dielectric,
          solute_dielectric=solute_dielectric
        )
    else:
      # Screened Coulomb (Legacy)
      # V = q_i * q_j / (eps * r) * exp(-kappa * r)
      # We use the provided dielectric_constant (usually 1.0 or 80.0)
      
      # Constants
      COULOMB_CONSTANT = 332.0637 / dielectric_constant
      KAPPA = 0.1  # Inverse Debye length
      
      if neighbor_idx is None:
        # Dense
        dr = space.map_product(displacement_fn)(r, r)
        dist = space.distance(dr)
        q_ij = charges[:, None] * charges[None, :]
        e_elec = COULOMB_CONSTANT * (q_ij / (dist + 1e-6)) * jnp.exp(-KAPPA * dist)
        # Mask self
        mask = 1.0 - jnp.eye(charges.shape[0])
        e_elec = jnp.where(mask, e_elec, 0.0)
        
        # Apply exclusion mask
        e_elec = jnp.where(exclusion_mask, e_elec, 0.0)

        return 0.5 * jnp.sum(e_elec)
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
        
        e_elec = COULOMB_CONSTANT * (q_ij / (dist + 1e-6)) * jnp.exp(-KAPPA * dist)
        
        # Mask padding
        mask_neighbors = idx < r.shape[0]
        
        # Exclusion lookup
        i_idx = jnp.arange(r.shape[0])[:, None]
        safe_idx = jnp.minimum(idx, r.shape[0] - 1)
        interaction_allowed = exclusion_mask[i_idx, safe_idx]
        
        final_mask = mask_neighbors & interaction_allowed
        e_elec = jnp.where(final_mask, e_elec, 0.0)
        
        return 0.5 * jnp.sum(e_elec)

  # Combine Non-Bonded
  # -------------------------------------------------------------------------
  
  # We separate LJ and Electrostatics because GB is global (or different structure).
  # LJ is always pairwise (dense or neighbor).
  
  def compute_lj(r, neighbor_idx=None):
    if neighbor_idx is None:
      # Dense
      dr = space.map_product(displacement_fn)(r, r)
      dist = space.distance(dr)
      
      sig_ij = 0.5 * (sigmas[:, None] + sigmas[None, :])
      eps_ij = jnp.sqrt(epsilons[:, None] * epsilons[None, :])
      
      e_lj = energy.lennard_jones(dist, sig_ij, eps_ij)
      
      # Mask self and exclusions
      # Exclusion mask is (N, N)
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
      
      # Mask padding and exclusions
      mask_neighbors = idx < r.shape[0]
      
      # Exclusion lookup
      i_idx = jnp.arange(r.shape[0])[:, None]
      safe_idx = jnp.minimum(idx, r.shape[0] - 1)
      interaction_allowed = exclusion_mask[i_idx, safe_idx]
      
      final_mask = mask_neighbors & interaction_allowed
      e_lj = jnp.where(final_mask, e_lj, 0.0)
      
      return 0.5 * jnp.sum(e_lj)

  # Total Energy Function
  # -------------------------------------------------------------------------
  def total_energy(r: Array, neighbor: partition.NeighborList | None = None, **kwargs) -> Array:
    e_bond = bond_energy_fn(r)
    e_angle = angle_energy_fn(r)
    
    neighbor_idx = neighbor.idx if neighbor is not None else None
    
    e_lj = compute_lj(r, neighbor_idx)
    e_elec = compute_electrostatics(r, neighbor_idx)
    
    # Note: GB energy already includes 0.5 factor and self-energy.
    # Screened Coulomb includes 0.5 factor.
    # LJ includes 0.5 factor.
    
    return e_bond + e_angle + e_lj + e_elec

  return total_energy
