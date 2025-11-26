"""System setup and energy function for implicit solvent MD."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax_md import energy, partition, space, util

from prxteinmpnn.physics import bonded
from prxteinmpnn.physics.jax_md_bridge import SystemParams

Array = util.Array


def make_energy_fn(
  displacement_fn: space.DisplacementFn,
  system_params: SystemParams,
  neighbor_list: partition.NeighborList | None = None,
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

  # Lennard-Jones
  # jax_md.energy.lennard_jones expects sigma, epsilon.
  # It computes 4*eps * ((sigma/r)^12 - (sigma/r)^6).
  # We need combination rules.
  # Standard Amber/Charmm: sigma_ij = (sigma_i + sigma_j)/2, eps_ij = sqrt(eps_i * eps_j)
  # jax_md `lennard_jones` takes sigma, epsilon as arrays and broadcasts?
  # No, it usually takes scalar or array of same shape as r.
  # `lennard_jones_neighbor_list` takes sigma, epsilon which can be per-particle?
  # We need to define a custom pair interaction or use `lennard_jones_pair` with mapped parameters.

  # Let's use `energy.lennard_jones_neighbor_list` but we need to pre-combine parameters
  # or use a custom energy function passed to `neighbor_list_energy`.
  # `jax_md` provides `energy.lennard_jones` which works on distances.

  def lj_pair(dr, sigma_i, sigma_j, eps_i, eps_j, **kwargs):
    sigma = 0.5 * (sigma_i + sigma_j)
    epsilon = jnp.sqrt(eps_i * eps_j)
    return energy.lennard_jones(dr, sigma, epsilon)

  # Electrostatics (Screened Coulomb)
  # V = q_i * q_j / (4 * pi * eps0 * r) * exp(-kappa * r)
  # In kcal/mol/A/e^2 units: 332.0637 * q_i * q_j / r
  # We use a constant factor.
  COULOMB_CONSTANT = 332.0637
  KAPPA = 0.1  # Inverse Debye length (1/A), approx 0.1M salt

  def coulomb_pair(dr, q_i, q_j, **kwargs):
    interaction = (q_i * q_j) / (dr + 1e-6)
    screening = jnp.exp(-KAPPA * dr)
    return COULOMB_CONSTANT * interaction * screening

  # Combine Non-Bonded
  def non_bonded_pair(dr, p_i, p_j, **kwargs):
    # p_i contains [q, sigma, epsilon]
    q_i, sig_i, eps_i = p_i
    q_j, sig_j, eps_j = p_j

    e_lj = lj_pair(dr, sig_i, sig_j, eps_i, eps_j)
    e_elec = coulomb_pair(dr, q_i, q_j)
    return e_lj + e_elec

  # We stack parameters for easy mapping
  # (N, 3)
  particle_params = jnp.stack([charges, sigmas, epsilons], axis=-1)

  # Create Neighbor List Energy
  # We define a custom function that uses the neighbor list
  def total_energy(r: Array, neighbor: partition.NeighborList | None = None, **kwargs) -> Array:
    e_bond = bond_energy_fn(r)
    e_angle = angle_energy_fn(r)

    # Non-bonded
    if neighbor is None:
      # Fallback to N^2 dense
      # Compute all pairs
      dr = space.map_product(displacement_fn)(r, r)
      dist = space.distance(dr)
      
      # Compute energy for all pairs
      # We vmap over i and j
      # non_bonded_pair expects scalars, so we vmap twice
      
      # Efficient way:
      # Pre-calculate sigma_ij, eps_ij, q_ij matrices
      # But let's stick to the pair function for clarity if JIT handles it.
      
      # Vectorized computation
      # Sigma: (N, 1) + (1, N)
      sig_ij = 0.5 * (sigmas[:, None] + sigmas[None, :])
      eps_ij = jnp.sqrt(epsilons[:, None] * epsilons[None, :])
      q_ij = charges[:, None] * charges[None, :]
      
      e_lj = energy.lennard_jones(dist, sig_ij, eps_ij)
      e_elec = COULOMB_CONSTANT * (q_ij / (dist + 1e-6)) * jnp.exp(-KAPPA * dist)
      
      e_nb = e_lj + e_elec
      
      # Apply mask
      # exclusion_mask is (N, N)
      e_nb = jnp.where(exclusion_mask, e_nb, 0.0)



      
      # Sum and divide by 2 (double counting)
      return e_bond + e_angle + 0.5 * jnp.sum(e_nb)

    else:
      # Neighbor List
      # neighbor.idx is (N, max_neighbors)
      # We gather neighbors
      
      idx = neighbor.idx
      
      # Gather positions
      r_neighbors = r[idx] # (N, max_neighbors, 3)
      r_central = r[:, None, :] # (N, 1, 3)
      
      dr = jax.vmap(lambda ra, rb: displacement_fn(ra, rb))(r_central, r_neighbors)
      dist = space.distance(dr)
      
      # Gather params
      sig_neighbors = sigmas[idx]
      eps_neighbors = epsilons[idx]
      q_neighbors = charges[idx]
      
      sig_central = sigmas[:, None]
      eps_central = epsilons[:, None]
      q_central = charges[:, None]
      
      # Compute
      sig_ij = 0.5 * (sig_central + sig_neighbors)
      eps_ij = jnp.sqrt(eps_central * eps_neighbors)
      q_ij = q_central * q_neighbors
      
      e_lj = energy.lennard_jones(dist, sig_ij, eps_ij)
      e_elec = COULOMB_CONSTANT * (q_ij / (dist + 1e-6)) * jnp.exp(-KAPPA * dist)
      
      e_nb = e_lj + e_elec
      
      # Apply Mask
      # We need to look up the exclusion mask for (i, j)
      # i is range(N), j is idx
      # mask_val = exclusion_mask[i, idx]
      # We can use vmap or advanced indexing
      
      # i indices: (N, 1) broadcasted
      i_idx = jnp.arange(r.shape[0])[:, None]
      
      # Mask lookup
      # Note: idx contains N (padding) which is out of bounds for mask if mask is (N, N)
      # But jax handles OOB by clamping or error?
      # jax_md neighbor lists pad with N.
      # We should ensure exclusion_mask is padded or handle N.
      # Actually, `idx` values < N are valid. `idx` == N is padding.
      # We can mask where idx < N.
      
      mask_neighbors = idx < r.shape[0]
      
      # Look up exclusion
      # We need to be careful with OOB.
      # Safe lookup: clamp idx to 0..N-1, then apply mask_neighbors
      safe_idx = jnp.minimum(idx, r.shape[0] - 1)
      interaction_allowed = exclusion_mask[i_idx, safe_idx]
      
      # Combine masks
      final_mask = mask_neighbors & interaction_allowed
      
      e_nb = jnp.where(final_mask, e_nb, 0.0)
      
      return e_bond + e_angle + 0.5 * jnp.sum(e_nb)

  return total_energy
