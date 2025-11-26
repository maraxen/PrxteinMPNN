"""Generalized Born implicit solvent model (GBSA) implementation.

References:
    Onufriev, Bashford, Case, "Exploring native states and large-scale dynamics with the generalized born model",
    Proteins 55, 383-394 (2004). (OBC Model II)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax_md import util

from prxteinmpnn.physics import constants

Array = util.Array

# OBC Type II Parameters
ALPHA_OBC = 1.0
BETA_OBC = 0.8
GAMMA_OBC = 4.85


def safe_norm(x: Array, axis: int = -1, eps: float = 1e-12) -> Array:
    """Computes norm safely to avoid NaN gradients at zero."""
    return jnp.sqrt(jnp.sum(x**2, axis=axis) + eps)


def compute_born_radii(
    positions: Array,
    radii: Array,
    dielectric_offset: float = 0.09,
    probe_radius: float = constants.PROBE_RADIUS,
) -> Array:
    """Computes effective Born radii using the OBC II approximation.

    Args:
        positions: Atom positions (N, 3).
        radii: Intrinsic atomic radii (N,).
        dielectric_offset: Offset for Born radius calculation (default 0.09 A).
        probe_radius: Solvent probe radius (default 1.4 A).

    Returns:
        Born radii (N,).
    """
    # 1. Calculate pairwise distances
    # (N, N, 3)
    dr = positions[:, None, :] - positions[None, :, :]
    # (N, N)
    r_ij = safe_norm(dr, axis=-1)
    
    # Avoid division by zero and self-interaction in distance terms
    # We add a large value to diagonal to make self-interaction terms vanish/stable
    # during the integral calculation (since we mask them out anyway).
    # If we use a small epsilon, the 1/r term in the integral can blow up gradients.
    r_ij_safe = r_ij + jnp.eye(r_ij.shape[0]) * 10.0

    # 2. Compute Pair Integrals (H_ij)
    # We integrate the volume of atom j over the space outside atom i.
    # Standard GBSA approximations (HCT/OBC) use an analytical overlap formula.
    
    # Offset radii
    rho_i = radii - dielectric_offset
    # Scaled radii for neighbor j (usually just radii, sometimes scaled)
    # In standard OBC, we integrate over the vdW sphere of j.
    # Some implementations scale radii by a factor (e.g. 1.0).
    r_j = radii
    
    # We compute the term H_ij for each pair.
    # H_ij is the contribution of atom j to the reduction of Born radius of atom i.
    
    # Vectorized computation of H_ij
    # r: distance r_ij
    # R_i: rho_i (radius of atom i)
    # R_j: r_j (radius of atom j)
    
    # Broadcast for (N, N) matrix
    R_i = rho_i[:, None]
    R_j = r_j[None, :]
    dist = r_ij_safe

    # We only sum over j != i.
    # Mask for i != j
    mask = 1.0 - jnp.eye(dist.shape[0])
    
    # Apply to all pairs
    # (N, N)
    H_ij = compute_pair_integral(dist, R_i, R_j)
    
    # Mask self-interactions (i=j)
    # H_ii should be 0.
    H_ij = jnp.where(mask, H_ij, 0.0)
    
    # Also, if r > cutoff, H_ij ~ 0.
    # We don't strictly need a cutoff for correctness, but for speed in dense, we just sum.
    
    # 3. Sum integrals to get effective inverse radius
    # I_i = sum_j H_ij
    I_i = jnp.sum(H_ij, axis=1)
    
    # 4. Calculate Born Radius (alpha_i) using OBC approximation
    # B_i^-1 = rho_i^-1 - I_i
    # But OBC adds a correction factor.
    # R_inv = 1/rho_i - 1/R_i * tanh(...)
    
    # OBC Formula:
    # B_inv = 1/rho_i - 1/rho_i * tanh(alpha*y - beta*y^2 + gamma*y^3)
    # where y = rho_i * I_i
    
    y = rho_i * I_i
    
    # OBC II coefficients
    # alpha=1.0, beta=0.8, gamma=4.85
    term = ALPHA_OBC * y - BETA_OBC * y**2 + GAMMA_OBC * y**3
    
    inv_born_radii = (1.0 / rho_i) * (1.0 - jnp.tanh(term))
    
    born_radii = 1.0 / inv_born_radii
    
    return born_radii


def f_gb(r: Array, alpha_i: Array, alpha_j: Array) -> Array:
    """Computes the GB effective distance function f_GB(r_ij).

    f_GB = sqrt(r^2 + alpha_i * alpha_j * exp(-r^2 / (4 * alpha_i * alpha_j)))

    Args:
        r: Pairwise distance (scalar or array).
        alpha_i: Born radius of atom i.
        alpha_j: Born radius of atom j.

    Returns:
        Effective GB distance.
    """
    a_prod = alpha_i * alpha_j
    exp_term = jnp.exp(- (r**2) / (4.0 * a_prod))
    return jnp.sqrt(r**2 + a_prod * exp_term)


def compute_gb_energy(
    positions: Array,
    charges: Array,
    radii: Array,
    solvent_dielectric: float = constants.DIELECTRIC_WATER,
    solute_dielectric: float = constants.DIELECTRIC_PROTEIN,
    dielectric_offset: float = 0.09,
) -> Array:
    """Computes the Generalized Born solvation energy.

    E_GB = -0.5 * (1/eps_in - 1/eps_out) * sum_ij (q_i * q_j / f_GB(r_ij))

    Args:
        positions: Atom positions (N, 3).
        charges: Atom charges (N,).
        radii: Atom radii (N,).
        solvent_dielectric: Solvent dielectric constant.
        solute_dielectric: Solute dielectric constant.
        dielectric_offset: Offset for Born radius calculation.

    Returns:
        Total GB energy (scalar).
    """
    # 1. Compute Born Radii
    alpha = compute_born_radii(positions, radii, dielectric_offset=dielectric_offset)
    
    # 2. Compute Pairwise Terms
    # (N, N)
    dr = positions[:, None, :] - positions[None, :, :]
    r_ij = safe_norm(dr, axis=-1)
    
    # Broadcast alphas
    # (N, N)
    alpha_i = alpha[:, None]
    alpha_j = alpha[None, :]
    
    # Compute f_GB
    f_gb_ij = f_gb(r_ij, alpha_i, alpha_j)
    
    # 3. Compute Energy
    # Pre-factor
    # constant * (1/eps_in - 1/eps_out)
    # Note: COULOMB_CONSTANT is in kcal/mol/A/e^2
    # The formula is usually:
    # E = -0.5 * C * (1/eps_in - 1/eps_out) * sum ...
    
    tau = (1.0 / solute_dielectric) - (1.0 / solvent_dielectric)
    prefactor = -0.5 * constants.COULOMB_CONSTANT * tau
    
    # Charge product
    q_ij = charges[:, None] * charges[None, :]
    
    # Term matrix
    E_ij = q_ij / f_gb_ij
    
    # Sum
    # We sum all pairs including i=j.
    # For i=j, r=0, f_gb = alpha_i.
    # This represents the self-solvation energy (Born energy of single ion).
    # E_self = -0.5 * (1/eps_in - 1/eps_out) * q^2 / alpha
    
    total_energy = prefactor * jnp.sum(E_ij)
    
    return total_energy


def compute_pair_integral(r: Array, r_i: Array, r_j: Array) -> Array:
    """Computes the pair integral term H_ij for OBC."""
    L = jnp.maximum(r_i, jnp.abs(r - r_j))
    U = r + r_j
    
    inv_L = 1.0 / L
    inv_U = 1.0 / U
    
    term1 = 0.5 * (inv_L - inv_U)
    term2 = 0.25 * r * (inv_U**2 - inv_L**2)
    
    r_safe = jnp.maximum(r, 1e-6)
    term3 = (0.5 / r_safe) * jnp.log(L / U)
    
    return term1 + term2 + term3


def compute_born_radii_neighbor_list(
    positions: Array,
    radii: Array,
    neighbor_idx: Array,
    dielectric_offset: float = 0.09,
) -> Array:
    """Computes effective Born radii using neighbor lists."""
    # neighbor_idx: (N, K)
    N, K = neighbor_idx.shape
    
    # Gather positions
    # (N, K, 3)
    r_neighbors = positions[neighbor_idx]
    r_central = positions[:, None, :]
    
    # Distances
    dr = r_central - r_neighbors
    r_ij = safe_norm(dr, axis=-1)
    
    # Mask padding (idx == N)
    # neighbor_idx values are in [0, N]. N is padding.
    mask_neighbors = neighbor_idx < N
    
    # Also mask self-interaction if present in neighbor list (usually not, but safe to check)
    # But neighbor lists usually exclude self unless explicitly requested.
    # Assuming standard jax_md neighbor lists which might include self if dr=0?
    # Usually they don't include self.
    
    # Radii
    rho_i = radii - dielectric_offset
    r_j = radii[neighbor_idx] # (N, K)
    
    R_i = rho_i[:, None]
    R_j = r_j
    
    # Compute integrals
    H_ij = compute_pair_integral(r_ij, R_i, R_j)
    
    # Apply mask
    H_ij = jnp.where(mask_neighbors, H_ij, 0.0)
    
    # Sum
    I_i = jnp.sum(H_ij, axis=1)
    
    # OBC Formula
    y = rho_i * I_i
    term = ALPHA_OBC * y - BETA_OBC * y**2 + GAMMA_OBC * y**3
    inv_born_radii = (1.0 / rho_i) * (1.0 - jnp.tanh(term))
    
    return 1.0 / inv_born_radii


def compute_gb_energy_neighbor_list(
    positions: Array,
    charges: Array,
    radii: Array,
    neighbor_idx: Array,
    solvent_dielectric: float = constants.DIELECTRIC_WATER,
    solute_dielectric: float = constants.DIELECTRIC_PROTEIN,
    dielectric_offset: float = 0.09,
) -> Array:
    """Computes GB energy using neighbor lists."""
    # 1. Born Radii
    alpha = compute_born_radii_neighbor_list(
        positions, radii, neighbor_idx, dielectric_offset
    )
    
    # 2. Pairwise Terms
    # We iterate over neighbors again
    r_neighbors = positions[neighbor_idx]
    r_central = positions[:, None, :]
    dr = r_central - r_neighbors
    r_ij = safe_norm(dr, axis=-1)
    
    alpha_i = alpha[:, None]
    alpha_j = alpha[neighbor_idx]
    
    f_gb_ij = f_gb(r_ij, alpha_i, alpha_j)
    
    # 3. Energy
    tau = (1.0 / solute_dielectric) - (1.0 / solvent_dielectric)
    prefactor = -0.5 * constants.COULOMB_CONSTANT * tau
    
    q_i = charges[:, None]
    q_j = charges[neighbor_idx]
    q_ij = q_i * q_j
    
    E_ij = q_ij / f_gb_ij
    
    # Mask
    N = positions.shape[0]
    mask_neighbors = neighbor_idx < N
    E_ij = jnp.where(mask_neighbors, E_ij, 0.0)
    
    # Sum over neighbors
    # So we just sum E_ij (neighbors) + E_ii (self).
    # And multiply by prefactor.
    
    term_neighbors = jnp.sum(E_ij)
    term_self = jnp.sum((charges**2) / alpha)
    
    total_energy = prefactor * (term_neighbors + term_self)
    
    return total_energy
