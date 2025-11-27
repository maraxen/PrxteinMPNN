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

    The Born radius $B_i$ is calculated as:

    $$
    B_i^{-1} = \\rho_i^{-1} - \\rho_i^{-1} \\tanh(\\alpha \\Psi_i - \\beta \\Psi_i^2 + \\gamma \\Psi_i^3)
    $$

    where $\\Psi_i = \\rho_i I_i$, and $I_i$ is the pairwise descreening integral.

    Args:
        positions: Atom positions (N, 3).
        radii: Intrinsic atomic radii (N,).
        dielectric_offset: Offset for Born radius calculation (default 0.09 A).
        probe_radius: Solvent probe radius (default 1.4 A).

    Returns:
        Born radii (N,).
    """
    delta_positions = positions[:, None, :] - positions[None, :, :] # (N, N, 3)
    distances = safe_norm(delta_positions, axis=-1) # (N, N)

    # Add large value to diagonal to avoid self-interaction singularities
    distances_safe = distances + jnp.eye(distances.shape[0]) * 10.0

    offset_radii = radii - dielectric_offset
    radii_j = radii

    radii_i_broadcast = offset_radii[:, None] # (N, 1)
    radii_j_broadcast = radii_j[None, :]      # (1, N)
    
    mask = 1.0 - jnp.eye(distances.shape[0]) # Mask for i != j
    
    pair_integrals = compute_pair_integral(distances_safe, radii_i_broadcast, radii_j_broadcast) # (N, N)
    pair_integrals = jnp.where(mask, pair_integrals, 0.0)
    
    born_radius_inverse_term = jnp.sum(pair_integrals, axis=1) # (N,)
    
    scaled_integral = offset_radii * born_radius_inverse_term
    tanh_argument = ALPHA_OBC * scaled_integral - BETA_OBC * scaled_integral**2 + GAMMA_OBC * scaled_integral**3
    inv_born_radii = (1.0 / offset_radii) * (1.0 - jnp.tanh(tanh_argument))
    
    return 1.0 / inv_born_radii


def f_gb(distance: Array, born_radii_i: Array, born_radii_j: Array) -> Array:
    """Computes the GB effective distance function $f_{GB}(r_{ij})$.

    $$
    f_{GB}(r_{ij}) = \\sqrt{r_{ij}^2 + B_i B_j \\exp\\left(-\\frac{r_{ij}^2}{4 B_i B_j}\\right)}
    $$

    Args:
        distance: Pairwise distance (scalar or array).
        born_radii_i: Born radius of atom i ($B_i$).
        born_radii_j: Born radius of atom j ($B_j$).

    Returns:
        Effective GB distance.
    """
    radii_product = born_radii_i * born_radii_j
    exp_term = jnp.exp(- (distance**2) / (4.0 * radii_product))
    return jnp.sqrt(distance**2 + radii_product * exp_term)


def compute_gb_energy(
    positions: Array,
    charges: Array,
    radii: Array,
    solvent_dielectric: float = constants.DIELECTRIC_WATER,
    solute_dielectric: float = constants.DIELECTRIC_PROTEIN,
    dielectric_offset: float = 0.09,
) -> Array:
    """Computes the Generalized Born solvation energy.

    $$
    \\Delta G_{pol} = -\\frac{1}{2} \\left(\\frac{1}{\\epsilon_{in}} - \\frac{1}{\\epsilon_{out}}\\right) \\sum_{ij} \\frac{q_i q_j}{f_{GB}(r_{ij})}
    $$

    Includes self-solvation energy terms ($i=j$).

    Args:
        positions: Atom positions (N, 3).
        charges: Atom charges (N,).
        radii: Atom radii (N,).
        solvent_dielectric: Solvent dielectric constant ($\\epsilon_{out}$).
        solute_dielectric: Solute dielectric constant ($\\epsilon_{in}$).
        dielectric_offset: Offset for Born radius calculation.

    Returns:
        Total GB energy (scalar).
    """
    born_radii = compute_born_radii(positions, radii, dielectric_offset=dielectric_offset)
    
    delta_positions = positions[:, None, :] - positions[None, :, :] # (N, N, 3)
    distances = safe_norm(delta_positions, axis=-1) # (N, N)
    
    born_radii_i = born_radii[:, None] # (N, 1)
    born_radii_j = born_radii[None, :] # (1, N)
    
    effective_distances = f_gb(distances, born_radii_i, born_radii_j)
    
    tau = (1.0 / solute_dielectric) - (1.0 / solvent_dielectric)
    prefactor = -0.5 * constants.COULOMB_CONSTANT * tau
    
    charge_products = charges[:, None] * charges[None, :] # (N, N)
    energy_terms = charge_products / effective_distances
    
    total_energy = prefactor * jnp.sum(energy_terms)
    
    return total_energy


def compute_pair_integral(distance: Array, radius_i: Array, radius_j: Array) -> Array:
    """Computes the pair integral term $H_{ij}$ for OBC.
    
    This integral represents the volume of atom j that overlaps with the descreening region of atom i.

    $$
    H_{ij} = \\frac{1}{2} \\left[ \\frac{1}{L_{ij}} - \\frac{1}{U_{ij}} \\right] + \\frac{1}{4r_{ij}} \\left[ \\frac{1}{U_{ij}^2} - \\frac{1}{L_{ij}^2} \\right] + \\frac{1}{2r_{ij}} \\ln \\frac{L_{ij}}{U_{ij}}
    $$

    where $L_{ij} = \\max(\\rho_i, |r_{ij} - \\rho_j|)$ and $U_{ij} = r_{ij} + \\rho_j$.
    
    Args:
        distance: Distance between atoms i and j ($r_{ij}$).
        radius_i: Radius of atom i ($\\rho_i$, usually offset radius).
        radius_j: Radius of atom j ($\\rho_j$, usually vdW radius).
        
    Returns:
        The integral value $H_{ij}$.
    """
    lower_limit = jnp.maximum(radius_i, jnp.abs(distance - radius_j))
    upper_limit = distance + radius_j
    
    inv_lower = 1.0 / lower_limit
    inv_upper = 1.0 / upper_limit
    
    term1 = 0.5 * (inv_lower - inv_upper)
    term2 = 0.25 * distance * (inv_upper**2 - inv_lower**2)
    
    distance_safe = jnp.maximum(distance, 1e-6)
    term3 = (0.5 / distance_safe) * jnp.log(lower_limit / upper_limit)
    
    return term1 + term2 + term3


def compute_born_radii_neighbor_list(
    positions: Array,
    radii: Array,
    neighbor_idx: Array,
    dielectric_offset: float = 0.09,
) -> Array:
    """Computes effective Born radii using neighbor lists.
    
    This version uses a neighbor list to compute interactions only within a cutoff,
    which approximates the full $N^2$ calculation.

    The Born radius $B_i$ is calculated as:

    $$
    B_i^{-1} = \\rho_i^{-1} - \\rho_i^{-1} \\tanh(\\alpha \\Psi_i - \\beta \\Psi_i^2 + \\gamma \\Psi_i^3)
    $$

    where $\\Psi_i = \\rho_i I_i$, and $I_i$ is the pairwise descreening integral summed over neighbors.
    
    Args:
        positions: Atom positions (N, 3).
        radii: Intrinsic atomic radii (N,).
        neighbor_idx: Neighbor list indices (N, K).
        dielectric_offset: Offset for Born radius calculation.
        
    Returns:
        Born radii (N,).
    """
    N, K = neighbor_idx.shape
    
    neighbor_positions = positions[neighbor_idx] # (N, K, 3)
    central_positions = positions[:, None, :]    # (N, 1, 3)
    
    delta_positions = central_positions - neighbor_positions # (N, K, 3)
    distances = safe_norm(delta_positions, axis=-1) # (N, K)
    
    mask_neighbors = neighbor_idx < N # Mask padding
    
    offset_radii = radii - dielectric_offset
    radii_j = radii[neighbor_idx] # (N, K)
    
    radii_i_broadcast = offset_radii[:, None] # (N, 1)
    
    pair_integrals = compute_pair_integral(distances, radii_i_broadcast, radii_j)
    pair_integrals = jnp.where(mask_neighbors, pair_integrals, 0.0)
    
    born_radius_inverse_term = jnp.sum(pair_integrals, axis=1)
    
    scaled_integral = offset_radii * born_radius_inverse_term
    tanh_argument = ALPHA_OBC * scaled_integral - BETA_OBC * scaled_integral**2 + GAMMA_OBC * scaled_integral**3
    inv_born_radii = (1.0 / offset_radii) * (1.0 - jnp.tanh(tanh_argument))
    
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
    """Computes GB energy using neighbor lists.
    
    Calculates the Generalized Born energy using a neighbor list for pairwise interactions.

    $$
    \\Delta G_{pol} = -\\frac{1}{2} \\left(\\frac{1}{\\epsilon_{in}} - \\frac{1}{\\epsilon_{out}}\\right) \\sum_{ij} \\frac{q_i q_j}{f_{GB}(r_{ij})}
    $$
    
    Args:
        positions: Atom positions (N, 3).
        charges: Atom charges (N,).
        radii: Atom radii (N,).
        neighbor_idx: Neighbor list indices (N, K).
        solvent_dielectric: Solvent dielectric constant ($\\epsilon_{out}$).
        solute_dielectric: Solute dielectric constant ($\\epsilon_{in}$).
        dielectric_offset: Offset for Born radius calculation.
        
    Returns:
        Total GB energy (scalar).
    """
    born_radii = compute_born_radii_neighbor_list(
        positions, radii, neighbor_idx, dielectric_offset
    )
    
    neighbor_positions = positions[neighbor_idx] # (N, K, 3)
    central_positions = positions[:, None, :]    # (N, 1, 3)
    delta_positions = central_positions - neighbor_positions # (N, K, 3)
    distances = safe_norm(delta_positions, axis=-1)          # (N, K)
    
    born_radii_i = born_radii[:, None]       # (N, 1)
    born_radii_j = born_radii[neighbor_idx]  # (N, K)
    
    effective_distances = f_gb(distances, born_radii_i, born_radii_j)
    
    tau = (1.0 / solute_dielectric) - (1.0 / solvent_dielectric)
    prefactor = -0.5 * constants.COULOMB_CONSTANT * tau
    
    charges_i = charges[:, None]        # (N, 1)
    charges_j = charges[neighbor_idx]   # (N, K)
    charge_products = charges_i * charges_j
    
    energy_terms = charge_products / effective_distances
    
    N = positions.shape[0]
    mask_neighbors = neighbor_idx < N
    energy_terms = jnp.where(mask_neighbors, energy_terms, 0.0)
    
    term_neighbors = jnp.sum(energy_terms)
    term_self = jnp.sum((charges**2) / born_radii)
    
    total_energy = prefactor * (term_neighbors + term_self)
    
    return total_energy
