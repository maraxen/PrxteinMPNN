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
ALPHA_OBC = 1.0 # Corrected from 1.2 to match Onufriev 2004 & OpenMM
BETA_OBC = 0.8
GAMMA_OBC = 4.85

# Non-polar Solvation Parameters
SURFACE_TENSION = 0.005  # kcal/mol/A^2


def safe_norm(x: Array, axis: int = -1, eps: float = 1e-12) -> Array:
    """Computes norm safely to avoid NaN gradients at zero."""
    return jnp.sqrt(jnp.sum(x**2, axis=axis) + eps)


def compute_born_radii(
    positions: Array,
    radii: Array,
    dielectric_offset: float = 0.09,
    probe_radius: float = constants.PROBE_RADIUS,
    mask: Array | None = None,
    scaled_radii: Array | None = None,
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
        mask: Optional mask (N, N) where 0.0 indicates exclusion.

    Returns:
        Born radii (N,).
    """
    delta_positions = positions[:, None, :] - positions[None, :, :] # (N, N, 3)
    distances = safe_norm(delta_positions, axis=-1) # (N, N)

    # Add large value to diagonal to avoid self-interaction singularities
    distances_safe = distances + jnp.eye(distances.shape[0]) * 10.0

    offset_radii = radii - dielectric_offset
    
    if scaled_radii is None:
        radii_j = radii
    else:
        radii_j = scaled_radii

    radii_i_broadcast = offset_radii[:, None] # (N, 1)
    radii_j_broadcast = radii_j[None, :]      # (1, N)
    
    # Default mask excludes self
    if mask is None:
        mask = 1.0 - jnp.eye(distances.shape[0])
    else:
        # Ensure self is excluded even if mask allows it (though mask usually handles it)
        mask = mask * (1.0 - jnp.eye(distances.shape[0]))
    
    pair_integrals = compute_pair_integral(distances_safe, radii_i_broadcast, radii_j_broadcast) # (N, N)
    
    if mask is not None:
        pair_integrals = pair_integrals * mask
    else:
        pass
        # Default excludes self (handled by mask usually, but if mask is None, we assume all pairs included except self)
        # Actually, compute_born_radii logic above:
        # if mask is None: mask = 1.0 - eye
        # else: mask = mask * (1.0 - eye)
        # So mask is NEVER None here because of lines 77-81.
        pass
        
    # pair_integrals = jnp.where(mask > 0.9, pair_integrals, 0.0)
    
    born_radius_inverse_term = jnp.sum(pair_integrals, axis=1) # (N,)
    
    scaled_integral = offset_radii * born_radius_inverse_term
    tanh_argument = ALPHA_OBC * scaled_integral - BETA_OBC * scaled_integral**2 + GAMMA_OBC * scaled_integral**3
    # OpenMM Formula: B = 1 / (1/or - tanh(psi - ...)/radius)
    # inv_B = 1/or - tanh(...)/radius
    # Note: OpenMM uses full radius in the tanh term denominator, and no 0.99 factor.
    inv_born_radii = 1.0 / offset_radii - jnp.tanh(tanh_argument) / radii
    
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
    mask: Array | None = None,
    energy_mask: Array | None = None,
    scaled_radii: Array | None = None,
) -> tuple[Array, Array]:
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
        mask: Optional mask (N, N) for Born radii calculation.
        energy_mask: Optional mask (N, N) for Energy summation.
        scaled_radii: Optional scaled radii for Born radii calculation.

    Returns:
        Tuple of (Total GB energy, Born Radii).
    """
    # DEBUG: Check if mask is being used
    # if mask is not None:
    #     pass
    born_radii = compute_born_radii(positions, radii, dielectric_offset=dielectric_offset, mask=mask, scaled_radii=scaled_radii)
    
    # DEBUG: Print Born Radii Stats
    # print(f"DEBUG: Born Radii Min={jnp.min(born_radii)}, Max={jnp.max(born_radii)}, Mean={jnp.mean(born_radii)}")
    
    delta_positions = positions[:, None, :] - positions[None, :, :] # (N, N, 3)
    distances = safe_norm(delta_positions, axis=-1) # (N, N)
    
    born_radii_i = born_radii[:, None] # (N, 1)
    born_radii_j = born_radii[None, :] # (1, N)
    
    effective_distances = f_gb(distances, born_radii_i, born_radii_j)
    
    tau = (1.0 / solute_dielectric) - (1.0 / solvent_dielectric)
    prefactor = -0.5 * constants.COULOMB_CONSTANT * tau
    
    charge_products = charges[:, None] * charges[None, :] # (N, N)
    energy_terms = charge_products / effective_distances
    
    if energy_mask is not None:
        # Mask excludes 1-2, 1-3.
        # But we need to KEEP self-interactions (i=i) for self-solvation.
        # The mask usually has 0.0 on diagonal if it's exclusion_mask from system.py?
        # In system.py, exclusion_mask is False (0) for excluded pairs.
        # And usually True (1) for self? No, self is usually excluded in VDW/Elec.
        # But GBSA needs self.
        
        # Let's assume mask is 0 for excluded pairs.
        # We need to ensure diagonal is 1 for the energy sum.
        # But if energy_mask is scaling matrix, we should NOT binarize it.
        # We assume energy_mask already handles diagonal (or we force it).
        # In system.py, we set 0.0 to 1.0, so diagonal is 1.0.
        mask_energy = energy_mask
        
        # Ensure self-interaction is included for GBSA (diagonal = 1.0)
        # exclusion_mask usually has 0.0 on diagonal, but GBSA needs self-energy.
        N = positions.shape[0]
        # Cast to float to avoid FutureWarning when setting 1.0
        mask_energy = mask_energy.astype(jnp.float32)
        mask_energy = mask_energy.at[jnp.diag_indices(N)].set(1.0)
        
        energy_terms = energy_terms * mask_energy
    
    # Legacy support: if mask is provided but energy_mask is not, use mask for energy too?
    # No, we want to decouple them. If energy_mask is None, include all (except self? No, self included).
    # But wait, if energy_mask is None, we sum everything.
    # If mask was provided (for radii), do we use it for energy?
    # Old behavior: yes.
    # New behavior: explicit energy_mask.
    elif mask is not None:
         # Fallback to old behavior if energy_mask not provided
         mask_energy = mask + jnp.eye(distances.shape[0])
         mask_energy = jnp.where(mask_energy > 0.0, 1.0, 0.0)
         energy_terms = energy_terms * mask_energy
        
    total_energy = prefactor * jnp.sum(energy_terms)
    
    return total_energy, born_radii


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

    # OpenMM CustomGBForce Formula (from amber14/implicit/obc2.xml)
    # Expression: 0.5*(1/L-1/U+0.25*(r-sr2^2/r)*(1/(U^2)-1/(L^2))+0.5*log(L/U)/r)
    
    inv_lower = 1.0 / lower_limit
    inv_upper = 1.0 / upper_limit
    
    # Term 1: 0.5 * (1/L - 1/U)
    term1 = 0.5 * (inv_lower - inv_upper)
    
    distance_safe = jnp.maximum(distance, 1e-6)
    r2 = distance_safe**2
    rj2 = radius_j**2
    
    # Term 2: 0.25/r * ln(L/U)
    # OpenMM: 0.5 * 0.5 * log(L/U)/r = 0.25 * log(L/U)/r
    term2 = (0.25 / distance_safe) * jnp.log(lower_limit / upper_limit)
    
    # Term 3: 0.125/r * (r^2 - rj^2) * (1/U^2 - 1/L^2)
    # OpenMM: 0.5 * 0.25 * ... = 0.125 * ...
    term3 = (0.125 / distance_safe) * (r2 - rj2) * (inv_upper**2 - inv_lower**2)
    
    total = term1 + term2 + term3
    
    # Condition: step(r + sr2 - or1) => r + rj - ri >= 0 => ri <= r + rj
    # If ri > r + rj (j inside i), result is 0.
    # We can implement this with jnp.where
    condition = radius_i <= (distance + radius_j)
    total = jnp.where(condition, total, 0.0)
    
    # Note: CustomGBForce expression does NOT include the conditional term for buried atoms (ri < rj - r).
    
    # Debug print for first pair
    # jax.debug.print("T1={t1} T2={t2} T3={t3} Tot={tot}", t1=term1[0,1], t2=term2[0,1], t3=term3[0,1], tot=total[0,1])
    
    return total


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
) -> tuple[Array, Array]:
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
        Tuple of (Total GB energy, Born Radii).
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
    
    return total_energy, born_radii


def compute_sasa(
    positions: Array,
    radii: Array,
    probe_radius: float = constants.PROBE_RADIUS,
) -> Array:
    """Computes the Solvent Accessible Surface Area (SASA) using a pairwise approximation.

    Uses the probabilistic approximation:
    $A_i = S_i \\prod_{j \\neq i} (1 - \\frac{b_{ij}}{S_i})$

    where $b_{ij}$ is the area of atom i covered by atom j.

    Args:
        positions: Atom positions (N, 3).
        radii: Atom radii (N,).
        probe_radius: Solvent probe radius.

    Returns:
        Total SASA (scalar).
    """
    N = positions.shape[0]
    
    # Expanded radii
    r_i = radii + probe_radius
    r_j = radii + probe_radius
    
    # Surface area of isolated spheres
    S_i = 4.0 * jnp.pi * r_i**2
    
    # Pairwise distances
    delta_positions = positions[:, None, :] - positions[None, :, :]
    d_ij = safe_norm(delta_positions, axis=-1)
    
    # Avoid self-interaction and division by zero
    d_ij = d_ij + jnp.eye(N) * 10.0
    
    # Broadcast radii
    r_i_mat = r_i[:, None]
    r_j_mat = r_j[None, :]
    
    # Calculate buried area b_ij
    # b_ij = (pi * r_i / d_ij) * (r_j^2 - (d_ij - r_i)^2)
    
    term1 = r_j_mat**2 - (d_ij - r_i_mat)**2
    b_ij = (jnp.pi * r_i_mat / d_ij) * term1
    
    # Conditions
    # 1. No overlap: d_ij >= r_i + r_j -> b_ij = 0
    no_overlap = d_ij >= (r_i_mat + r_j_mat)
    
    # 2. i inside j: r_j >= r_i + d_ij -> b_ij = S_i (fully buried)
    i_inside_j = r_j_mat >= (r_i_mat + d_ij)
    
    # 3. j inside i: r_i >= r_j + d_ij -> b_ij = 0 (j doesn't cover i's surface)
    j_inside_i = r_i_mat >= (r_j_mat + d_ij)
    
    # Apply conditions
    b_ij = jnp.where(no_overlap, 0.0, b_ij)
    b_ij = jnp.where(i_inside_j, S_i[:, None], b_ij)
    b_ij = jnp.where(j_inside_i, 0.0, b_ij)
    
    # Mask self
    mask = 1.0 - jnp.eye(N)
    b_ij = b_ij * mask
    
    # Fraction of area exposed
    # f_i = Product_j (1 - b_ij / S_i)
    # We clamp (1 - b_ij/S_i) to [0, 1] to avoid negative areas
    
    fraction_covered = b_ij / S_i[:, None]
    fraction_exposed_pair = jnp.maximum(0.0, 1.0 - fraction_covered)
    
    total_fraction_exposed = jnp.prod(fraction_exposed_pair, axis=1)
    
    A_i = S_i * total_fraction_exposed
    
    return jnp.sum(A_i)


def compute_sasa_neighbor_list(
    positions: Array,
    radii: Array,
    neighbor_idx: Array,
    probe_radius: float = constants.PROBE_RADIUS,
) -> Array:
    """Computes SASA using neighbor lists.

    Args:
        positions: Atom positions (N, 3).
        radii: Atom radii (N,).
        neighbor_idx: Neighbor list indices (N, K).
        probe_radius: Solvent probe radius.

    Returns:
        Total SASA (scalar).
    """
    N, K = neighbor_idx.shape
    
    # Expanded radii
    r_i = radii + probe_radius
    r_j_all = radii + probe_radius
    
    # Surface area of isolated spheres
    S_i = 4.0 * jnp.pi * r_i**2
    
    # Neighbor positions
    neighbor_positions = positions[neighbor_idx] # (N, K, 3)
    central_positions = positions[:, None, :]    # (N, 1, 3)
    
    delta_positions = central_positions - neighbor_positions
    d_ij = safe_norm(delta_positions, axis=-1) # (N, K)
    
    # Radii
    r_i_mat = r_i[:, None]      # (N, 1)
    r_j_mat = r_j_all[neighbor_idx] # (N, K)
    
    # Calculate buried area b_ij
    term1 = r_j_mat**2 - (d_ij - r_i_mat)**2
    
    # Avoid division by zero for self/padding (d_ij approx 0)
    d_ij_safe = d_ij + 1e-6
    b_ij = (jnp.pi * r_i_mat / d_ij_safe) * term1
    
    # Conditions
    no_overlap = d_ij >= (r_i_mat + r_j_mat)
    i_inside_j = r_j_mat >= (r_i_mat + d_ij)
    j_inside_i = r_i_mat >= (r_j_mat + d_ij)
    
    # Apply conditions
    b_ij = jnp.where(no_overlap, 0.0, b_ij)
    b_ij = jnp.where(i_inside_j, S_i[:, None], b_ij)
    b_ij = jnp.where(j_inside_i, 0.0, b_ij)
    
    # Mask neighbors
    mask_neighbors = neighbor_idx < N
    b_ij = jnp.where(mask_neighbors, b_ij, 0.0)
    
    # Fraction exposed
    fraction_covered = b_ij / S_i[:, None]
    fraction_exposed_pair = jnp.maximum(0.0, 1.0 - fraction_covered)
    
    total_fraction_exposed = jnp.prod(fraction_exposed_pair, axis=1)
    
    A_i = S_i * total_fraction_exposed
    
    return jnp.sum(A_i)


def compute_nonpolar_energy(
    positions: Array,
    radii: Array,
    surface_tension: float = SURFACE_TENSION,
    probe_radius: float = constants.PROBE_RADIUS,
    neighbor_idx: Array | None = None,
) -> Array:
    """Computes the non-polar solvation energy (SASA term).

    $$
    \\Delta G_{np} = \\gamma \\cdot SASA
    $$

    Args:
        positions: Atom positions (N, 3).
        radii: Atom radii (N,).
        surface_tension: Surface tension parameter (gamma).
        probe_radius: Solvent probe radius.
        neighbor_idx: Optional neighbor list indices.

    Returns:
        Non-polar energy (scalar).
    """
    if neighbor_idx is None:
        sasa = compute_sasa(positions, radii, probe_radius)
    else:
        sasa = compute_sasa_neighbor_list(positions, radii, neighbor_idx, probe_radius)
        
    return surface_tension * sasa


def compute_ace_nonpolar_energy(
    radii: Array,
    born_radii: Array,
    surface_tension: float = SURFACE_TENSION,
    probe_radius: float = constants.PROBE_RADIUS,
) -> Array:
    """Computes non-polar solvation energy using the ACE approximation.
    
    Approximation used by OpenMM's CustomGBForce (OBC1/2):
    E = 4 * pi * gamma * (r + r_probe)^2 * (r / B)^6
    
    Args:
        radii: Atomic radii (N,).
        born_radii: Born radii (N,).
        surface_tension: Surface tension (gamma) in kcal/mol/A^2.
        probe_radius: Solvent probe radius in Angstroms.
        
    Returns:
        Total non-polar energy (scalar).
    """
    # Coefficient: 4 * pi * gamma
    # For gamma=0.00542, coeff ~= 0.068
    coeff = 4.0 * jnp.pi * surface_tension
    
    term1 = (radii + probe_radius)**2
    term2 = (radii / born_radii)**6
    
    energy_per_atom = coeff * term1 * term2
    
    return jnp.sum(energy_per_atom)
