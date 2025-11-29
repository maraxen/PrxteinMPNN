# File: src/prxteinmpnn/physics/sasa.py
import jax
import jax.numpy as jnp

def compute_sasa_energy_approx(
    positions: jnp.ndarray, # (N, 3)
    radii: jnp.ndarray,     # (N,)
    gamma: float = 0.00542, # kcal/mol/A^2
    offset: float = 0.92,   # kcal/mol
    probe_radius: float = 1.4
) -> jnp.ndarray:
    """
    Computes Non-Polar Solvation Energy using a differentiable SASA approximation.
    
    Formula: E = gamma * SASA + offset
    """
    effective_radii = radii + probe_radius
    
    # Pairwise distances
    diff = positions[:, None, :] - positions[None, :, :]
    dists = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-8)
    
    # Smooth pairwise overlap calculation (Hasel-flavor)
    # Area_i = 4*pi*R_i^2 - Sum_j(Overlap_ij)
    
    # 1. Base Areas
    surface_areas = 4 * jnp.pi * effective_radii**2
    
    # 2. Analytical Pairwise Overlap (Spherical Cap Approximation)
    # Area_i = 4*pi*R_i^2 - Sum_j (Area_overlap_ij)
    # We use a soft clamp to prevent negative areas.
    
    r_i = effective_radii[:, None]
    r_j = effective_radii[None, :]
    
    # Distance between centers
    d = dists
    
    # Mask for interacting pairs: d < r_i + r_j AND d > |r_i - r_j| (partial overlap)
    # If d <= |r_i - r_j|, one is inside the other.
    # If d >= r_i + r_j, no overlap.
    
    # We only care about how much of i is covered by j.
    # If j is inside i (r_j < r_i and d + r_j < r_i): j covers a chunk of i's volume but is internal? 
    # Actually for SASA, we care about the solvent accessible surface.
    # If j is inside i, it doesn't block the surface of i from the outside? 
    # Wait, if j is inside i, it's buried. But we are calculating A_i.
    # If j is inside i, j doesn't reduce A_i.
    # If i is inside j, A_i is 0.
    
    # Case 1: i inside j (d + r_i <= r_j) -> Area_i = 0
    # Case 2: j inside i (d + r_j <= r_i) -> Overlap is 0 (j doesn't block i's outer surface)
    # Case 3: Partial overlap
    
    # Overlap Area of spherical cap of i due to j:
    # h_i = (r_j^2 - (r_i - d)^2) / (??)
    # Standard formula:
    # Area_cap_i = (pi * r_i / d) * (r_i + r_j - d) * (d + r_j - r_i)
    # This is valid for |r_i - r_j| <= d <= r_i + r_j
    
    # Let's implement this with soft masking.
    
    # Conditions
    no_overlap = d >= (r_i + r_j)
    i_inside_j = d <= (r_j - r_i) # r_j >= r_i + d
    j_inside_i = d <= (r_i - r_j) # r_i >= r_j + d
    
    # Cap Area Formula
    # A_cap = (pi * r_i / d) * (r_i + r_j - d) * (d + r_j - r_i)
    # Safe division
    d_safe = jnp.where(d < 1e-6, 1.0, d)
    
    term1 = (jnp.pi * r_i / d_safe)
    term2 = (r_i + r_j - d)
    term3 = (d + r_j - r_i)
    
    cap_area = term1 * term2 * term3
    
    # Apply logic
    # If no_overlap: 0
    # If i_inside_j: Full Area (4*pi*r_i^2) -> so A_i becomes 0
    # If j_inside_i: 0 (j is internal to i, doesn't block surface)
    
    overlap_area = jnp.where(no_overlap, 0.0, cap_area)
    overlap_area = jnp.where(i_inside_j, 4 * jnp.pi * r_i**2, overlap_area)
    overlap_area = jnp.where(j_inside_i, 0.0, overlap_area)
    
    # Mask self
    mask = 1.0 - jnp.eye(positions.shape[0])
    overlap_area = overlap_area * mask
    
    # Sum overlaps
    # Note: Simple summation overestimates overlap (double counting of multiple neighbors).
    # We need a correction factor or scaling.
    # LCPO uses P1, P2, P3, P4 parameters.
    # Here we use a scaling factor `S` to dampen the over-subtraction.
    # A_i = Max(0, 4*pi*r_i^2 - S * Sum(Overlap))
    
    overlap_scale = 0.1 # Empirical scaling to account for multi-body overlap redundancy
    total_overlap = jnp.sum(overlap_area, axis=1) * overlap_scale
    
    net_area = jax.nn.relu(surface_areas - total_overlap)
    
    total_sasa = jnp.sum(net_area)
    
    return gamma * total_sasa + offset
