# File: src/prxteinmpnn/physics/cmap.py
import jax
import jax.numpy as jnp

# Inverse matrix for bicubic spline interpolation on unit square
# Maps [f00, f10, f01, f11, fx00, fx10, fx01, fx11, fy00, fy10, fy01, fy11, fxy00, fxy10, fxy01, fxy11]
# to polynomial coefficients a_ij.
A_INV = jnp.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-3, 3, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, -2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -2, -1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 1, 1, 0, 0],
    [-3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0],
    [9, -9, -9, 9, 6, 3, -6, -3, 6, -6, 3, -3, 4, 2, 2, 1],
    [-6, 6, 6, -6, -3, -3, 3, 3, -4, 4, -2, 2, -2, -2, -1, -1],
    [2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    [-6, 6, 6, -6, -4, -2, 4, 2, -3, 3, -3, 3, -2, -1, -2, -1],
    [4, -4, -4, 4, 2, 2, -2, -2, 2, -2, 2, -2, 1, 1, 1, 1]
], dtype=jnp.float32)


def eval_bicubic(coeffs_grid, x, y):
    """
    Evaluate bicubic spline at (x, y) using precomputed coefficients.
    
    Args:
        coeffs_grid: (Grid, Grid, 4) array [f, fx, fy, fxy]
        x, y: Coordinates in grid units
    """
    grid_size = coeffs_grid.shape[0]
    
    # Indices
    i = jnp.floor(x).astype(jnp.int32) % grid_size
    j = jnp.floor(y).astype(jnp.int32) % grid_size
    
    i1 = (i + 1) % grid_size
    j1 = (j + 1) % grid_size
    
    # Local coordinates
    u = x - jnp.floor(x)
    v = y - jnp.floor(y)
    
    # Gather parameters at 4 corners
    # Shape: (4,) for each
    p00 = coeffs_grid[i, j]
    p10 = coeffs_grid[i1, j]
    p01 = coeffs_grid[i, j1]
    p11 = coeffs_grid[i1, j1]
    
    # Construct x vector for A_inv * x
    # Order: f, fx, fy, fxy for 00, 10, 01, 11
    # But A_inv expects:
    # f00, f10, f01, f11
    # fx00, fx10, fx01, fx11
    # fy00, fy10, fy01, fy11
    # fxy00, fxy10, fxy01, fxy11
    
    x_vec = jnp.concatenate([
        jnp.array([p00[0], p10[0], p01[0], p11[0]]), # f
        jnp.array([p00[1], p10[1], p01[1], p11[1]]), # fx
        jnp.array([p00[2], p10[2], p01[2], p11[2]]), # fy
        jnp.array([p00[3], p10[3], p01[3], p11[3]])  # fxy
    ])
    
    # Solve for polynomial coefficients
    poly_coeffs = jnp.dot(A_INV, x_vec)
    
    # Evaluate polynomial: sum a_ij * u^i * v^j
    # poly_coeffs order: a00, a10, a20, a30, a01, ...
    
    # We can vectorize this dot product
    # Powers of u: [1, u, u^2, u^3]
    # Powers of v: [1, v, v^2, v^3]
    
    u_pow = jnp.array([1.0, u, u**2, u**3])
    v_pow = jnp.array([1.0, v, v**2, v**3])
    
    # Outer product to get u^i * v^j terms
    # Shape (4, 4). Flatten to (16,) matching poly_coeffs order (j outer, i inner)
    
    terms = jnp.outer(v_pow, u_pow).flatten()
    
    return jnp.dot(poly_coeffs, terms)


def compute_cmap_energy(phi_angles, psi_angles, map_indices, cmap_coeffs):
    """
    Compute CMAP energy using Natural Bicubic Spline.
    
    Args:
        phi_angles, psi_angles: (N_torsions,) angles in radians
        map_indices: (N_torsions,) index of map to use
        cmap_coeffs: (N_maps, Grid, Grid, 4) spline coefficients
    """
    grid_size = cmap_coeffs.shape[1]
    
    # Normalize angles to grid coordinates [0, grid_size)
    # Standard CMAP is defined on [-pi, pi]
    phi_norm = (phi_angles + jnp.pi) / (2 * jnp.pi) * grid_size
    psi_norm = (psi_angles + jnp.pi) / (2 * jnp.pi) * grid_size
    
    def sample_one(m_idx, p, s):
        return eval_bicubic(cmap_coeffs[m_idx], p, s)
        
    energies = jax.vmap(sample_one)(map_indices, phi_norm, psi_norm)
    
    # OpenMM CMAP usually does not need 0.5 factor if defined in XML correctly.
    # But previous implementation had 0.5.
    # Let's check if the XML values are already scaled or if the force implies it.
    # Standard Amber CMAP in OpenMM is just sum of energies.
    # However, if the values in XML are "energy", then we just sum them.
    # The 0.5 factor might have been a misunderstanding or specific to how JAX MD implemented torsion?
    # Torsions in JAX MD often have 0.5 if they double count.
    # But CMAP is usually listed once per residue.
    # Let's try WITHOUT 0.5 first, as OpenMM Reference implementation just adds the value.
    
    return jnp.sum(energies)
