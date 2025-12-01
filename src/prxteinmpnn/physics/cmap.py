# File: src/prxteinmpnn/physics/cmap.py
import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

def cubic_hermite(p0, p1, p2, p3, x):
    """
    Cubic Hermite spline interpolation.
    p0, p1, p2, p3: Values at x=-1, 0, 1, 2.
    x: Fraction between p1 and p2 (0 to 1).
    """
    a = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3
    b = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3
    c = -0.5 * p0 + 0.5 * p2
    d = p1
    
    return a * x**3 + b * x**2 + c * x + d

def bicubic_interp(grid, x, y):
    """
    Bicubic interpolation on a 2D grid with periodic boundary conditions.
    x, y: Coordinates in grid units.
    """
    grid_size = grid.shape[0]
    
    # Indices
    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    
    # Fractional parts
    dx = x - x0
    dy = y - y0
    
    # 4x4 neighborhood indices (wrapped)
    x_indices = (x0 + jnp.arange(-1, 3)) % grid_size
    y_indices = (y0 + jnp.arange(-1, 3)) % grid_size
    
    # Gather 4x4 grid points
    # grid[x, y]
    # We need to index grid with meshgrid of indices
    # patch: (4, 4)
    patch = grid[x_indices[:, None], y_indices[None, :]]
    
    # Interpolate along Y for each of the 4 X rows
    # We have 4 rows. Each row has 4 points.
    # We interpolate at y for each row.
    
    col0 = cubic_hermite(patch[0, 0], patch[0, 1], patch[0, 2], patch[0, 3], dy)
    col1 = cubic_hermite(patch[1, 0], patch[1, 1], patch[1, 2], patch[1, 3], dy)
    col2 = cubic_hermite(patch[2, 0], patch[2, 1], patch[2, 2], patch[2, 3], dy)
    col3 = cubic_hermite(patch[3, 0], patch[3, 1], patch[3, 2], patch[3, 3], dy)
    
    # Interpolate along X
    value = cubic_hermite(col0, col1, col2, col3, dx)
    
    return value

def compute_cmap_energy(
    phi_angles: jnp.ndarray, # (N_torsions,) radians
    psi_angles: jnp.ndarray, # (N_torsions,) radians
    map_indices: jnp.ndarray, # (N_torsions,)
    energy_grids: jnp.ndarray # (N_maps, Grid, Grid)
) -> jnp.ndarray:
    """
    Computes CMAP energy correction using bicubic interpolation.
    """
    grid_size = energy_grids.shape[1]
    
    # Normalize angles to grid coordinates [0, grid_size)
    # Standard CMAP is defined on [-pi, pi]
    phi_norm = (phi_angles + jnp.pi) / (2 * jnp.pi) * grid_size
    psi_norm = (psi_angles + jnp.pi) / (2 * jnp.pi) * grid_size
    
    # Wrap for periodicity (0 to grid_size)
    # Note: bicubic_interp handles wrapping of indices, but we should ensure x, y are reasonable
    # Actually, x, y can be anything, the floor and mod will handle it.
    
    # Vmap over torsions to sample correct map
    def sample_one(m_idx, p, s):
        return bicubic_interp(energy_grids[m_idx], p, s)
        
    energies = jax.vmap(sample_one)(map_indices, phi_norm, psi_norm)
    # Factor of 0.5 required to match OpenMM (likely due to XML definition)
    return 0.5 * jnp.sum(energies)
