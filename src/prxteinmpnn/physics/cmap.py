# File: src/prxteinmpnn/physics/cmap.py
import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

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
    
    # Wrap for periodicity
    phi_norm = phi_norm % grid_size
    psi_norm = psi_norm % grid_size
    
    # Vmap over torsions to sample correct map
    def sample_one(m_idx, p, s):
        # Coordinates shape (2, 1) for map_coordinates
        coords = jnp.array([[p], [s]])
        # Interpolate on the specific map
        return map_coordinates(
            energy_grids[m_idx], 
            coords, 
            order=3, # Bicubic
            mode='wrap'
        )[0]
        
    energies = jax.vmap(sample_one)(map_indices, phi_norm, psi_norm)
    return jnp.sum(energies)
