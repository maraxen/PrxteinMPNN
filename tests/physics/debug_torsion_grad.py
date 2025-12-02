import jax
import jax.numpy as jnp
from jax_md import space
import numpy as np

def run_debug():
    print("Debugging Torsion Gradient...")
    
    # Define a single torsion
    # p1-p2-p3-p4
    # Let's use coordinates that might be problematic (e.g. planar or linear)
    # Or just random
    
    # Case 1: Random
    r = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.5, 0.8, 0.0],
        [1.5, 0.8, 1.0]
    ], dtype=jnp.float32)
    
    displacement_fn, shift_fn = space.free()
    
    def compute_dihedral(r):
        r_i, r_j, r_k, r_l = r[0], r[1], r[2], r[3]
        
        b0 = -1.0 * displacement_fn(r_j, r_i) # i->j
        b1 = displacement_fn(r_k, r_j) # j->k
        b2 = displacement_fn(r_l, r_k) # k->l
        
        # Normalize b1
        b1_norm = jnp.linalg.norm(b1) + 1e-8
        b1_unit = b1 / b1_norm
        
        # Project b0 and b2 onto plane perpendicular to b1
        # v = b0 - (b0.b1)*b1
        v = b0 - jnp.dot(b0, b1_unit) * b1_unit
        w = b2 - jnp.dot(b2, b1_unit) * b1_unit
        
        x = jnp.dot(v, w)
        y = jnp.dot(jnp.cross(b1_unit, v), w)
        
        return jnp.arctan2(y, x)
        
    def energy_fn(r):
        angle = compute_dihedral(r)
        # Simple energy: k * (1 + cos(n*phi - phase))
        k = 10.0
        n = 1.0 # Use n=1 to ensure non-zero force at random angle
        phase = 0.0
        return k * (1.0 + jnp.cos(n * angle - phase))

        
    # Check Grad
    grad_fn = jax.grad(energy_fn)
    forces = -grad_fn(r)
    
    print("Forces (Random Config):")
    print(forces)
    
    # Finite Diff
    eps = 1e-4
    fd_forces = np.zeros_like(forces)
    for i in range(4):
        for j in range(3):
            r_p = r.at[i, j].add(eps)
            e_p = energy_fn(r_p)
            r_m = r.at[i, j].add(-eps)
            e_m = energy_fn(r_m)
            fd_forces[i, j] = -(e_p - e_m) / (2 * eps)
            
    print("FD Forces:")
    print(fd_forces)
    print("Diff:")
    print(np.abs(forces - fd_forces))
    
    # Case 2: Planar (0 or 180 degrees)
    print("\nCase 2: Planar Config")
    r_planar = jnp.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0] # U-shape (0 deg)
    ], dtype=jnp.float32)
    
    forces_planar = -grad_fn(r_planar)
    print("Forces (Planar):")
    print(forces_planar)
    
    # FD
    fd_forces_planar = np.zeros_like(forces_planar)
    for i in range(4):
        for j in range(3):
            r_p = r_planar.at[i, j].add(eps)
            e_p = energy_fn(r_p)
            r_m = r_planar.at[i, j].add(-eps)
            e_m = energy_fn(r_m)
            fd_forces_planar[i, j] = -(e_p - e_m) / (2 * eps)
            
    print("FD Forces (Planar):")
    print(fd_forces_planar)
    print("Diff:")
    print(np.abs(forces_planar - fd_forces_planar))
    
    # Case 3: Linear (Singularity)
    print("\nCase 3: Linear Config (Singularity)")
    r_linear = jnp.array([
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0]
    ], dtype=jnp.float32)
    
    forces_linear = -grad_fn(r_linear)
    print("Forces (Linear):")
    print(forces_linear)

if __name__ == "__main__":
    run_debug()
