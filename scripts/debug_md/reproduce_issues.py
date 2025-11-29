
import jax
import jax.numpy as jnp
from jax_md import space
import numpy as np
from prxteinmpnn.physics import sasa, bonded

def test_torsion():
    print("Testing Torsion...")
    # Atoms: 95-94-106-120
    # Params: 1.0, 0.00, 2.00
    # JAX E: 4.0000, JAX Phi: ~0.0
    
    # Let's create dummy positions for a Cis bond (0 degrees)
    # A-B-C-D
    # A=(1,0,0), B=(0,0,0), C=(0,1,0), D=(1,1,0)
    # B-C is along Y.
    # A-B is along X.
    # C-D is along X.
    # Dihedral A-B-C-D should be 0.
    
    r = jnp.array([
        [1.0, 0.0, 0.0], # A
        [0.0, 0.0, 0.0], # B
        [0.0, 1.0, 0.0], # C
        [1.0, 1.0, 0.0]  # D
    ])
    
    indices = jnp.array([[0, 1, 2, 3]])
    params = jnp.array([[1.0, 0.0, 2.0]]) # per, phase, k
    
    displacement_fn, _ = space.free()
    
    energy_fn = bonded.make_dihedral_energy_fn(displacement_fn, indices, params)
    
    e = energy_fn(r)
    print(f"Torsion Energy (0 deg): {e}")
    
    # Check 180 deg (Trans)
    # D=(-1, 1, 0)
    r_trans = jnp.array([
        [1.0, 0.0, 0.0], # A
        [0.0, 0.0, 0.0], # B
        [0.0, 1.0, 0.0], # C
        [-1.0, 1.0, 0.0] # D
    ])
    e_trans = energy_fn(r_trans)
    print(f"Torsion Energy (180 deg): {e_trans}")
    
    # Check OpenMM formula: E = k * (1 + cos(n*phi - phi0))
    # 0 deg: 2 * (1 + 1) = 4.
    # 180 deg: 2 * (1 - 1) = 0.
    
    print("Expected: 0 deg -> 4.0, 180 deg -> 0.0")

def test_sasa():
    print("\nTesting SASA...")
    # Create a single sphere
    r = jnp.array([[0.0, 0.0, 0.0]])
    radii = jnp.array([1.5])
    gamma = 0.005
    offset = 0.0
    
    e = sasa.compute_sasa_energy_approx(r, radii, gamma=gamma, offset=offset)
    print(f"SASA Energy (1 atom): {e}")
    
    # Expected: 4 * pi * (1.5 + 1.4)^2 * 0.005
    # R_eff = 2.9
    # Area = 4 * 3.14159 * 2.9^2 = 105.68
    # E = 105.68 * 0.005 = 0.528
    
    expected = 4 * np.pi * (1.5 + 1.4)**2 * gamma
    print(f"Expected: {expected}")
    
    # Two spheres far apart
    r2 = jnp.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    radii2 = jnp.array([1.5, 1.5])
    e2 = sasa.compute_sasa_energy_approx(r2, radii2, gamma=gamma, offset=offset)
    print(f"SASA Energy (2 atoms far): {e2}")
    print(f"Expected: {expected * 2}")

if __name__ == "__main__":
    test_torsion()
    test_sasa()
