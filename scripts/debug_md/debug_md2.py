"""Test MD with realistic initial coordinates."""
import jax
import jax.numpy as jnp
import numpy as np
from prxteinmpnn.physics import simulate, force_fields, jax_md_bridge, system
from jax_md import space

# Create a simple 1-residue system
res_names = ["ALA"]
from prxteinmpnn.utils import residue_constants
atom_names = residue_constants.residue_atoms["ALA"]

print(f"Atom names for ALA: {atom_names}")

ff = force_fields.load_force_field_from_hub("ff14SB")
params = jax_md_bridge.parameterize_system(ff, res_names, atom_names)

n_atoms = len(params["charges"])
print(f"Number of atoms: {n_atoms}")

# Create realistic initial coordinates for ALA
# Typical bond lengths: C-C ~1.5Å, C-N ~1.3Å, C-O ~1.2Å, C-H ~1.1Å
# Let's build a simple extended chain conformation
coords = np.array([
    [0.0, 0.0, 0.0],      # N
    [1.45, 0.0, 0.0],     # CA  
    [2.0, 1.5, 0.0],      # C
    [3.2, 1.7, 0.0],      # O
    [2.0, -0.5, 1.0],     # CB
], dtype=np.float32)

# If we have more atoms (hydrogens), add them at reasonable distances
if n_atoms > 5:
    # Add hydrogens at ~1.1Å from heavy atoms
    extra_coords = []
    for i in range(n_atoms - 5):
        # Place near the backbone with some offset
        extra_coords.append([0.5 + i*0.3, 0.5, 0.5])
    coords = np.vstack([coords, np.array(extra_coords, dtype=np.float32)])

coords = jnp.array(coords[:n_atoms])  # Ensure we have exactly n_atoms

print(f"Coords shape: {coords.shape}")
print(f"Coords:\n{coords}")

# Create energy function
displacement_fn, shift_fn = space.free()
energy_fn = system.make_energy_fn(displacement_fn, params)

# Test energy
E_init = energy_fn(coords)
print(f"\nInitial energy: {E_init:.2e}")
print(f"Is finite: {jnp.isfinite(E_init)}")

if jnp.isfinite(E_init) and E_init < 1e10:  # Reasonable energy
    print("\n✓ Initial energy is reasonable")
    
    # Try minimization with smaller timestep
    print(f"\nRunning minimization with dt_start=1e-5...")
    
    # Manually create minimizer with smaller timestep
    from jax_md import minimize
    init_fn, apply_fn = minimize.fire_descent(
        energy_fn, 
        shift_fn=shift_fn, 
        dt_start=1e-5,  # Much smaller
        dt_max=1e-3     # Much smaller max too
    )
    state = init_fn(coords)
    
    # Run a few steps manually to monitor
    for i in range(20):
        state = apply_fn(state)
        E = energy_fn(state.position)
        if i % 5 == 0:
            print(f"Step {i}: E = {E:.2e}, finite = {jnp.isfinite(E)}")
        if not jnp.isfinite(E):
            print(f"⚠️ NaN at step {i}")
            break
    
    if jnp.isfinite(E):
        print(f"\n✓ Minimization stable!")
        print(f"Final energy: {E:.2e}")
        print(f"Energy drop: {E_init - E:.2e}")
else:
    print(f"\n⚠️ Initial energy too large: {E_init:.2e}")
    print("Need better initial coordinates")
