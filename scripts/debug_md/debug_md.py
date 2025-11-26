"""Debug MD simulation NaN issue."""
import jax
import jax.numpy as jnp
import numpy as np
from prxteinmpnn.physics import simulate, force_fields, jax_md_bridge, system
from jax_md import space

# Create a simple 1-residue system
res_names = ["ALA"]
from prxteinmpnn.utils import residue_constants
atom_names = residue_constants.residue_atoms["ALA"]

print(f"Loading force field...")
ff = force_fields.load_force_field_from_hub("ff14SB")
print(f"Force field loaded: {type(ff)}")

print(f"\nParameterizing system...")
params = jax_md_bridge.parameterize_system(ff, res_names, atom_names)
print(f"Number of atoms: {len(params['charges'])}")
print(f"Charges range: [{params['charges'].min():.3f}, {params['charges'].max():.3f}]")
print(f"Sigmas range: [{params['sigmas'].min():.3f}, {params['sigmas'].max():.3f}]")
print(f"Epsilons range: [{params['epsilons'].min():.3f}, {params['epsilons'].max():.3f}]")
print(f"Number of bonds: {len(params['bonds'])}")
print(f"Number of angles: {len(params['angles'])}")

# Create simple initial coordinates
n_atoms = len(params["charges"])
print(f"\nCreating initial coordinates for {n_atoms} atoms...")
coords = jnp.zeros((n_atoms, 3))
# Add small random displacement
coords = coords + jax.random.normal(jax.random.PRNGKey(0), (n_atoms, 3)) * 0.1

print(f"Coords shape: {coords.shape}")
print(f"Coords range: [{coords.min():.3f}, {coords.max():.3f}]")

# Create energy function
print(f"\nCreating energy function...")
displacement_fn, shift_fn = space.free()
energy_fn = system.make_energy_fn(displacement_fn, params)

# Test energy calculation
print(f"\nTesting energy calculation...")
E = energy_fn(coords)
print(f"Initial energy: {E}")
print(f"Is finite: {jnp.isfinite(E)}")

if not jnp.isfinite(E):
    print("\n⚠️ Energy is not finite at initial coordinates!")
    print("Debugging energy components...")
    
    # Try to isolate which component is causing NaN
    # We'll need to look at the energy function implementation
else:
    print("\n✓ Initial energy is finite")
    
    # Try one minimization step
    print(f"\nTesting minimization...")
    try:
        r_min = simulate.run_simulation(
            params,
            coords,
            temperature=0.0,
            min_steps=10,
            therm_steps=0,
            key=jax.random.PRNGKey(1)
        )
        E_min = energy_fn(r_min)
        print(f"Energy after 10 steps: {E_min}")
        print(f"Is finite: {jnp.isfinite(E_min)}")
        
        if jnp.isfinite(E_min):
            print(f"Energy change: {E - E_min:.3f}")
        else:
            print("\n⚠️ Energy became NaN during minimization!")
            
    except Exception as e:
        print(f"\n⚠️ Error during minimization: {e}")
        import traceback
        traceback.print_exc()
