"""Debug temperature test."""
import jax
import jax.numpy as jnp
import numpy as np
from prxteinmpnn.physics import simulate, force_fields, jax_md_bridge, system
from jax_md import quantity, space, simulate as jax_simulate

key = jax.random.PRNGKey(2)

# Create a system (Single ALA)
res_names = ["ALA"]
from prxteinmpnn.utils import residue_constants
atom_names = residue_constants.residue_atoms["ALA"]

ff = force_fields.load_force_field_from_hub("ff14SB")
params = jax_md_bridge.parameterize_system(ff, res_names, atom_names)

n_atoms = len(params["charges"])

# Create realistic initial coordinates
coords = np.array([
    [0.0, 0.0, 0.0],      # N
    [1.45, 0.0, 0.0],     # CA  
    [2.0, 1.5, 0.0],      # C
    [3.2, 1.7, 0.0],      # O
    [2.0, -0.5, 1.0],     # CB
], dtype=np.float32)

# Add hydrogens if needed
if n_atoms > 5:
    extra = np.random.randn(n_atoms - 5, 3).astype(np.float32) * 0.3
    extra += coords[1]  # Near CA
    coords = np.vstack([coords, extra])

coords = jnp.array(coords[:n_atoms])

# Run NVT
target_temp = 300.0

displacement_fn, shift_fn = space.free()
energy_fn = system.make_energy_fn(displacement_fn, params)

dt = 1e-3  # 1 fs
kT = target_temp * 0.001987  # kcal/mol

print(f"Target temperature: {target_temp} K")
print(f"kT: {kT:.4f} kcal/mol")
print(f"Number of atoms: {n_atoms}")

# Initialize Langevin dynamics
init_fn, apply_fn = jax_simulate.nvt_langevin(
    energy_fn,
    shift_fn=shift_fn,
    dt=dt,
    kT=kT,
    gamma=1.0
)

state = init_fn(key, coords)

# Equilibrate and monitor
print("\nEquilibration:")
for i in range(100):
    state = apply_fn(state)
    if i % 20 == 0:
        K = quantity.kinetic_energy(velocity=state.velocity, mass=1.0)
        print(f"Step {i}: KE = {K:.2f}")

# Sample kinetic energies
print("\nSampling:")
K_history = []
for i in range(100):
    state = apply_fn(state)
    K = quantity.kinetic_energy(velocity=state.velocity, mass=1.0)
    K_history.append(K)
    if i % 20 == 0:
        print(f"Step {i}: KE = {K:.2f}")

avg_K = np.mean(K_history)
std_K = np.std(K_history)

kB = 0.001987  # kcal/(mol·K)
# Expected: KE = (3/2) * N_atoms * kB * T
expected_K = (3 * n_atoms / 2) * kB * target_temp

print(f"\nResults:")
print(f"Average KE: {avg_K:.2f} ± {std_K:.2f}")
print(f"Expected KE: {expected_K:.2f}")
print(f"Ratio: {avg_K/expected_K:.2f}")

# Infer temperature from KE
inferred_T = avg_K / ((3 * n_atoms / 2) * kB)
print(f"Inferred temperature: {inferred_T:.1f} K")
