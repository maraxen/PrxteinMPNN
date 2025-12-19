import jax
import jax.numpy as jnp
import equinox as eqx
from prxteinmpnn.physics import force_fields
import sys

try:
    ff_path = "src/prxteinmpnn/physics/force_fields/eqx/protein19SB.eqx"
    print(f"Loading {ff_path}...")
    ff = force_fields.load_force_field(ff_path)
    print("Success!")
    print(f"CMAP shape: {ff.cmap_energy_grids.shape}")
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()
