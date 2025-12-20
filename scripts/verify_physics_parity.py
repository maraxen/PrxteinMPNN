import sys
import os
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from collections import namedtuple

# Add scripts/ to path to import legacy impl
sys.path.append(str(Path(__file__).parent))
try:
    import legacy_physics_impl
except ImportError:
    print("Warning: Could not import legacy_physics_impl. Create it first.")
    sys.exit(1)

# Import new implementation
# We suspect prxteinmpnn.physics uses proxide
from prxteinmpnn.physics import features as proxide_features

# Dummy ProteinTuple mimicking prxteinmpnn.utils.data_structures.ProteinTuple
ProteinTuple = namedtuple(
    "ProteinTuple",
    [
        "coordinates", "full_coordinates", "charges", "aatype",
        "sigmas", "epsilons"
    ]
)

def create_dummy_protein():
    key = jax.random.PRNGKey(42)
    # 2 residues, 5 backbone atoms each (N, CA, C, O, CB)
    # Total 10 atoms for backbone
    # Let's add some sidechain atoms too -> 15 atoms total
    
    n_res = 2
    n_atoms = 15
    
    # Random coords
    full_coords = jax.random.normal(key, (n_atoms, 3)) * 10.0
    
    # Backbone coords: (n_res, 5, 3)
    # We take first 10 atoms as backbone atoms
    bb_coords = full_coords[:n_res*5].reshape(n_res, 5, 3)
    
    # Charges
    charges = jax.random.normal(key, (n_atoms,))
    
    # VdW
    sigmas = jnp.abs(jax.random.normal(key, (n_atoms,))) + 1.0
    epsilons = jnp.abs(jax.random.normal(key, (n_atoms,))) * 0.1
    
    # Aatype (not used for physics but needed for tuple)
    aatype = jnp.zeros((n_res,), dtype=jnp.int32)
    
    return ProteinTuple(
        coordinates=bb_coords,
        full_coordinates=full_coords,
        charges=charges,
        aatype=aatype,
        sigmas=sigmas,
        epsilons=epsilons
    )

def create_shifted_dummy_protein():
    """Create a protein where backbone atoms are shifted to avoid self-interaction/NaNs."""
    p = create_dummy_protein()
    # Shift backbone far away from source atoms
    new_coords = p.coordinates + 1000.0
    return ProteinTuple(
        coordinates=new_coords,
        full_coordinates=p.full_coordinates,
        charges=p.charges,
        aatype=p.aatype,
        sigmas=p.sigmas,
        epsilons=p.epsilons
    )

def run_comparison(protein):
    # 1. Electrostatics
    print("\n--- Electrostatics ---")
    try:
        legacy_estat = legacy_physics_impl.compute_electrostatic_node_features(protein)
        proxide_estat = proxide_features.compute_electrostatic_node_features(protein)
        
        # Check NaNs
        if jnp.any(jnp.isnan(legacy_estat)) or jnp.any(jnp.isnan(proxide_estat)):
            print("NaNs detected!")
            if jnp.all(jnp.isnan(legacy_estat) == jnp.isnan(proxide_estat)):
                 print("NaN pattern matches.")
            else:
                 print("NaN pattern MISMATCH.")

        # Filter NaNs for numeric comparison
        mask = jnp.isfinite(legacy_estat) & jnp.isfinite(proxide_estat)
        if jnp.sum(mask) == 0:
            print("No finite values to compare.")
        else:
            diff = jnp.abs(legacy_estat[mask] - proxide_estat[mask])
            max_diff = jnp.max(diff)
            mean_diff = jnp.mean(diff)
            print(f"Max diff: {max_diff:.6e}")
            print(f"Mean diff: {mean_diff:.6e}")
            
            if np.allclose(legacy_estat[mask], proxide_estat[mask], atol=1e-5):
                print("MATCHED (Finite values)!")
            else:
                print("MISMATCH (Finite values)!")
            
    except Exception as e:
        print(f"Error in Electrostatics: {e}")

    # 2. VdW
    print("\n--- VdW ---")
    try:
        legacy_vdw = legacy_physics_impl.compute_vdw_node_features(protein)
        proxide_vdw = proxide_features.compute_vdw_node_features(protein)
        
        if jnp.any(jnp.isnan(legacy_vdw)) or jnp.any(jnp.isnan(proxide_vdw)):
            print("NaNs detected!")
            if jnp.all(jnp.isnan(legacy_vdw) == jnp.isnan(proxide_vdw)):
                 print("NaN pattern matches.")
            else:
                 print("NaN pattern MISMATCH.")

        mask = jnp.isfinite(legacy_vdw) & jnp.isfinite(proxide_vdw)
        if jnp.sum(mask) == 0:
             print("No finite values to compare.")
        else:
            diff = jnp.abs(legacy_vdw[mask] - proxide_vdw[mask])
            max_diff = jnp.max(diff)
            mean_diff = jnp.mean(diff)
            print(f"Max diff: {max_diff:.6e}")
            print(f"Mean diff: {mean_diff:.6e}")
            
            if np.allclose(legacy_vdw[mask], proxide_vdw[mask], atol=1e-5):
                print("MATCHED (Finite values)!")
            else:
                print("MISMATCH (Finite values)!")
                if jnp.sum(mask) > 0:
                    print(f"Legacy sample: {legacy_vdw[mask][0]}")
                    print(f"Proxide sample: {proxide_vdw[mask][0]}")
            
    except Exception as e:
        print(f"Error in VdW: {e}")

def main():
    print("Verifying Physics Parity...")
    
    # Test 1: Standard (Overlapping)
    print("\n=== Test 1: Standard (Overlapping) ===")
    protein = create_dummy_protein()
    run_comparison(protein)
    
    # Test 2: Shifted (No Self-Interaction)
    print("\n=== Test 2: Shifted (No Self-Interaction) ===")
    protein_shifted = create_shifted_dummy_protein()
    run_comparison(protein_shifted)

if __name__ == "__main__":
    main()
