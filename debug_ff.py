
import jax
import jax.numpy as jnp
from prxteinmpnn.physics import force_fields, jax_md_bridge

def inspect_ff():
    ff = force_fields.load_force_field("src/prxteinmpnn/physics/force_fields/eqx/protein19SB.eqx")
    
    print(f"Loaded FF with {len(ff.bonds)} bonds and {len(ff.angles)} angles.")
    
    print("\nSample Bonds (first 10):")
    for i, b in enumerate(ff.bonds[:10]):
        print(f"  {b}")
        
    print("\nSample Angles (first 10):")
    for i, a in enumerate(ff.angles[:10]):
        print(f"  {a}")
        
    print("\nSample Atom Class Map (first 10):")
    keys = list(ff.atom_class_map.keys())
    for k in keys[:10]:
        print(f"  {k} -> {ff.atom_class_map[k]}")
        
    # Check specific ALA atoms
    print("\nChecking ALA atoms:")
    for atom in ["N", "CA", "C", "O", "CB"]:
        key = f"ALA_{atom}"
        cls = ff.atom_class_map.get(key, "MISSING")
        print(f"  ALA {atom} -> {cls}")

if __name__ == "__main__":
    inspect_ff()
