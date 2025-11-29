
import jax
from prxteinmpnn.physics import force_fields

def inspect():
    print("Loading force field...")
    ff = force_fields.load_force_field_from_hub("ff14SB")
    
    print("\n--- Propers (First 2) ---")
    for i, p in enumerate(ff.propers[:2]):
        print(f"{i}: {p}")
        
    print("\n--- Impropers (First 2) ---")
    for i, p in enumerate(ff.impropers[:2]):
        print(f"{i}: {p}")

    print("\n--- Atom Class Map (First 5) ---")
    count = 0
    for k, v in ff.atom_class_map.items():
        print(f"{k}: {v}")
        count += 1
        if count >= 5: break

if __name__ == "__main__":
    inspect()
