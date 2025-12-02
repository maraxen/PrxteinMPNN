
import argparse
import jax
from prxteinmpnn.physics import force_fields

def inspect():
    parser = argparse.ArgumentParser()
    parser.add_argument("force_field_path", type=str)
    args = parser.parse_args()

    print(f"Loading force field from {args.force_field_path}...")
    ff = force_fields.load_force_field(args.force_field_path)
    print(f"Loaded Force Field: {args.force_field_path}")
    print(f"Atom Class Map Size: {len(ff.atom_class_map)}")
    print(f"CMAP Torsions Count: {len(ff.cmap_torsions)}")
    
    print("\n--- Propers (First 2) ---")
    for i, p in enumerate(ff.propers[:2]):
        print(f"{i}: {p}")
    print("\nCMAP Torsions:")
    for cmap in ff.cmap_torsions:
        print(f"  {cmap['classes']}")

    print("\nImpropers for protein-N:")
    for imp in ff.impropers:
        if imp['classes'][0] == 'protein-N':
            print(f"  {imp['classes']}")

    print("\n--- Atom Class Map (First 5) ---")
    count = 0
    for k, v in ff.atom_class_map.items():
        print(f"{k}: {v}")
        count += 1
        if count >= 5: break

if __name__ == "__main__":
    inspect()
