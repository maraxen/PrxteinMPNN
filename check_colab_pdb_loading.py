"""Check what ColabDesign actually loads from PDB."""

from colabdesign.af.prep import prep_pdb
from colabdesign.shared.protein import residue_constants

print("="*80)
print("COLABDESIGN PDB LOADING")
print("="*80)

pdb_path = "tests/data/1ubq.pdb"

# Use ColabDesign's prep_pdb
pdb = prep_pdb(pdb_path, chain=None, ignore_missing=True)

print(f"\n1. Loaded PDB structure:")
print(f"   all_atom_positions shape: {pdb['batch']['all_atom_positions'].shape}")
print(f"   all_atom_mask shape: {pdb['batch']['all_atom_mask'].shape}")

# Check first residue
first_res = pdb['batch']['all_atom_positions'][0]
print(f"\n2. First residue atoms (raw atom37):")
for i, atom_name in enumerate(['N', 'CA', 'C', 'CB', 'O']):
    idx = residue_constants.atom_order[atom_name]
    print(f"   {atom_name:3s} (index {idx}): {first_res[idx]}")

# Now extract backbone like ColabDesign does
atom_idx = tuple(residue_constants.atom_order[k] for k in ["N","CA","C","O"])
print(f"\n3. Backbone extraction indices: {atom_idx}")
X = pdb['batch']['all_atom_positions'][:,atom_idx]
print(f"   X shape: {X.shape}")
print(f"   First residue backbone:")
for i, atom_name in enumerate(['N', 'CA', 'C', 'O']):
    print(f"     {atom_name:3s}: {X[0, i]}")
