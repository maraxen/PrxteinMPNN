"""Debug how atoms are being mapped during PDB parsing."""

import biotite.structure.io.pdb as pdb
from prxteinmpnn.utils.residue_constants import atom_order

print("="*80)
print("PDB ATOM MAPPING DEBUG")
print("="*80)

pdb_path = "tests/data/1ubq.pdb"

# Load PDB with Biotite
pdb_file = pdb.PDBFile.read(pdb_path)
atom_array = pdb_file.get_structure()[0]

# Get first residue atoms
first_res_mask = atom_array.res_id == 1
first_res_atoms = atom_array[first_res_mask]

print("\n1. First residue atoms from PDB file:")
for i, atom in enumerate(first_res_atoms):
    print(f"   Position {i}: {atom.atom_name:4s} at {atom.coord}")

print("\n2. Our atom_order mapping:")
for atom_name in ['N', 'CA', 'C', 'CB', 'O']:
    idx = atom_order.get(atom_name, -1)
    print(f"   {atom_name:4s} → index {idx}")

print("\n3. What happens when we map:")
print("   If PDB has atoms in order: N, CA, C, CB, O")
print("   And we map them using atom_order:")
for i, atom_name in enumerate(['N', 'CA', 'C', 'CB', 'O']):
    target_idx = atom_order[atom_name]
    print(f"     PDB position {i} ({atom_name}) → atom37 index {target_idx}")
