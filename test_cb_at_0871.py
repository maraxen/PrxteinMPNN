"""Check CB coordinates at 0.871 correlation commit."""

import jax
from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import Protein
from prxteinmpnn.utils.residue_constants import atom_order

pdb_path = "tests/data/1ubq.pdb"

protein_tuple = next(parse_input(pdb_path))
protein = Protein.from_tuple(protein_tuple)

print("At commit acae2d6 (0.871 correlation):")
print(f"  CB (index 3): {protein.coordinates[0, atom_order['CB']]}")
print(f"  O (index 4): {protein.coordinates[0, atom_order['O']]}")

print("\nExpected from PDB:")
print(f"  CB should be: [25.112, 24.880, 3.649]")
print(f"  O should be: [27.886, 26.463, 4.263]")
