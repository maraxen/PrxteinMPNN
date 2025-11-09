from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import Protein
import jax.numpy as jnp

protein_tuple = next(parse_input('tests/data/1ubq.pdb'))
protein = Protein.from_tuple(protein_tuple)

print(f'Coordinates shape: {protein.coordinates.shape}')
print(f'First residue atoms shape: {protein.coordinates[0].shape}')

# Check which atoms are non-zero
coords = protein.coordinates[0]
nonzero_mask = (coords != 0).any(axis=1)
nonzero_count = nonzero_mask.sum()
print(f'Non-zero atoms in first residue: {nonzero_count}')

print('\nFirst 10 atoms:')
for i in range(min(10, protein.coordinates.shape[1])):
    c = protein.coordinates[0, i]
    if (c != 0).any():
        print(f'  Atom {i}: {c}')

print(f'\nExpected: Should have 4-5 atoms (N, CA, C, O, possibly CB)')
print(f'Actual: Has {protein.coordinates.shape[1]} atom slots')

# Check if ColabDesign expects (L, 4, 3) or something else
print(f'\nFor ColabDesign compatibility, need shape (L, 4, 3) or (L, 5, 3)')
print(f'Current shape: {protein.coordinates.shape}')
