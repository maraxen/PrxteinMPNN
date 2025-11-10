"""Compare backbone atom coordinates between implementations."""

import jax.numpy as jnp
from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import Protein
from prxteinmpnn.utils.coordinates import compute_backbone_coordinates
from prxteinmpnn.utils.residue_constants import atom_order
from colabdesign.mpnn import mk_mpnn_model

print("="*80)
print("COMPARING BACKBONE ATOM COORDINATES")
print("="*80)

pdb_path = "tests/data/1ubq.pdb"

# 1. ColabDesign
print("\n1. ColabDesign backbone extraction...")
mpnn_model = mk_mpnn_model()
mpnn_model.prep_inputs(pdb_filename=pdb_path)
colab_X = mpnn_model._inputs['X']  # (L, 4, 3) - N, CA, C, O
print(f"   Shape: {colab_X.shape}")
print(f"   First residue atoms:")
print(f"     N:  {colab_X[0, 0]}")
print(f"     CA: {colab_X[0, 1]}")
print(f"     C:  {colab_X[0, 2]}")
print(f"     O:  {colab_X[0, 3]}")

# 2. PrxteinMPNN
print("\n2. PrxteinMPNN backbone extraction...")
protein_tuple = next(parse_input(pdb_path))
protein = Protein.from_tuple(protein_tuple)

# PrxteinMPNN uses compute_backbone_coordinates which extracts N, CA, C, O, CB
backbone = compute_backbone_coordinates(protein.coordinates)
print(f"   Shape: {backbone.shape}")
print(f"   First residue atoms:")
print(f"     N:  {backbone[0, 0]}")
print(f"     CA: {backbone[0, 1]}")
print(f"     C:  {backbone[0, 2]}")
print(f"     O:  {backbone[0, 3]}")
print(f"     CB: {backbone[0, 4]}")

# 3. Compare N, CA, C, O
print("\n3. Comparison...")
atoms = ['N', 'CA', 'C', 'O']
for i, atom in enumerate(atoms):
    match = jnp.allclose(colab_X[0, i], backbone[0, i], atol=1e-5)
    diff = jnp.max(jnp.abs(colab_X[0, i] - backbone[0, i]))
    print(f"   {atom:3s}: match={match}, max_diff={diff:.6f}")
    if not match:
        print(f"        ColabDesign: {colab_X[0, i]}")
        print(f"        PrxteinMPNN: {backbone[0, i]}")

# 4. Check raw input coordinates
print("\n4. Raw PDB atom coordinates...")
print(f"   protein.coordinates shape: {protein.coordinates.shape}")
print(f"   atom_order: {atom_order}")
print(f"\n   First residue from protein.coordinates:")
for atom_name in ['N', 'CA', 'C', 'O']:
    idx = atom_order[atom_name]
    print(f"     {atom_name} (index {idx}): {protein.coordinates[0, idx]}")
