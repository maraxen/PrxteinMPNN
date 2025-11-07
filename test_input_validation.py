"""
Validate that inputs are the same between PrxteinMPNN and ColabDesign.

This script compares:
1. Coordinate preprocessing
2. Mask values
3. Residue indices
4. Chain indices
5. Any other input features
"""

import jax
import jax.numpy as jnp
import numpy as np
from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import Protein
from colabdesign.mpnn.model import mk_mpnn_model

# Define alphabets for reference
MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AF_ALPHABET = "ARNDCQEGHILKMFPSTWYVX"

# Load test PDB
print("="*80)
print("Loading and parsing 1UBQ with both implementations")
print("="*80)
pdb_path = "tests/data/1ubq.pdb"

# PrxteinMPNN parsing
protein_tuple = next(parse_input(pdb_path))
protein = Protein.from_tuple(protein_tuple)

# ColabDesign parsing
colab_model = mk_mpnn_model(model_name="v_48_020", weights="original", seed=42)
colab_model.prep_inputs(pdb_filename=pdb_path)

print("\n" + "="*80)
print("BASIC STRUCTURE INFORMATION")
print("="*80)
print(f"{'Metric':<30} {'PrxteinMPNN':<20} {'ColabDesign':<20} {'Match?':<10}")
print("-" * 80)

# Sequence length
prx_len = int(protein.mask.sum())
colab_len = int(colab_model._inputs['mask'].sum())
print(f"{'Sequence length':<30} {prx_len:<20} {colab_len:<20} {'✅' if prx_len == colab_len else '❌'}")

# Total residues (including masked)
prx_total = len(protein.mask)
colab_total = len(colab_model._inputs['mask'])
print(f"{'Total residues':<30} {prx_total:<20} {colab_total:<20} {'✅' if prx_total == colab_total else '❌'}")

# Coordinates shape
prx_coords_shape = protein.coordinates.shape
colab_coords_shape = colab_model._inputs['X'].shape
print(f"{'Coordinates shape':<30} {str(prx_coords_shape):<20} {str(colab_coords_shape):<20} {'✅' if prx_coords_shape == colab_coords_shape else '❌'}")

print("\n" + "="*80)
print("SEQUENCE COMPARISON (in their respective alphabets)")
print("="*80)
print(f"{'Position':<10} {'PDB':<8} {'PrxteinMPNN(MPNN)':<20} {'ColabDesign(AF)':<20} {'Converted Match?':<15}")
print("-" * 80)

# Get actual PDB sequence for reference
import biotite.structure.io.pdb as pdb_io
pdb_file = pdb_io.PDBFile.read(pdb_path)
structure = pdb_file.get_structure(model=1)
ca_atoms = structure[(structure.atom_name == "CA")]
residue_names = ca_atoms.res_name

three_to_one = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    'UNK': 'X'
}
pdb_sequence = [three_to_one.get(res, 'X') for res in residue_names]

# Convert ColabDesign (AF) to MPNN for comparison
af_to_mpnn_perm = np.array([AF_ALPHABET.index(aa) for aa in MPNN_ALPHABET])
colab_seq_in_mpnn = af_to_mpnn_perm[colab_model._inputs['S']]

for i in range(min(10, prx_len)):
    pdb_aa = pdb_sequence[i]
    prx_aa = MPNN_ALPHABET[protein.aatype[i]]
    colab_aa = AF_ALPHABET[colab_model._inputs['S'][i]]
    match = '✅' if protein.aatype[i] == colab_seq_in_mpnn[i] else '❌'
    print(f"{i:<10} {pdb_aa:<8} {prx_aa:<20} {colab_aa:<20} {match:<15}")

print("\n" + "="*80)
print("COORDINATE COMPARISON (N, CA, C, O atoms)")
print("="*80)

# PrxteinMPNN uses all 37 atoms, ColabDesign uses 4 backbone atoms
# Extract the same 4 backbone atoms from PrxteinMPNN for comparison
# Assuming PrxteinMPNN atom order: N=0, CA=1, C=2, O=3 (need to verify)

print("Checking first few residues...")
for res_idx in range(min(3, prx_len)):
    print(f"\nResidue {res_idx}:")

    # Get PrxteinMPNN backbone coordinates (first 4 atoms: N, CA, C, O)
    prx_backbone = protein.coordinates[res_idx, :4, :]  # shape: (4, 3)

    # Get ColabDesign backbone coordinates
    colab_backbone = colab_model._inputs['X'][res_idx]  # shape: (4, 3)

    # Compare
    for atom_idx, atom_name in enumerate(['N', 'CA', 'C', 'O']):
        prx_coord = prx_backbone[atom_idx]
        colab_coord = colab_backbone[atom_idx]
        diff = np.linalg.norm(prx_coord - colab_coord)
        match = '✅' if diff < 0.01 else '❌'
        print(f"  {atom_name:<4} PrxteinMPNN: {prx_coord}  ColabDesign: {colab_coord}  Diff: {diff:.6f} {match}")

print("\n" + "="*80)
print("MASK COMPARISON")
print("="*80)
mask_match = np.array_equal(protein.mask, colab_model._inputs['mask'])
print(f"Masks identical: {'✅ YES' if mask_match else '❌ NO'}")
if not mask_match:
    diff_positions = np.where(protein.mask != colab_model._inputs['mask'])[0]
    print(f"Different mask values at {len(diff_positions)} positions: {diff_positions[:10]}")
else:
    print(f"First 20 mask values: {protein.mask[:20]}")

print("\n" + "="*80)
print("RESIDUE INDEX COMPARISON")
print("="*80)
res_idx_match = np.array_equal(protein.residue_index, colab_model._inputs['residue_idx'])
print(f"Residue indices identical: {'✅ YES' if res_idx_match else '❌ NO'}")
print(f"PrxteinMPNN first 10: {protein.residue_index[:10]}")
print(f"ColabDesign first 10: {colab_model._inputs['residue_idx'][:10]}")

if not res_idx_match:
    diff = protein.residue_index - colab_model._inputs['residue_idx']
    print(f"Difference first 10: {diff[:10]}")
    print(f"Max difference: {np.abs(diff).max()}")

print("\n" + "="*80)
print("CHAIN INDEX COMPARISON")
print("="*80)
chain_idx_match = np.array_equal(protein.chain_index, colab_model._inputs['chain_idx'])
print(f"Chain indices identical: {'✅ YES' if chain_idx_match else '❌ NO'}")
print(f"PrxteinMPNN first 10: {protein.chain_index[:10]}")
print(f"ColabDesign first 10: {colab_model._inputs['chain_idx'][:10]}")
print(f"PrxteinMPNN unique chains: {np.unique(protein.chain_index)}")
print(f"ColabDesign unique chains: {np.unique(colab_model._inputs['chain_idx'])}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
issues = []

if not mask_match:
    issues.append("❌ Masks differ")
if not res_idx_match:
    issues.append("❌ Residue indices differ")
if not chain_idx_match:
    issues.append("❌ Chain indices differ")
if prx_coords_shape != colab_coords_shape:
    issues.append("❌ Coordinate shapes differ")

if issues:
    print("Issues found:")
    for issue in issues:
        print(f"  {issue}")
else:
    print("✅ All basic inputs match! Differences must be in model architecture or weights.")

print("\nNote: Sequence encoding is intentionally different (MPNN vs AF alphabet)")
print("      but can be converted. This is NOT an issue if handled correctly in comparisons.")
