"""
Debug script to investigate sequence parsing differences between PrxteinMPNN and ColabDesign.

This script:
1. Parses 1UBQ.pdb with both implementations
2. Compares the resulting integer sequences
3. Traces the conversion at each step
4. Identifies where the implementations diverge
"""

import jax
import jax.numpy as jnp
import numpy as np
from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import Protein

print("="*80)
print("STEP 1: Parse with PrxteinMPNN")
print("="*80)

# Use the test PDB from the repository
pdb_path = "tests/data/1ubq.pdb"

protein_tuple = next(parse_input(pdb_path))
protein = Protein.from_tuple(protein_tuple)
print(f"Sequence length: {protein.mask.sum():.0f}")
print(f"First 20 residues: {protein.aatype[:20]}")
print(f"aatype dtype: {protein.aatype.dtype}")
print(f"aatype shape: {protein.aatype.shape}")
print(f"Full sequence: {protein.aatype}")

print("\n" + "="*80)
print("STEP 2: Parse with ColabDesign")
print("="*80)

try:
    from colabdesign.mpnn.model import mk_mpnn_model

    colab_model = mk_mpnn_model(model_name="v_48_020", weights="original")
    colab_model.prep_inputs(pdb_filename=pdb_path)
    print(f"Sequence length: {colab_model._inputs['mask'].sum():.0f}")
    print(f"First 20 residues: {colab_model._inputs['S'][:20]}")
    print(f"S dtype: {colab_model._inputs['S'].dtype}")
    print(f"S shape: {colab_model._inputs['S'].shape}")
    print(f"Full sequence: {colab_model._inputs['S']}")

    print("\n" + "="*80)
    print("STEP 3: Compare Sequences")
    print("="*80)
    if np.array_equal(protein.aatype, colab_model._inputs['S']):
        print("✅ Sequences match!")
    else:
        print("❌ Sequences differ!")

        # Find differences
        min_len = min(len(protein.aatype), len(colab_model._inputs['S']))
        diffs = np.where(protein.aatype[:min_len] != colab_model._inputs['S'][:min_len])[0]
        print(f"Number of differences: {len(diffs)} / {min_len}")
        print(f"First 10 differences:")
        for i in diffs[:10]:
            print(f"  Position {i}: PrxteinMPNN={protein.aatype[i]}, ColabDesign={colab_model._inputs['S'][i]}")
except ImportError as e:
    print(f"⚠️  Could not import ColabDesign: {e}")
    print("Installing ColabDesign...")
    import subprocess
    subprocess.run(["pip", "install", "-e", "/tmp/ColabDesign"], check=False)
    print("Please run this script again after installation completes.")
    exit(0)

print("\n" + "="*80)
print("STEP 4: Check Alphabet Conversion")
print("="*80)

# Check PrxteinMPNN alphabets
try:
    from prxteinmpnn.io.parsing.mappings import MPNN_ALPHABET, AF_ALPHABET
    from prxteinmpnn.utils.residue_constants import restypes

    print(f"PrxteinMPNN MPNN alphabet: {MPNN_ALPHABET if 'MPNN_ALPHABET' in dir() else 'NOT FOUND'}")
    print(f"PrxteinMPNN AF alphabet:   {AF_ALPHABET if 'AF_ALPHABET' in dir() else 'NOT FOUND'}")
    print(f"PrxteinMPNN restypes:      {''.join(restypes)}")
except Exception as e:
    print(f"Error checking PrxteinMPNN alphabets: {e}")

# Check ColabDesign alphabets
try:
    import inspect
    import sys
    sys.path.insert(0, '/tmp/ColabDesign')

    # Try to find the alphabet in ColabDesign
    colab_mpnn_module = sys.modules.get('colabdesign.mpnn.model')
    if colab_mpnn_module:
        source = inspect.getsource(colab_mpnn_module)
        # Look for alphabet definitions
        if 'ACDEFGHIKLMNPQRSTVWYX' in source:
            print("\nColabDesign MPNN alphabet found in source: ACDEFGHIKLMNPQRSTVWYX")
        if 'ARNDCQEGHILKMFPSTWYVX' in source:
            print("ColabDesign AF alphabet found in source:   ARNDCQEGHILKMFPSTWYVX")
except Exception as e:
    print(f"Error checking ColabDesign alphabets: {e}")

print("\n" + "="*80)
print("STEP 5: Detailed Alphabet Mapping")
print("="*80)

# Define both alphabets explicitly
MPNN_ORDER = "ACDEFGHIKLMNPQRSTVWYX"  # ProteinMPNN native order
AF_ORDER = "ARNDCQEGHILKMFPSTWYVX"    # AlphaFold order

print(f"\nMPNN alphabet: {MPNN_ORDER}")
print(f"AF alphabet:   {AF_ORDER}")
print("\nIndex mapping (MPNN -> AF):")
print(f"{'AA':<4} {'MPNN_idx':<10} {'AF_idx':<10}")
for i, aa in enumerate(MPNN_ORDER):
    af_idx = AF_ORDER.index(aa) if aa in AF_ORDER else -1
    print(f"{aa:<4} {i:<10} {af_idx:<10}")

print("\n" + "="*80)
print("STEP 6: Get actual PDB sequence")
print("="*80)

# Read the PDB file to get the actual sequence
try:
    import biotite.structure.io.pdb as pdb
    pdb_file = pdb.PDBFile.read(pdb_path)
    structure = pdb_file.get_structure(model=1)

    # Get CA atoms only to extract sequence
    ca_atoms = structure[(structure.atom_name == "CA")]
    residue_names = ca_atoms.res_name

    # Convert 3-letter to 1-letter codes
    three_to_one = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
        'UNK': 'X'
    }

    sequence_str = ''.join([three_to_one.get(res, 'X') for res in residue_names])
    print(f"PDB sequence (1-letter codes): {sequence_str}")
    print(f"Length: {len(sequence_str)}")

    # Convert to MPNN indices
    mpnn_indices = [MPNN_ORDER.index(aa) if aa in MPNN_ORDER else 20 for aa in sequence_str]
    print(f"\nExpected MPNN indices: {mpnn_indices[:20]}... (first 20)")
    print(f"PrxteinMPNN indices:   {list(protein.aatype[:20])}")

    # Convert to AF indices
    af_indices = [AF_ORDER.index(aa) if aa in AF_ORDER else 20 for aa in sequence_str]
    print(f"\nExpected AF indices: {af_indices[:20]}... (first 20)")
    print(f"ColabDesign indices: {list(colab_model._inputs['S'][:20])}")

    print("\n" + "="*80)
    print("DIAGNOSIS")
    print("="*80)

    if list(protein.aatype[:20]) == mpnn_indices[:20]:
        print("✅ PrxteinMPNN uses MPNN alphabet correctly")
    elif list(protein.aatype[:20]) == af_indices[:20]:
        print("❌ PrxteinMPNN is using AF alphabet (should use MPNN!)")
    else:
        print("❓ PrxteinMPNN uses neither alphabet - unexpected!")

    if list(colab_model._inputs['S'][:20]) == mpnn_indices[:20]:
        print("✅ ColabDesign uses MPNN alphabet correctly")
    elif list(colab_model._inputs['S'][:20]) == af_indices[:20]:
        print("⚠️  ColabDesign is using AF alphabet (might be intentional)")
    else:
        print("❓ ColabDesign uses neither alphabet - unexpected!")

except Exception as e:
    print(f"Error reading PDB sequence: {e}")
    import traceback
    traceback.print_exc()
