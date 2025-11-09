"""Check if PDB C-beta matches computed C-beta."""

import jax.numpy as jnp
from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import Protein
from prxteinmpnn.utils.residue_constants import atom_order
from prxteinmpnn.utils.coordinates import compute_backbone_coordinates
from compare_pure_jax import colabdesign_get_cb, compare


def main():
    print("="*80)
    print("PDB C-BETA vs COMPUTED C-BETA")
    print("="*80)

    # Load protein
    pdb_path = "tests/data/1ubq.pdb"
    protein_tuple = next(parse_input(pdb_path))
    protein = Protein.from_tuple(protein_tuple)

    # Get C-beta from PDB (atom37 index 3)
    cb_from_pdb = protein.coordinates[:, atom_order["CB"], :]
    print(f"C-beta from PDB shape: {cb_from_pdb.shape}")
    print(f"Sample (residue 0): {cb_from_pdb[0]}")

    # Compute C-beta from backbone
    backbone_indices = [atom_order["N"], atom_order["CA"], atom_order["C"], atom_order["O"]]
    X_backbone = protein.coordinates[:, backbone_indices, :]

    # ColabDesign method
    Y = X_backbone.swapaxes(0, 1)
    cb_colabdesign = colabdesign_get_cb(Y)
    print(f"\nC-beta from ColabDesign computation: {cb_colabdesign.shape}")
    print(f"Sample (residue 0): {cb_colabdesign[0]}")

    # PrxteinMPNN method
    backbone_prx = compute_backbone_coordinates(protein.coordinates)
    cb_prxteinmpnn = backbone_prx[:, 4, :]  # Index 4 is CB
    print(f"\nC-beta from PrxteinMPNN computation: {cb_prxteinmpnn.shape}")
    print(f"Sample (residue 0): {cb_prxteinmpnn[0]}")

    print("\n" + "="*80)
    print("COMPARISONS")
    print("="*80)

    corr1 = compare("\n1. PDB C-beta vs ColabDesign computed", cb_from_pdb, cb_colabdesign)
    corr2 = compare("\n2. PDB C-beta vs PrxteinMPNN computed", cb_from_pdb, cb_prxteinmpnn)
    corr3 = compare("\n3. ColabDesign computed vs PrxteinMPNN computed", cb_colabdesign, cb_prxteinmpnn)

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    if corr1 < 0.999:
        print("❌ PDB C-beta does NOT match computed C-beta!")
        print("   This could explain the 0.971 correlation if PrxteinMPNN uses PDB C-beta")
        print("   while ColabDesign computes it!")
    else:
        print("✅ PDB C-beta matches computed C-beta perfectly")

if __name__ == "__main__":
    main()
