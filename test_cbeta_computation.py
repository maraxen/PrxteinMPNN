"""Compare C-beta computation between ColabDesign and PrxteinMPNN."""

import jax.numpy as jnp
from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import Protein
from prxteinmpnn.utils.residue_constants import atom_order
from prxteinmpnn.utils.coordinates import compute_backbone_coordinates, compute_c_beta
from compare_pure_jax import colabdesign_get_cb, compare


def main():
    print("="*80)
    print("C-BETA COMPUTATION COMPARISON")
    print("="*80)

    # Load protein
    pdb_path = "tests/data/1ubq.pdb"
    protein_tuple = next(parse_input(pdb_path))
    protein = Protein.from_tuple(protein_tuple)

    # Extract backbone atoms
    backbone_indices = [atom_order["N"], atom_order["CA"], atom_order["C"], atom_order["O"]]
    X_backbone = protein.coordinates[:, backbone_indices, :]  # (L, 4, 3)

    print("\n1. ColabDesign C-beta computation...")
    # ColabDesign expects (4, L, 3)
    Y_colab = X_backbone.swapaxes(0, 1)  # (4, L, 3)
    Cb_colab = colabdesign_get_cb(Y_colab)  # (L, 3)
    print(f"   Shape: {Cb_colab.shape}")

    print("\n2. PrxteinMPNN C-beta computation...")
    # PrxteinMPNN computes from full backbone
    backbone_prx = compute_backbone_coordinates(protein.coordinates)  # (L, 5, 3)
    Cb_prx = backbone_prx[:, 4, :]  # Extract Cb (index 4)
    print(f"   Shape: {Cb_prx.shape}")

    # Also compute directly using the function
    N = X_backbone[:, 0, :]
    CA = X_backbone[:, 1, :]
    C = X_backbone[:, 2, :]
    ca_to_n = CA - N
    c_to_ca = C - CA
    Cb_prx_direct = compute_c_beta(ca_to_n, c_to_ca, CA)
    print(f"   Direct computation shape: {Cb_prx_direct.shape}")

    print("\n" + "="*80)
    print("COMPARISONS")
    print("="*80)

    compare("\n1. C-beta (ColabDesign vs PrxteinMPNN from backbone)", Cb_colab, Cb_prx)
    compare("\n2. C-beta (ColabDesign vs PrxteinMPNN direct)", Cb_colab, Cb_prx_direct)

    # Check the coefficients
    print("\n" + "="*80)
    print("COEFFICIENTS")
    print("="*80)
    print("ColabDesign: f1=-0.58273431, f2=0.56802827, f3=-0.54067466")
    print("PrxteinMPNN: f1=-0.58273431, f2=0.56802827, f3=-0.54067466")
    print("(Should be identical)")

    # Sample values
    print(f"\nSample C-beta (residue 0):")
    print(f"  ColabDesign: {Cb_colab[0]}")
    print(f"  PrxteinMPNN: {Cb_prx[0]}")
    print(f"  Difference:  {jnp.abs(Cb_colab[0] - Cb_prx[0])}")


if __name__ == "__main__":
    main()
