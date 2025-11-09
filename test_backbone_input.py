"""Test if using only backbone atoms (4) improves correlation."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import pearsonr
import joblib

from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import Protein
from prxteinmpnn.utils.residue_constants import atom_order
from load_weights_comprehensive import load_prxteinmpnn_with_colabdesign_weights
from compare_pure_jax import colabdesign_forward, compare


def main():
    print("Testing if backbone-only input improves correlation...")

    # Load protein
    pdb_path = "tests/data/1ubq.pdb"
    protein_tuple = next(parse_input(pdb_path))
    protein = Protein.from_tuple(protein_tuple)

    # Load models
    key = jax.random.PRNGKey(42)
    colab_weights_path = "/tmp/ColabDesign/colabdesign/mpnn/weights/v_48_020.pkl"
    prx_model = load_prxteinmpnn_with_colabdesign_weights(colab_weights_path, key=key)
    params = joblib.load(colab_weights_path)['model_state_dict']

    # Extract backbone atoms (N, CA, C, O)
    backbone_indices = [atom_order["N"], atom_order["CA"], atom_order["C"], atom_order["O"]]
    X_backbone = protein.coordinates[:, backbone_indices, :]  # (76, 4, 3)

    # Create a modified atom37 array with only backbone atoms filled in
    # and zeros for all other atoms
    backbone_only_atom37 = jnp.zeros_like(protein.coordinates)  # (76, 37, 3)
    for i, idx in enumerate(backbone_indices):
        backbone_only_atom37 = backbone_only_atom37.at[:, idx, :].set(X_backbone[:, i, :])

    print(f"\nBackbone-only atom37 shape: {backbone_only_atom37.shape}")
    print(f"Non-zero atoms per residue: {(backbone_only_atom37[0] != 0).any(axis=1).sum()}")

    # Run ColabDesign
    colab_logits, _, _, _ = colabdesign_forward(
        X_backbone, protein.mask, protein.residue_index,
        protein.chain_index, params, k_neighbors=48
    )

    # Run PrxteinMPNN with full atom37
    print("\n1. PrxteinMPNN with full atom37 (37 atoms):")
    _, prx_logits_full = prx_model(
        protein.coordinates, protein.mask, protein.residue_index,
        protein.chain_index, "unconditional", prng_key=key
    )
    corr_full = compare("  Logits (full atom37)", colab_logits, prx_logits_full)

    # Run PrxteinMPNN with backbone-only atom37
    print("\n2. PrxteinMPNN with backbone-only atom37 (4 atoms in atom37 format):")
    _, prx_logits_backbone = prx_model(
        backbone_only_atom37, protein.mask, protein.residue_index,
        protein.chain_index, "unconditional", prng_key=key
    )
    corr_backbone = compare("  Logits (backbone-only atom37)", colab_logits, prx_logits_backbone)

    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"Full atom37 correlation:          {corr_full:.6f}")
    print(f"Backbone-only atom37 correlation: {corr_backbone:.6f}")
    print(f"Improvement:                      {corr_backbone - corr_full:.6f}")

    if corr_backbone > 0.90:
        print("\n✅ SUCCESS! Backbone-only input achieves >0.90 correlation")
    elif corr_backbone > corr_full:
        print(f"\n🟡 IMPROVEMENT! Backbone-only is better (+{corr_backbone - corr_full:.6f})")
    else:
        print("\n❌ No improvement from backbone-only input")


if __name__ == "__main__":
    main()
