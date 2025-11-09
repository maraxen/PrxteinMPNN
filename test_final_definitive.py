"""Final definitive test: model.features() vs colabdesign_features() + W_e."""

import jax
import jax.numpy as jnp
import joblib
from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import Protein
from prxteinmpnn.utils.residue_constants import atom_order
from load_weights_comprehensive import load_prxteinmpnn_with_colabdesign_weights
from compare_pure_jax import colabdesign_features, compare


def main():
    print("="*80)
    print("FINAL DEFINITIVE TEST")
    print("="*80)

    # Load protein
    pdb_path = "tests/data/1ubq.pdb"
    protein_tuple = next(parse_input(pdb_path))
    protein = Protein.from_tuple(protein_tuple)

    # Load model and params
    key = jax.random.PRNGKey(42)
    colab_weights_path = "/tmp/ColabDesign/colabdesign/mpnn/weights/v_48_020.pkl"
    model = load_prxteinmpnn_with_colabdesign_weights(colab_weights_path, key=key)
    params = joblib.load(colab_weights_path)['model_state_dict']

    # Extract backbone
    backbone_indices = [atom_order["N"], atom_order["CA"], atom_order["C"], atom_order["O"]]
    X_backbone = protein.coordinates[:, backbone_indices, :]

    print("\n1. ColabDesign: colabdesign_features() + manual W_e...")
    colab_E, colab_E_idx = colabdesign_features(
        X_backbone, protein.mask, protein.residue_index,
        protein.chain_index, params, k_neighbors=48
    )
    print(f"   colab_E shape (after LayerNorm, before W_e): {colab_E.shape}")

    w_e = params['protein_mpnn/~/W_e']['w']
    b_e = params['protein_mpnn/~/W_e']['b']
    colab_E_final = colab_E @ w_e + b_e
    print(f"   colab_E_final shape (after W_e): {colab_E_final.shape}")

    print("\n2. PrxteinMPNN: model.features()...")
    prx_edge, prx_idx, _ = model.features(
        key, protein.coordinates, protein.mask,
        protein.residue_index, protein.chain_index, None
    )
    print(f"   prx_edge shape (final output): {prx_edge.shape}")

    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)

    compare("\nNeighbor indices", colab_E_idx, prx_idx)
    corr = compare("\nEdge features (FINAL)", colab_E_final, prx_edge)

    print("\n" + "="*80)
    print("DEBUG INFO")
    print("="*80)
    print(f"Sample values (residue 0, neighbor 0, first 5 features):")
    print(f"  ColabDesign: {colab_E_final[0, 0, :5]}")
    print(f"  PrxteinMPNN:  {prx_edge[0, 0, :5]}")
    print(f"  Difference:   {jnp.abs(colab_E_final[0, 0, :5] - prx_edge[0, 0, :5])}")

    if corr > 0.99:
        print("\n✅ PERFECT! Correlation > 0.99")
        print("The features are matching correctly!")
        return True
    elif corr > 0.97:
        print(f"\n🟡 CLOSE but not perfect: {corr:.6f}")
        print("There's a subtle numerical difference somewhere...")
        return False
    else:
        print(f"\n❌ DIVERGENCE: {corr:.6f}")
        print("Something is fundamentally wrong!")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
