"""Compare actual features module output vs manual computation."""

import jax
import jax.numpy as jnp
import joblib
from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import Protein
from prxteinmpnn.model.features import ProteinFeatures
from compare_pure_jax import compare, colabdesign_features
from prxteinmpnn.utils.residue_constants import atom_order


def inject_weights(features_module, params):
    """Inject ColabDesign weights."""
    import equinox as eqx

    # w_pos
    w = params['protein_mpnn/~/protein_features/~/positional_encodings/~/embedding_linear']['w'].T
    b = params['protein_mpnn/~/protein_features/~/positional_encodings/~/embedding_linear']['b']
    features_module = eqx.tree_at(lambda m: m.w_pos.weight, features_module, w)
    features_module = eqx.tree_at(lambda m: m.w_pos.bias, features_module, b)

    # w_e (no bias)
    w = params['protein_mpnn/~/protein_features/~/edge_embedding']['w'].T
    features_module = eqx.tree_at(lambda m: m.w_e.weight, features_module, w)

    # norm_edges
    scale = params['protein_mpnn/~/protein_features/~/norm_edges']['scale']
    offset = params['protein_mpnn/~/protein_features/~/norm_edges']['offset']
    features_module = eqx.tree_at(lambda m: m.norm_edges.weight, features_module, scale)
    features_module = eqx.tree_at(lambda m: m.norm_edges.bias, features_module, offset)

    # w_e_proj
    w = params['protein_mpnn/~/W_e']['w'].T
    b = params['protein_mpnn/~/W_e']['b']
    features_module = eqx.tree_at(lambda m: m.w_e_proj.weight, features_module, w)
    features_module = eqx.tree_at(lambda m: m.w_e_proj.bias, features_module, b)

    return features_module


def main():
    print("="*80)
    print("FEATURES MODULE vs MANUAL COMPUTATION")
    print("="*80)

    # Load protein
    pdb_path = "tests/data/1ubq.pdb"
    protein_tuple = next(parse_input(pdb_path))
    protein = Protein.from_tuple(protein_tuple)

    # Load params
    key = jax.random.PRNGKey(42)
    colab_weights_path = "/tmp/ColabDesign/colabdesign/mpnn/weights/v_48_020.pkl"
    params = joblib.load(colab_weights_path)['model_state_dict']

    # Extract backbone atoms
    backbone_indices = [atom_order["N"], atom_order["CA"], atom_order["C"], atom_order["O"]]
    X_backbone = protein.coordinates[:, backbone_indices, :]

    print("\n1. ColabDesign manual computation...")
    colab_E, colab_E_idx = colabdesign_features(
        X_backbone, protein.mask, protein.residue_index,
        protein.chain_index, params, k_neighbors=48
    )
    # Apply W_e projection
    w_e = params['protein_mpnn/~/W_e']['w']
    b_e = params['protein_mpnn/~/W_e']['b']
    colab_E_proj = colab_E @ w_e + b_e

    print("\n2. PrxteinMPNN features module...")
    features = ProteinFeatures(128, 128, 48, key=key)
    features = inject_weights(features, params)

    prx_edge, prx_idx, _ = features(
        key, protein.coordinates, protein.mask,
        protein.residue_index, protein.chain_index, None
    )

    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)

    compare("\nNeighbor indices", colab_E_idx, prx_idx)
    corr = compare("\nEdge features (final output)", colab_E_proj, prx_edge)

    if corr > 0.99:
        print("\n✅ SUCCESS! Features module matches ColabDesign!")
        return True
    elif corr > 0.97:
        print(f"\n🟡 CLOSE! Correlation: {corr:.6f}")
        print("The 0.03 gap might be acceptable...")
        return False
    else:
        print(f"\n❌ DIVERGENCE! Correlation: {corr:.6f}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
