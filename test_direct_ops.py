"""Test if direct matrix operations improve correlation."""

import jax
import jax.numpy as jnp
import joblib
from scipy.stats import pearsonr

from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import Protein
from prxteinmpnn.utils.residue_constants import atom_order
from prxteinmpnn.model.features import ProteinFeatures
from prxteinmpnn.model.features_direct import ProteinFeaturesDirect
from compare_pure_jax import colabdesign_features, compare


def inject_weights_to_features(features_module, params):
    """Inject ColabDesign weights into a features module."""
    import equinox as eqx

    # w_pos
    w = params['protein_mpnn/~/protein_features/~/positional_encodings/~/embedding_linear']['w'].T
    b = params['protein_mpnn/~/protein_features/~/positional_encodings/~/embedding_linear']['b']
    features_module = eqx.tree_at(
        lambda m: m.w_pos.weight, features_module, w
    )
    features_module = eqx.tree_at(
        lambda m: m.w_pos.bias, features_module, b
    )

    # w_e (edge embedding, no bias)
    w = params['protein_mpnn/~/protein_features/~/edge_embedding']['w'].T
    features_module = eqx.tree_at(
        lambda m: m.w_e.weight, features_module, w
    )

    # norm_edges
    scale = params['protein_mpnn/~/protein_features/~/norm_edges']['scale']
    offset = params['protein_mpnn/~/protein_features/~/norm_edges']['offset']
    features_module = eqx.tree_at(
        lambda m: m.norm_edges.weight, features_module, scale
    )
    features_module = eqx.tree_at(
        lambda m: m.norm_edges.bias, features_module, offset
    )

    # w_e_proj (this is W_e in ColabDesign!)
    w = params['protein_mpnn/~/W_e']['w'].T
    b = params['protein_mpnn/~/W_e']['b']
    features_module = eqx.tree_at(
        lambda m: m.w_e_proj.weight, features_module, w
    )
    features_module = eqx.tree_at(
        lambda m: m.w_e_proj.bias, features_module, b
    )

    return features_module


def main():
    print("="*80)
    print("TESTING DIRECT MATRIX OPERATIONS vs VMAP")
    print("="*80)

    # Load protein
    pdb_path = "tests/data/1ubq.pdb"
    protein_tuple = next(parse_input(pdb_path))
    protein = Protein.from_tuple(protein_tuple)

    # Load ColabDesign params
    key = jax.random.PRNGKey(42)
    colab_weights_path = "/tmp/ColabDesign/colabdesign/mpnn/weights/v_48_020.pkl"
    params = joblib.load(colab_weights_path)['model_state_dict']

    # Extract backbone atoms
    backbone_indices = [atom_order["N"], atom_order["CA"], atom_order["C"], atom_order["O"]]
    X_backbone = protein.coordinates[:, backbone_indices, :]

    # Get ColabDesign features (ground truth)
    print("\n1. Running ColabDesign features (ground truth)...")
    colab_E, colab_E_idx = colabdesign_features(
        X_backbone, protein.mask, protein.residue_index,
        protein.chain_index, params, k_neighbors=48
    )
    # Apply W_e projection to match PrxteinMPNN's output
    w_e = params['protein_mpnn/~/W_e']['w']
    b_e = params['protein_mpnn/~/W_e']['b']
    colab_E_proj = colab_E @ w_e + b_e

    # Create and test original vmap version
    print("\n2. Running PrxteinMPNN with vmap...")
    features_vmap = ProteinFeatures(128, 128, 48, key=key)
    features_vmap = inject_weights_to_features(features_vmap, params)

    prx_edge_vmap, prx_idx_vmap, _ = features_vmap(
        key, protein.coordinates, protein.mask,
        protein.residue_index, protein.chain_index, None
    )

    # Create and test direct ops version
    print("\n3. Running PrxteinMPNN with direct matrix ops...")
    features_direct = ProteinFeaturesDirect(128, 128, 48, key=key)
    features_direct = inject_weights_to_features(features_direct, params)

    prx_edge_direct, prx_idx_direct, _ = features_direct(
        key, protein.coordinates, protein.mask,
        protein.residue_index, protein.chain_index, None
    )

    # Compare results
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)

    print("\n📊 NEIGHBOR INDICES:")
    compare("  ColabDesign vs vmap", colab_E_idx, prx_idx_vmap)
    compare("  ColabDesign vs direct", colab_E_idx, prx_idx_direct)

    print("\n📊 EDGE FEATURES:")
    corr_vmap = compare("  ColabDesign vs vmap", colab_E_proj, prx_edge_vmap)
    corr_direct = compare("  ColabDesign vs direct", colab_E_proj, prx_edge_direct)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Vmap version correlation:   {corr_vmap:.6f}")
    print(f"Direct version correlation: {corr_direct:.6f}")
    print(f"Improvement:                {corr_direct - corr_vmap:+.6f}")

    if corr_direct > 0.99:
        print("\n✅ SUCCESS! Direct ops achieve >0.99 correlation")
        return True
    elif corr_direct > corr_vmap:
        print(f"\n🟡 IMPROVEMENT! Direct ops are better (+{corr_direct - corr_vmap:.6f})")
        return corr_direct > 0.95
    else:
        print("\n❌ No improvement from direct ops")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
