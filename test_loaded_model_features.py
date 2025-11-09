"""Test features from the actual loaded model."""

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
    print("LOADED MODEL FEATURES TEST")
    print("="*80)

    # Load protein
    pdb_path = "tests/data/1ubq.pdb"
    protein_tuple = next(parse_input(pdb_path))
    protein = Protein.from_tuple(protein_tuple)

    # Load model using the comprehensive loader
    key = jax.random.PRNGKey(42)
    colab_weights_path = "/tmp/ColabDesign/colabdesign/mpnn/weights/v_48_020.pkl"
    params = joblib.load(colab_weights_path)['model_state_dict']

    print("\nLoading model with load_weights_comprehensive...")
    model = load_prxteinmpnn_with_colabdesign_weights(colab_weights_path, key=key)

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

    print("\n2. Loaded model features...")
    prx_edge, prx_idx, _ = model.features(
        key, protein.coordinates, protein.mask,
        protein.residue_index, protein.chain_index, None
    )

    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)

    compare("\nNeighbor indices", colab_E_idx, prx_idx)
    corr = compare("\nEdge features (final output)", colab_E_proj, prx_edge)

    # Debug: Check individual weight values
    print("\n" + "="*80)
    print("WEIGHT INSPECTION")
    print("="*80)

    # Check if w_pos was loaded correctly
    w_pos_expected = params['protein_mpnn/~/protein_features/~/positional_encodings/~/embedding_linear']['w'].T
    w_pos_actual = model.features.w_pos.weight
    w_pos_match = jnp.allclose(w_pos_expected, w_pos_actual)
    print(f"w_pos matches: {w_pos_match}")
    if not w_pos_match:
        print(f"  Max diff: {jnp.max(jnp.abs(w_pos_expected - w_pos_actual))}")

    # Check w_e
    w_e_expected = params['protein_mpnn/~/protein_features/~/edge_embedding']['w'].T
    w_e_actual = model.features.w_e.weight
    w_e_match = jnp.allclose(w_e_expected, w_e_actual)
    print(f"w_e matches: {w_e_match}")
    if not w_e_match:
        print(f"  Max diff: {jnp.max(jnp.abs(w_e_expected - w_e_actual))}")

    # Check w_e_proj
    w_e_proj_expected = params['protein_mpnn/~/W_e']['w'].T
    w_e_proj_actual = model.features.w_e_proj.weight
    w_e_proj_match = jnp.allclose(w_e_proj_expected, w_e_proj_actual)
    print(f"w_e_proj matches: {w_e_proj_match}")
    if not w_e_proj_match:
        print(f"  Max diff: {jnp.max(jnp.abs(w_e_proj_expected - w_e_proj_actual))}")

    return corr > 0.99


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
