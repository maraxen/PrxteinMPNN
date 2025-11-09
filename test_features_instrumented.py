"""Instrument the features module to capture intermediates."""

import jax
import jax.numpy as jnp
import joblib
from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import Protein
from prxteinmpnn.utils.residue_constants import atom_order
from prxteinmpnn.utils.coordinates import compute_backbone_coordinates, compute_backbone_distance
from prxteinmpnn.utils.radial_basis import compute_radial_basis
from prxteinmpnn.utils.graph import compute_neighbor_offsets
from load_weights_comprehensive import load_prxteinmpnn_with_colabdesign_weights
from compare_pure_jax import colabdesign_features, compare

def main():
    print("="*80)
    print("INSTRUMENTED FEATURES MODULE TEST")
    print("="*80)

    # Load protein
    pdb_path = "tests/data/1ubq.pdb"
    protein_tuple = next(parse_input(pdb_path))
    protein = Protein.from_tuple(protein_tuple)

    # Load model
    key = jax.random.PRNGKey(42)
    colab_weights_path = "/tmp/ColabDesign/colabdesign/mpnn/weights/v_48_020.pkl"
    params = joblib.load(colab_weights_path)['model_state_dict']

    model = load_prxteinmpnn_with_colabdesign_weights(colab_weights_path, key=key)

    # Extract backbone
    backbone_indices = [atom_order["N"], atom_order["CA"], atom_order["C"], atom_order["O"]]
    X_backbone = protein.coordinates[:, backbone_indices, :]

    print("\n1. ColabDesign features...")
    colab_E, colab_E_idx = colabdesign_features(
        X_backbone, protein.mask, protein.residue_index,
        protein.chain_index, params, k_neighbors=48
    )

    print("\n2. PrxteinMPNN features (step by step instrumentation)...")

    # Manually step through what the features module does
    noised_coordinates = protein.coordinates  # No noise
    backbone_atom_coordinates = compute_backbone_coordinates(noised_coordinates)
    distances = compute_backbone_distance(backbone_atom_coordinates)

    distances_masked = jnp.where(
        (protein.mask[:, None] * protein.mask[None, :]).astype(bool),
        distances, jnp.inf
    )

    top_k = jax.jit(jax.lax.top_k, static_argnames=("k",))
    _, neighbor_indices = top_k(-distances_masked, 48)
    neighbor_indices = jnp.array(neighbor_indices, dtype=jnp.int32)

    rbf = compute_radial_basis(backbone_atom_coordinates, neighbor_indices)
    neighbor_offsets = compute_neighbor_offsets(protein.residue_index, neighbor_indices)

    edge_chains = (protein.chain_index[:, None] == protein.chain_index[None, :]).astype(int)
    edge_chains_neighbors = jnp.take_along_axis(edge_chains, neighbor_indices, axis=1)

    neighbor_offset_factor = jnp.clip(neighbor_offsets + 32, 0, 64)
    edge_chain_factor = (1 - edge_chains_neighbors) * 65
    encoded_offset = neighbor_offset_factor * edge_chains_neighbors + edge_chain_factor
    encoded_offset_one_hot = jax.nn.one_hot(encoded_offset, 66)

    print("   Step 1: Positional encoding...")
    encoded_positions = jax.vmap(jax.vmap(model.features.w_pos))(encoded_offset_one_hot)

    print("   Step 2: Concatenate with RBF...")
    edges = jnp.concatenate([encoded_positions, rbf], axis=-1)

    print("   Step 3: Edge embedding (w_e)...")
    edge_features = jax.vmap(jax.vmap(model.features.w_e))(edges)

    print("   Step 4: LayerNorm...")
    edge_features_normed = jax.vmap(jax.vmap(model.features.norm_edges))(edge_features)

    print("   Step 5: W_e projection...")
    edge_features_proj = jax.vmap(jax.vmap(model.features.w_e_proj))(edge_features_normed)

    print("\n3. Compare with ColabDesign after each step...")

    # Get ColabDesign intermediates
    Y = X_backbone.swapaxes(0, 1)
    from compare_pure_jax import colabdesign_get_cb, colabdesign_get_edge_idx, colabdesign_get_rbf
    Cb = colabdesign_get_cb(Y)
    Y = jnp.concatenate([Y, Cb[None]], 0)

    colab_E_idx = colabdesign_get_edge_idx(Y[1], protein.mask, 48)

    edges_pairs = jnp.array([[1,1],[0,0],[2,2],[3,3],[4,4],
                              [1,0],[1,2],[1,3],[1,4],[0,2],
                              [0,3],[0,4],[4,2],[4,3],[3,2],
                              [0,1],[2,1],[3,1],[4,1],[2,0],
                              [3,0],[4,0],[2,4],[3,4],[2,3]])

    colab_RBF = jax.vmap(lambda x: colabdesign_get_rbf(Y[x[0]], Y[x[1]], colab_E_idx))(edges_pairs)
    colab_RBF = colab_RBF.transpose((1, 2, 0, 3)).reshape(colab_RBF.shape[1], colab_RBF.shape[2], -1)

    offset = protein.residue_index[:,None] - protein.residue_index[None,:]
    colab_offset = jnp.take_along_axis(offset, colab_E_idx, 1)
    E_chains = (protein.chain_index[:,None] == protein.chain_index[None,:]).astype(int)
    colab_E_chains = jnp.take_along_axis(E_chains, colab_E_idx, 1)

    colab_d = jnp.clip(colab_offset + 32, 0, 64) * colab_E_chains + (1 - colab_E_chains) * 65
    colab_d_onehot = jax.nn.one_hot(colab_d, 66)

    w_pos = params['protein_mpnn/~/protein_features/~/positional_encodings/~/embedding_linear']['w']
    b_pos = params['protein_mpnn/~/protein_features/~/positional_encodings/~/embedding_linear']['b']
    colab_E_positional = colab_d_onehot @ w_pos + b_pos

    colab_E_concat = jnp.concatenate((colab_E_positional, colab_RBF), -1)

    w_e = params['protein_mpnn/~/protein_features/~/edge_embedding']['w']
    colab_E_embedded = colab_E_concat @ w_e

    # LayerNorm
    mean = colab_E_embedded.mean(axis=-1, keepdims=True)
    var = colab_E_embedded.var(axis=-1, keepdims=True)
    colab_E_normed = (colab_E_embedded - mean) / jnp.sqrt(var + 1e-5)
    scale = params['protein_mpnn/~/protein_features/~/norm_edges']['scale']
    offset_param = params['protein_mpnn/~/protein_features/~/norm_edges']['offset']
    colab_E_normed = colab_E_normed * scale + offset_param

    w_e_proj = params['protein_mpnn/~/W_e']['w']
    b_e_proj = params['protein_mpnn/~/W_e']['b']
    colab_E_proj = colab_E_normed @ w_e_proj + b_e_proj

    print("\n" + "="*80)
    print("STEP-BY-STEP COMPARISONS")
    print("="*80)

    compare("\n1. Neighbor indices", colab_E_idx, neighbor_indices)
    compare("\n2. RBF", colab_RBF, rbf)
    compare("\n3. Positional encoding", colab_E_positional, encoded_positions)
    compare("\n4. Concatenated", colab_E_concat, edges)
    compare("\n5. After w_e (edge embedding)", colab_E_embedded, edge_features)
    compare("\n6. After LayerNorm", colab_E_normed, edge_features_normed)
    corr = compare("\n7. After W_e projection (FINAL)", colab_E_proj, edge_features_proj)

    if corr > 0.99:
        print("\n✅ Found the issue or confirmed no issue!")
    else:
        print(f"\n❌ Still diverged at correlation {corr:.6f}")

if __name__ == "__main__":
    main()
