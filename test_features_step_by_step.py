"""Compare features step by step to find where divergence occurs."""

import jax
import jax.numpy as jnp
import joblib
from scipy.stats import pearsonr

from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import Protein
from prxteinmpnn.utils.residue_constants import atom_order
from prxteinmpnn.model.features import ProteinFeatures
from prxteinmpnn.utils.coordinates import compute_backbone_coordinates
from prxteinmpnn.utils.radial_basis import compute_radial_basis
from prxteinmpnn.utils.graph import compute_neighbor_offsets
from compare_pure_jax import colabdesign_features, compare, colabdesign_get_rbf, colabdesign_get_cb


def inject_weights(features_module, params):
    """Inject ColabDesign weights."""
    import equinox as eqx

    w = params['protein_mpnn/~/protein_features/~/positional_encodings/~/embedding_linear']['w'].T
    b = params['protein_mpnn/~/protein_features/~/positional_encodings/~/embedding_linear']['b']
    features_module = eqx.tree_at(lambda m: m.w_pos.weight, features_module, w)
    features_module = eqx.tree_at(lambda m: m.w_pos.bias, features_module, b)

    w = params['protein_mpnn/~/protein_features/~/edge_embedding']['w'].T
    features_module = eqx.tree_at(lambda m: m.w_e.weight, features_module, w)

    scale = params['protein_mpnn/~/protein_features/~/norm_edges']['scale']
    offset = params['protein_mpnn/~/protein_features/~/norm_edges']['offset']
    features_module = eqx.tree_at(lambda m: m.norm_edges.weight, features_module, scale)
    features_module = eqx.tree_at(lambda m: m.norm_edges.bias, features_module, offset)

    w = params['protein_mpnn/~/W_e']['w'].T
    b = params['protein_mpnn/~/W_e']['b']
    features_module = eqx.tree_at(lambda m: m.w_e_proj.weight, features_module, w)
    features_module = eqx.tree_at(lambda m: m.w_e_proj.bias, features_module, b)

    return features_module


def main():
    print("="*80)
    print("STEP-BY-STEP FEATURE COMPARISON")
    print("="*80)

    # Load protein
    pdb_path = "tests/data/1ubq.pdb"
    protein_tuple = next(parse_input(pdb_path))
    protein = Protein.from_tuple(protein_tuple)

    # Load params
    key = jax.random.PRNGKey(42)
    colab_weights_path = "/tmp/ColabDesign/colabdesign/mpnn/weights/v_48_020.pkl"
    params = joblib.load(colab_weights_path)['model_state_dict']

    # Extract backbone
    backbone_indices = [atom_order["N"], atom_order["CA"], atom_order["C"], atom_order["O"]]
    X_backbone = protein.coordinates[:, backbone_indices, :]

    print("\n1. COLABDESIGN FEATURES (step by step)...")

    # ColabDesign backbone
    Y = X_backbone.swapaxes(0, 1)  # (4, L, 3)
    Cb = colabdesign_get_cb(Y)
    Y = jnp.concatenate([Y, Cb[None]], 0)  # (5, L, 3)

    # Get neighbor indices (both should match)
    from compare_pure_jax import colabdesign_get_edge_idx
    colab_E_idx = colabdesign_get_edge_idx(Y[1], protein.mask, 48)

    # RBF for all 25 atom pairs
    edges_pairs = jnp.array([[1,1],[0,0],[2,2],[3,3],[4,4],
                              [1,0],[1,2],[1,3],[1,4],[0,2],
                              [0,3],[0,4],[4,2],[4,3],[3,2],
                              [0,1],[2,1],[3,1],[4,1],[2,0],
                              [3,0],[4,0],[2,4],[3,4],[2,3]])

    colab_RBF = jax.vmap(lambda x: colabdesign_get_rbf(Y[x[0]], Y[x[1]], colab_E_idx))(edges_pairs)
    colab_RBF = colab_RBF.transpose((1, 2, 0, 3))
    colab_RBF = colab_RBF.reshape(colab_RBF.shape[:-2] + (-1,))
    print(f"   ColabDesign RBF shape: {colab_RBF.shape}")

    # Positional encoding
    offset = protein.residue_index[:,None] - protein.residue_index[None,:]
    colab_offset = jnp.take_along_axis(offset, colab_E_idx, 1)

    E_chains = (protein.chain_index[:,None] == protein.chain_index[None,:]).astype(int)
    colab_E_chains = jnp.take_along_axis(E_chains, colab_E_idx, 1)

    max_rel = 32
    colab_d = jnp.clip(colab_offset + max_rel, 0, 2*max_rel) * colab_E_chains + (1 - colab_E_chains) * (2*max_rel + 1)
    colab_d_onehot = jax.nn.one_hot(colab_d, 2*max_rel + 2)

    w_pos = params['protein_mpnn/~/protein_features/~/positional_encodings/~/embedding_linear']['w']
    b_pos = params['protein_mpnn/~/protein_features/~/positional_encodings/~/embedding_linear']['b']
    colab_E_positional = colab_d_onehot @ w_pos + b_pos
    print(f"   ColabDesign positional shape: {colab_E_positional.shape}")

    # Concatenate
    colab_E_concat = jnp.concatenate((colab_E_positional, colab_RBF), -1)
    print(f"   ColabDesign concat shape: {colab_E_concat.shape}")

    # Edge embedding
    w_e = params['protein_mpnn/~/protein_features/~/edge_embedding']['w']
    colab_E_embedded = colab_E_concat @ w_e
    print(f"   ColabDesign embedded shape: {colab_E_embedded.shape}")

    print("\n2. PRXTEINMPNN FEATURES (step by step)...")

    # PrxteinMPNN backbone
    prx_backbone = compute_backbone_coordinates(protein.coordinates)  # (L, 5, 3)
    print(f"   PrxteinMPNN backbone shape: {prx_backbone.shape}")

    # Get neighbor indices
    from prxteinmpnn.utils.coordinates import compute_backbone_distance
    top_k = jax.jit(jax.lax.top_k, static_argnames=("k",))
    distances = compute_backbone_distance(prx_backbone)
    distances_masked = jnp.where(
        (protein.mask[:, None] * protein.mask[None, :]).astype(bool),
        distances, jnp.inf
    )
    _, prx_neighbor_indices = top_k(-distances_masked, 48)
    prx_neighbor_indices = jnp.array(prx_neighbor_indices, dtype=jnp.int32)

    # RBF
    prx_rbf = compute_radial_basis(prx_backbone, prx_neighbor_indices)
    print(f"   PrxteinMPNN RBF shape: {prx_rbf.shape}")

    # Positional encoding
    prx_neighbor_offsets = compute_neighbor_offsets(protein.residue_index, prx_neighbor_indices)

    prx_edge_chains = (protein.chain_index[:, None] == protein.chain_index[None, :]).astype(int)
    prx_edge_chains_neighbors = jnp.take_along_axis(prx_edge_chains, prx_neighbor_indices, axis=1)

    prx_neighbor_offset_factor = jnp.clip(prx_neighbor_offsets + 32, 0, 2*32)
    prx_edge_chain_factor = (1 - prx_edge_chains_neighbors) * (2*32 + 1)
    prx_encoded_offset = prx_neighbor_offset_factor * prx_edge_chains_neighbors + prx_edge_chain_factor
    prx_encoded_offset_one_hot = jax.nn.one_hot(prx_encoded_offset, 2*32 + 2)

    prx_encoded_positions = jax.vmap(jax.vmap(lambda x: x @ w_pos + b_pos))(prx_encoded_offset_one_hot)
    print(f"   PrxteinMPNN positional shape: {prx_encoded_positions.shape}")

    # Concatenate
    prx_edges = jnp.concatenate([prx_encoded_positions, prx_rbf], axis=-1)
    print(f"   PrxteinMPNN concat shape: {prx_edges.shape}")

    # Edge embedding
    prx_edge_features = jax.vmap(jax.vmap(lambda x: x @ w_e))(prx_edges)
    print(f"   PrxteinMPNN embedded shape: {prx_edge_features.shape}")

    print("\n" + "="*80)
    print("COMPARISONS")
    print("="*80)

    compare("\n1. Neighbor indices", colab_E_idx, prx_neighbor_indices)
    compare("\n2. RBF features", colab_RBF, prx_rbf)
    compare("\n3. Positional encoding", colab_E_positional, prx_encoded_positions)
    compare("\n4. Concatenated features", colab_E_concat, prx_edges)
    compare("\n5. Edge embedded features", colab_E_embedded, prx_edge_features)

    # LayerNorm
    print("\n6. LayerNorm...")

    # ColabDesign LayerNorm
    mean_colab = colab_E_embedded.mean(axis=-1, keepdims=True)
    var_colab = colab_E_embedded.var(axis=-1, keepdims=True)
    colab_E_normed = (colab_E_embedded - mean_colab) / jnp.sqrt(var_colab + 1e-5)

    scale = params['protein_mpnn/~/protein_features/~/norm_edges']['scale']
    offset = params['protein_mpnn/~/protein_features/~/norm_edges']['offset']
    colab_E_normed = colab_E_normed * scale + offset

    # PrxteinMPNN LayerNorm (using vmap)
    def norm_fn(x):
        mean = x.mean()
        var = x.var()
        return ((x - mean) / jnp.sqrt(var + 1e-5)) * scale + offset
    prx_E_normed = jax.vmap(jax.vmap(norm_fn))(prx_edge_features)

    compare("\n6. After LayerNorm", colab_E_normed, prx_E_normed)

    # W_e projection
    print("\n7. W_e projection...")

    # ColabDesign W_e weights
    w_e_proj_colab = params['protein_mpnn/~/W_e']['w']  # (128, 128)
    b_e_proj = params['protein_mpnn/~/W_e']['b']

    # PrxteinMPNN loads them transposed (see load_weights_comprehensive.py line 103)
    w_e_proj_prx = w_e_proj_colab.T  # (128, 128)

    # ColabDesign forward: input @ W_e + b
    colab_E_proj = colab_E_normed @ w_e_proj_colab + b_e_proj

    # PrxteinMPNN forward with eqx.nn.Linear: input @ weight.T + bias
    # Since weight = W_e.T, this becomes: input @ (W_e.T).T + b = input @ W_e + b
    # So we should use the SAME operation!
    prx_E_proj_correct = colab_E_normed @ w_e_proj_colab + b_e_proj  # Should match!

    # What the vmap version currently does (if weight is stored as W_e.T):
    prx_E_proj_vmap = jax.vmap(jax.vmap(lambda x: x @ w_e_proj_prx.T + b_e_proj))(prx_E_normed)

    compare("\n7a. After W_e projection (direct)", colab_E_proj, prx_E_proj_correct)
    compare("\n7b. After W_e projection (vmap with loaded weight)", colab_E_proj, prx_E_proj_vmap)

    print("\n" + "="*80)
    print("KEY FINDING")
    print("="*80)
    print("If LayerNorm or W_e_proj show divergence, that's the root cause!")


if __name__ == "__main__":
    main()
