"""
Detailed feature extraction comparison.
This breaks down the feature extraction step-by-step to find the exact divergence point.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import pearsonr
import joblib

from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import Protein
from load_weights_comprehensive import load_prxteinmpnn_with_colabdesign_weights


def compare(name, arr1, arr2):
    """Compare two arrays."""
    a1, a2 = np.array(arr1).flatten(), np.array(arr2).flatten()
    valid = np.isfinite(a1) & np.isfinite(a2)
    if not valid.all():
        a1, a2 = a1[valid], a2[valid]
    corr = pearsonr(a1, a2)[0] if len(a1) > 1 else 0.0
    max_diff = np.abs(a1 - a2).max()
    status = "✅" if (corr > 0.99 and max_diff < 0.001) else ("🟡" if corr > 0.90 else "❌")
    print(f"  {status} {name}: Corr={corr:.6f}, Max diff={max_diff:.6f}")
    return corr


def colabdesign_get_cb(Y):
    """Compute Cb from N, Ca, C following ColabDesign."""
    b = Y[1] - Y[0]
    c = Y[2] - Y[1]
    Cb = -0.58273431 * jnp.cross(b, c) + 0.56802827 * b - 0.54067466 * c + Y[1]
    return Cb


def main():
    print("="*80)
    print("DETAILED FEATURE EXTRACTION COMPARISON")
    print("="*80)

    # Load protein
    pdb_path = "tests/data/1ubq.pdb"
    protein_tuple = next(parse_input(pdb_path))
    protein = Protein.from_tuple(protein_tuple)

    # Load models
    key = jax.random.PRNGKey(42)
    colab_weights_path = "/tmp/ColabDesign/colabdesign/mpnn/weights/v_48_020.pkl"
    prx_model = load_prxteinmpnn_with_colabdesign_weights(colab_weights_path, key=key)
    params = joblib.load(colab_weights_path)['model_state_dict']

    # Extract only backbone atoms (N, CA, C, O) like ColabDesign does
    from prxteinmpnn.utils.residue_constants import atom_order
    backbone_indices = [atom_order["N"], atom_order["CA"], atom_order["C"], atom_order["O"]]
    X = protein.coordinates[:, backbone_indices, :]  # (L, 4, 3)

    mask = protein.mask
    residue_idx = protein.residue_index
    chain_idx = protein.chain_index

    print("\n1. BACKBONE COORDINATES")
    print("   ColabDesign uses: N, Ca, C, O, Cb (computed)")
    print("   PrxteinMPNN uses: Extracted N, CA, C, O from atom37, then computes Cb")
    print(f"   Input shape: {X.shape}")

    # ColabDesign processing
    Y_colab = X.swapaxes(0, 1)  # (L, 4, 3) -> (4, L, 3)
    if Y_colab.shape[0] == 4:
        Cb_colab = colabdesign_get_cb(Y_colab)
        Y_colab = jnp.concatenate([Y_colab, Cb_colab[None]], 0)

    # PrxteinMPNN uses coordinates[:, :5, :] after adding Cb
    print(f"   ColabDesign Y shape: {Y_colab.shape}")  # (5, L, 3)
    print(f"   PrxteinMPNN X shape: {X.shape}")  # (L, 4, 3)

    print("\n2. NEIGHBOR INDICES (K-NN)")
    # ColabDesign
    mask_2D_colab = mask[...,None,:] * mask[...,:,None]
    dX_colab = Y_colab[1][...,None,:,:] - Y_colab[1][...,:,None,:]  # Ca-Ca distances
    D_colab = jnp.sqrt(jnp.square(dX_colab).sum(-1) + 1e-6)
    D_masked_colab = jnp.where(mask_2D_colab, D_colab, D_colab.max(-1, keepdims=True))
    E_idx_colab = jax.lax.approx_min_k(D_masked_colab, 48, reduction_dimension=-1)[1]

    # PrxteinMPNN (needs full atom37 coordinates)
    prx_edge_features, E_idx_prx, _ = prx_model.features(
        key, protein.coordinates, mask, residue_idx, chain_idx, None
    )

    compare("Neighbor indices", E_idx_colab, E_idx_prx)

    print("\n3. RBF FEATURES")
    print("   Computing RBF for 25 atom pairs...")

    # ColabDesign RBF
    def colabdesign_rbf(D):
        D_min, D_max, D_count = 2., 22., 16
        D_mu = jnp.linspace(D_min, D_max, D_count)
        D_sigma = (D_max - D_min) / D_count
        return jnp.exp(-((D[...,None] - D_mu) / D_sigma)**2)

    def colabdesign_get_rbf(A, B, E_idx):
        D = jnp.sqrt(jnp.square(A[...,:,None,:] - B[...,None,:,:]).sum(-1) + 1e-6)
        D_neighbors = jnp.take_along_axis(D, E_idx, 1)
        return colabdesign_rbf(D_neighbors)

    edges_pairs = jnp.array([[1,1],[0,0],[2,2],[3,3],[4,4],
                              [1,0],[1,2],[1,3],[1,4],[0,2],
                              [0,3],[0,4],[4,2],[4,3],[3,2],
                              [0,1],[2,1],[3,1],[4,1],[2,0],
                              [3,0],[4,0],[2,4],[3,4],[2,3]])

    RBF_colab = jax.vmap(lambda x: colabdesign_get_rbf(Y_colab[x[0]], Y_colab[x[1]], E_idx_colab))(edges_pairs)
    RBF_colab = RBF_colab.transpose((1, 2, 0, 3))
    RBF_colab_flat = RBF_colab.reshape(RBF_colab.shape[:-2] + (-1,))

    print(f"   ColabDesign RBF shape: {RBF_colab_flat.shape}")  # (L, K, 400)

    # PrxteinMPNN RBF - we need to extract this from the model
    # Let's manually compute it to compare
    from prxteinmpnn.utils.coordinates import compute_backbone_coordinates
    from prxteinmpnn.utils.radial_basis import compute_radial_basis

    backbone_prx = compute_backbone_coordinates(protein.coordinates)
    RBF_prx = compute_radial_basis(backbone_prx, E_idx_prx)

    print(f"   PrxteinMPNN RBF shape: {RBF_prx.shape}")  # (L, K, 400)

    # Check a specific pair (Ca-Ca)
    rbf_ca_ca_colab = RBF_colab[:, :, 0, :]  # First pair is [1,1] = Ca-Ca
    rbf_ca_ca_prx = RBF_prx[:, :, :16]  # First 16 are Ca-Ca

    compare("RBF (Ca-Ca distances)", rbf_ca_ca_colab, rbf_ca_ca_prx)
    compare("RBF (all pairs)", RBF_colab_flat, RBF_prx)

    print("\n4. POSITIONAL ENCODING")
    # ColabDesign
    offset_colab = residue_idx[:,None] - residue_idx[None,:]
    offset_colab = jnp.take_along_axis(offset_colab, E_idx_colab, 1)
    E_chains_colab = (chain_idx[:,None] == chain_idx[None,:]).astype(int)
    E_chains_colab = jnp.take_along_axis(E_chains_colab, E_idx_colab, 1)

    max_rel = 32
    d_colab = jnp.clip(offset_colab + max_rel, 0, 2*max_rel) * E_chains_colab + (1 - E_chains_colab) * (2*max_rel + 1)
    d_onehot_colab = jax.nn.one_hot(d_colab, 2*max_rel + 1 + 1)

    # Apply linear layer
    w_pos = params['protein_mpnn/~/protein_features/~/positional_encodings/~/embedding_linear']['w']
    b_pos = params['protein_mpnn/~/protein_features/~/positional_encodings/~/embedding_linear']['b']
    E_positional_colab = d_onehot_colab @ w_pos + b_pos

    print(f"   ColabDesign positional encoding shape: {E_positional_colab.shape}")

    # PrxteinMPNN
    from prxteinmpnn.utils.graph import compute_neighbor_offsets

    offset_prx = compute_neighbor_offsets(residue_idx, E_idx_prx)
    E_chains_prx = (chain_idx[:, None] == chain_idx[None, :]).astype(int)
    E_chains_prx = jnp.take_along_axis(E_chains_prx, E_idx_prx, axis=1)

    neighbor_offset_factor = jnp.clip(offset_prx + 32, 0, 2*32)
    edge_chain_factor = (1 - E_chains_prx) * (2*32 + 1)
    encoded_offset = neighbor_offset_factor * E_chains_prx + edge_chain_factor
    encoded_offset_one_hot = jax.nn.one_hot(encoded_offset, 2*32 + 2)

    # Apply same weights
    E_positional_prx = encoded_offset_one_hot @ w_pos + b_pos

    print(f"   PrxteinMPNN positional encoding shape: {E_positional_prx.shape}")

    # But wait, PrxteinMPNN uses vmap! Let me check if that makes a difference
    E_positional_prx_vmap = jax.vmap(jax.vmap(lambda x: x @ w_pos + b_pos))(encoded_offset_one_hot)

    compare("Positional encoding (no vmap)", E_positional_colab, E_positional_prx)
    compare("Positional encoding (with vmap)", E_positional_colab, E_positional_prx_vmap)

    print("\n5. EDGE CONCATENATION")
    E_concat_colab = jnp.concatenate((E_positional_colab, RBF_colab_flat), -1)
    E_concat_prx = jnp.concatenate([E_positional_prx_vmap, RBF_prx], axis=-1)

    print(f"   ColabDesign concatenated shape: {E_concat_colab.shape}")
    print(f"   PrxteinMPNN concatenated shape: {E_concat_prx.shape}")

    compare("Concatenated edges", E_concat_colab, E_concat_prx)

    print("\n6. EDGE EMBEDDING (w_e)")
    w_e = params['protein_mpnn/~/protein_features/~/edge_embedding']['w']
    E_embed_colab = E_concat_colab @ w_e
    E_embed_prx = E_concat_prx @ w_e  # Without vmap
    E_embed_prx_vmap = jax.vmap(jax.vmap(lambda x: x @ w_e))(E_concat_prx)

    compare("Edge embedding (no vmap)", E_embed_colab, E_embed_prx)
    compare("Edge embedding (with vmap)", E_embed_colab, E_embed_prx_vmap)

    print("\n7. EDGE NORMALIZATION")
    scale = params['protein_mpnn/~/protein_features/~/norm_edges']['scale']
    offset_norm = params['protein_mpnn/~/protein_features/~/norm_edges']['offset']

    # ColabDesign LayerNorm
    mean_colab = E_embed_colab.mean(axis=-1, keepdims=True)
    var_colab = E_embed_colab.var(axis=-1, keepdims=True)
    E_norm_colab = (E_embed_colab - mean_colab) / jnp.sqrt(var_colab + 1e-5)
    E_norm_colab = E_norm_colab * scale + offset_norm

    # PrxteinMPNN uses vmap
    def norm_fn(x):
        mean = x.mean()
        var = x.var()
        return ((x - mean) / jnp.sqrt(var + 1e-5)) * scale + offset_norm

    E_norm_prx_vmap = jax.vmap(jax.vmap(norm_fn))(E_embed_prx_vmap)

    compare("Edge normalization", E_norm_colab, E_norm_prx_vmap)

    print("\n8. FINAL COMPARISON WITH PRXTEINMPNN OUTPUT")
    compare("PrxteinMPNN edge features", E_norm_colab, prx_edge_features)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("Check correlations above to identify where divergence starts")


if __name__ == "__main__":
    main()
