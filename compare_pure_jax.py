"""
Pure JAX implementation of ColabDesign's forward pass.
This allows us to compare every operation with PrxteinMPNN.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import pearsonr
import joblib
import os

from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import Protein
from load_weights_comprehensive import load_prxteinmpnn_with_colabdesign_weights


def compare(name, arr1, arr2, verbose=True):
    """Compare two arrays and return correlation."""
    a1, a2 = np.array(arr1).flatten(), np.array(arr2).flatten()

    # Filter NaN/Inf
    valid = np.isfinite(a1) & np.isfinite(a2)
    if not valid.all():
        a1, a2 = a1[valid], a2[valid]

    if len(a1) == 0:
        if verbose: print(f"  ❌ {name}: No valid values!")
        return 0.0

    corr = pearsonr(a1, a2)[0] if len(a1) > 1 else 0.0
    max_diff = np.abs(a1 - a2).max()
    mean_diff = np.abs(a1 - a2).mean()

    status = "✅" if (corr > 0.99 and max_diff < 0.001) else ("🟡" if corr > 0.90 else "❌")

    if verbose:
        print(f"  {status} {name}:")
        print(f"     Corr: {corr:.6f}, Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")

    return corr


# ============================================================================
# COLABDESIGN OPERATIONS IN PURE JAX
# ============================================================================

def colabdesign_get_cb(Y):
    """Compute Cb from N, Ca, C following ColabDesign."""
    b = Y[1] - Y[0]  # Ca - N
    c = Y[2] - Y[1]  # C - Ca
    Cb = -0.58273431 * jnp.cross(b, c) + 0.56802827 * b - 0.54067466 * c + Y[1]
    return Cb


def colabdesign_get_edge_idx(Ca, mask, top_k_val):
    """Get K-NN indices following ColabDesign exactly."""
    mask_2D = mask[...,None,:] * mask[...,:,None]
    dX = Ca[...,None,:,:] - Ca[...,:,None,:]
    D = jnp.sqrt(jnp.square(dX).sum(-1) + 1e-6)
    D_masked = jnp.where(mask_2D, D, D.max(-1, keepdims=True))
    k = min(top_k_val, Ca.shape[-2])
    return jax.lax.approx_min_k(D_masked, k, reduction_dimension=-1)[1]


def colabdesign_rbf(D, num_rbf=16):
    """RBF following ColabDesign."""
    D_min, D_max, D_count = 2., 22., num_rbf
    D_mu = jnp.linspace(D_min, D_max, D_count)
    D_sigma = (D_max - D_min) / D_count
    return jnp.exp(-((D[...,None] - D_mu) / D_sigma)**2)


def colabdesign_get_rbf(A, B, E_idx):
    """Get RBF for atom pair following ColabDesign."""
    D = jnp.sqrt(jnp.square(A[...,:,None,:] - B[...,None,:,:]).sum(-1) + 1e-6)
    D_neighbors = jnp.take_along_axis(D, E_idx, 1)
    return colabdesign_rbf(D_neighbors)


def colabdesign_features(X, mask, residue_idx, chain_idx, params, k_neighbors=48):
    """Extract features following ColabDesign EXACTLY."""
    # Swap axes: (L, 4, 3) -> (4, L, 3)
    Y = X.swapaxes(0, 1)

    # Add Cb if not present
    if Y.shape[0] == 4:
        Cb = colabdesign_get_cb(Y)
        Y = jnp.concatenate([Y, Cb[None]], 0)

    # Get edge indices based on Ca-Ca distances
    E_idx = colabdesign_get_edge_idx(Y[1], mask, k_neighbors)

    # RBF encode distances between all atom pairs
    edges_pairs = jnp.array([[1,1],[0,0],[2,2],[3,3],[4,4],
                              [1,0],[1,2],[1,3],[1,4],[0,2],
                              [0,3],[0,4],[4,2],[4,3],[3,2],
                              [0,1],[2,1],[3,1],[4,1],[2,0],
                              [3,0],[4,0],[2,4],[3,4],[2,3]])

    RBF_all = jax.vmap(lambda x: colabdesign_get_rbf(Y[x[0]], Y[x[1]], E_idx))(edges_pairs)
    RBF_all = RBF_all.transpose((1, 2, 0, 3))
    RBF_all = RBF_all.reshape(RBF_all.shape[:-2] + (-1,))

    # Position embedding
    offset = residue_idx[:,None] - residue_idx[None,:]
    offset = jnp.take_along_axis(offset, E_idx, 1)

    # Chain index offset
    E_chains = (chain_idx[:,None] == chain_idx[None,:]).astype(int)
    E_chains = jnp.take_along_axis(E_chains, E_idx, 1)

    # Positional encoding
    max_rel = 32
    d = jnp.clip(offset + max_rel, 0, 2*max_rel) * E_chains + (1 - E_chains) * (2*max_rel + 1)
    d_onehot = jax.nn.one_hot(d, 2*max_rel + 1 + 1)

    # Apply linear layer
    w = params['protein_mpnn/~/protein_features/~/positional_encodings/~/embedding_linear']['w']
    b = params['protein_mpnn/~/protein_features/~/positional_encodings/~/embedding_linear']['b']
    E_positional = d_onehot @ w + b

    # Concatenate and embed
    E = jnp.concatenate((E_positional, RBF_all), -1)

    # Edge embedding (w_e)
    w = params['protein_mpnn/~/protein_features/~/edge_embedding']['w']
    E = E @ w

    # Norm edges
    scale = params['protein_mpnn/~/protein_features/~/norm_edges']['scale']
    offset = params['protein_mpnn/~/protein_features/~/norm_edges']['offset']
    E = E * scale + offset  # LayerNorm is just scale + offset in Haiku when applied per-element
    # Actually, LayerNorm normalizes first
    mean = E.mean(axis=-1, keepdims=True)
    var = E.var(axis=-1, keepdims=True)
    E = (E - mean) / jnp.sqrt(var + 1e-5)
    E = E * scale + offset

    return E, E_idx


def colabdesign_encoder_layer(h_V, h_E, E_idx, mask, mask_attend, params, layer_idx, scale=30.0):
    """Single encoder layer following ColabDesign EXACTLY."""
    layer_name = 'enc_layer' if layer_idx == 0 else f'enc_layer_{layer_idx}'
    prefix = f'protein_mpnn/~/{layer_name}/~/enc{layer_idx}'
    dense_prefix = f'protein_mpnn/~/{layer_name}/~/position_wise_feed_forward/~/enc{layer_idx}'

    # cat_neighbors_nodes
    def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
        h_nodes_gathered = h_nodes[E_idx]
        return jnp.concatenate([h_neighbors, h_nodes_gathered], -1)

    # First message passing
    h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
    h_V_expand = jnp.tile(jnp.expand_dims(h_V, -2), [1, h_EV.shape[-2], 1])
    h_EV = jnp.concatenate([h_V_expand, h_EV], -1)

    # W1 -> GELU -> W2 -> GELU -> W3
    w1 = params[f'{prefix}_W1']['w']
    b1 = params[f'{prefix}_W1']['b']
    h = h_EV @ w1 + b1
    h = jax.nn.gelu(h, approximate=False)

    w2 = params[f'{prefix}_W2']['w']
    b2 = params[f'{prefix}_W2']['b']
    h = h @ w2 + b2
    h = jax.nn.gelu(h, approximate=False)

    w3 = params[f'{prefix}_W3']['w']
    b3 = params[f'{prefix}_W3']['b']
    h_message = h @ w3 + b3

    # Apply mask_attend
    if mask_attend is not None:
        h_message = jnp.expand_dims(mask_attend, -1) * h_message

    dh = jnp.sum(h_message, -2) / scale
    h_V = h_V + dh

    # Norm1
    scale_n1 = params[f'{prefix}_norm1']['scale']
    offset_n1 = params[f'{prefix}_norm1']['offset']
    mean = h_V.mean(axis=-1, keepdims=True)
    var = h_V.var(axis=-1, keepdims=True)
    h_V = (h_V - mean) / jnp.sqrt(var + 1e-5)
    h_V = h_V * scale_n1 + offset_n1

    # Dense (feedforward)
    w_in = params[f'{dense_prefix}_dense_W_in']['w']
    b_in = params[f'{dense_prefix}_dense_W_in']['b']
    h_ff = h_V @ w_in + b_in
    h_ff = jax.nn.gelu(h_ff, approximate=False)

    w_out = params[f'{dense_prefix}_dense_W_out']['w']
    b_out = params[f'{dense_prefix}_dense_W_out']['b']
    dh = h_ff @ w_out + b_out

    h_V = h_V + dh

    # Norm2
    scale_n2 = params[f'{prefix}_norm2']['scale']
    offset_n2 = params[f'{prefix}_norm2']['offset']
    mean = h_V.mean(axis=-1, keepdims=True)
    var = h_V.var(axis=-1, keepdims=True)
    h_V = (h_V - mean) / jnp.sqrt(var + 1e-5)
    h_V = h_V * scale_n2 + offset_n2

    # Apply mask_V
    if mask is not None:
        h_V = jnp.expand_dims(mask, -1) * h_V

    # Edge update
    h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
    h_V_expand = jnp.tile(jnp.expand_dims(h_V, -2), [1, h_EV.shape[-2], 1])
    h_EV = jnp.concatenate([h_V_expand, h_EV], -1)

    # W11 -> GELU -> W12 -> GELU -> W13
    w11 = params[f'{prefix}_W11']['w']
    b11 = params[f'{prefix}_W11']['b']
    h = h_EV @ w11 + b11
    h = jax.nn.gelu(h, approximate=False)

    w12 = params[f'{prefix}_W12']['w']
    b12 = params[f'{prefix}_W12']['b']
    h = h @ w12 + b12
    h = jax.nn.gelu(h, approximate=False)

    w13 = params[f'{prefix}_W13']['w']
    b13 = params[f'{prefix}_W13']['b']
    h_message = h @ w13 + b13

    h_E = h_E + h_message

    # Norm3
    scale_n3 = params[f'{prefix}_norm3']['scale']
    offset_n3 = params[f'{prefix}_norm3']['offset']
    mean = h_E.mean(axis=-1, keepdims=True)
    var = h_E.var(axis=-1, keepdims=True)
    h_E = (h_E - mean) / jnp.sqrt(var + 1e-5)
    h_E = h_E * scale_n3 + offset_n3

    return h_V, h_E


def colabdesign_decoder_layer(h_V, h_E, mask, params, layer_idx, scale=30.0):
    """Single decoder layer following ColabDesign EXACTLY."""
    layer_name = 'dec_layer' if layer_idx == 0 else f'dec_layer_{layer_idx}'
    prefix = f'protein_mpnn/~/{layer_name}/~/dec{layer_idx}'
    dense_prefix = f'protein_mpnn/~/{layer_name}/~/position_wise_feed_forward/~/dec{layer_idx}'

    # Concatenate h_V_i to h_E_ij
    h_V_expand = jnp.tile(jnp.expand_dims(h_V, -2), [1, h_E.shape[-2], 1])
    h_EV = jnp.concatenate([h_V_expand, h_E], -1)

    # W1 -> GELU -> W2 -> GELU -> W3
    w1 = params[f'{prefix}_W1']['w']
    b1 = params[f'{prefix}_W1']['b']
    h = h_EV @ w1 + b1
    h = jax.nn.gelu(h, approximate=False)

    w2 = params[f'{prefix}_W2']['w']
    b2 = params[f'{prefix}_W2']['b']
    h = h @ w2 + b2
    h = jax.nn.gelu(h, approximate=False)

    w3 = params[f'{prefix}_W3']['w']
    b3 = params[f'{prefix}_W3']['b']
    h_message = h @ w3 + b3

    dh = jnp.sum(h_message, -2) / scale
    h_V = h_V + dh

    # Norm1
    scale_n1 = params[f'{prefix}_norm1']['scale']
    offset_n1 = params[f'{prefix}_norm1']['offset']
    mean = h_V.mean(axis=-1, keepdims=True)
    var = h_V.var(axis=-1, keepdims=True)
    h_V = (h_V - mean) / jnp.sqrt(var + 1e-5)
    h_V = h_V * scale_n1 + offset_n1

    # Dense (feedforward)
    w_in = params[f'{dense_prefix}_dense_W_in']['w']
    b_in = params[f'{dense_prefix}_dense_W_in']['b']
    h_ff = h_V @ w_in + b_in
    h_ff = jax.nn.gelu(h_ff, approximate=False)

    w_out = params[f'{dense_prefix}_dense_W_out']['w']
    b_out = params[f'{dense_prefix}_dense_W_out']['b']
    dh = h_ff @ w_out + b_out

    h_V = h_V + dh

    # Norm2
    scale_n2 = params[f'{prefix}_norm2']['scale']
    offset_n2 = params[f'{prefix}_norm2']['offset']
    mean = h_V.mean(axis=-1, keepdims=True)
    var = h_V.var(axis=-1, keepdims=True)
    h_V = (h_V - mean) / jnp.sqrt(var + 1e-5)
    h_V = h_V * scale_n2 + offset_n2

    # Apply mask_V
    if mask is not None:
        h_V = jnp.expand_dims(mask, -1) * h_V

    return h_V


def colabdesign_forward(X, mask, residue_idx, chain_idx, params, k_neighbors=48):
    """Complete forward pass following ColabDesign EXACTLY."""
    # Features
    E, E_idx = colabdesign_features(X, mask, residue_idx, chain_idx, params, k_neighbors)

    # W_e projection
    w = params['protein_mpnn/~/W_e']['w']
    b = params['protein_mpnn/~/W_e']['b']
    h_E = E @ w + b

    h_V = jnp.zeros((E.shape[0], E.shape[-1]))

    # Encoder
    mask_attend = jnp.take_along_axis(mask[:,None] * mask[None,:], E_idx, 1)

    for i in range(3):
        h_V, h_E = colabdesign_encoder_layer(h_V, h_E, E_idx, mask, mask_attend, params, i)

    # Build decoder context
    def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
        h_nodes_gathered = h_nodes[E_idx]
        return jnp.concatenate([h_neighbors, h_nodes_gathered], -1)

    h_EX_encoder = cat_neighbors_nodes(jnp.zeros_like(h_V), h_E, E_idx)
    h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

    # Decoder
    for i in range(3):
        h_V = colabdesign_decoder_layer(h_V, h_EXV_encoder, mask, params, i)

    # Output
    w = params['protein_mpnn/~/W_out']['w']
    b = params['protein_mpnn/~/W_out']['b']
    logits = h_V @ w + b

    return logits, E, E_idx, h_EXV_encoder


# ============================================================================
# MAIN COMPARISON
# ============================================================================

def main():
    print("="*80)
    print("PURE JAX COLABDESIGN vs PRXTEINMPNN - STEP BY STEP")
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

    # ColabDesign expects only N, CA, C, O (backbone atoms)
    # Extract from atom37: N=0, CA=1, C=2, O=4 (NOT 3! CB is at 3)
    from prxteinmpnn.utils.residue_constants import atom_order
    backbone_indices = [atom_order["N"], atom_order["CA"], atom_order["C"], atom_order["O"]]  # [0, 1, 2, 4]
    X_backbone = protein.coordinates[:, backbone_indices, :]  # (L, 4, 3)

    print("\n1. Running ColabDesign (pure JAX)...")
    print(f"   Using backbone atoms only: {X_backbone.shape}")
    colab_logits, colab_E, colab_E_idx, colab_context = colabdesign_forward(
        X_backbone,
        protein.mask,
        protein.residue_index,
        protein.chain_index,
        params,
        k_neighbors=48,
    )

    print("\n2. Running PrxteinMPNN...")
    _, prx_logits = prx_model(
        protein.coordinates,
        protein.mask,
        protein.residue_index,
        protein.chain_index,
        "unconditional",
        prng_key=key,
    )

    # Extract PrxteinMPNN intermediates
    prx_edge_features, prx_neighbor_indices, _ = prx_model.features(
        key, protein.coordinates, protein.mask,
        protein.residue_index, protein.chain_index, None,
    )

    print("\n" + "="*80)
    print("3. COMPARISONS")
    print("="*80)

    print("\n📊 NEIGHBOR INDICES:")
    compare("Neighbor indices", colab_E_idx, prx_neighbor_indices)

    print("\n📊 EDGE FEATURES:")
    # ColabDesign: E after features module (before W_e)
    # PrxteinMPNN: edge_features after w_e_proj
    # These should match after applying W_e to colab_E
    w_e = params['protein_mpnn/~/W_e']['w']
    b_e = params['protein_mpnn/~/W_e']['b']
    colab_E_proj = colab_E @ w_e + b_e
    compare("Edge features (after W_e/w_e_proj)", colab_E_proj, prx_edge_features)

    print("\n📊 FINAL LOGITS:")
    final_corr = compare("Final logits", colab_logits, prx_logits)

    print("\n" + "="*80)
    print("4. SUMMARY")
    print("="*80)
    print(f"Final correlation: {final_corr:.6f}")
    if final_corr < 0.90:
        print("❌ DIVERGENCE FOUND - Check outputs above to see where")
    else:
        print("✅ SUCCESS!")

    # Show sample values
    print(f"\nSample logits (residue 0, first 5):")
    print(f"  ColabDesign: {colab_logits[0, :5]}")
    print(f"  PrxteinMPNN: {prx_logits[0, :5]}")


if __name__ == "__main__":
    main()
