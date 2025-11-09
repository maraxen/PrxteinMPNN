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
from prxteinmpnn.utils.concatenate import concatenate_neighbor_nodes
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


def colabdesign_forward(X, mask, residue_idx, chain_idx, params, k_neighbors=48, return_intermediates=False):
    """Complete forward pass following ColabDesign EXACTLY."""
    intermediates = {}

    # Features
    E, E_idx = colabdesign_features(X, mask, residue_idx, chain_idx, params, k_neighbors)
    intermediates['features_E'] = E

    # W_e projection
    w = params['protein_mpnn/~/W_e']['w']
    b = params['protein_mpnn/~/W_e']['b']
    h_E = E @ w + b
    intermediates['h_E_init'] = h_E

    h_V = jnp.zeros((E.shape[0], E.shape[-1]))
    intermediates['h_V_init'] = h_V

    # Encoder
    mask_attend = jnp.take_along_axis(mask[:,None] * mask[None,:], E_idx, 1)

    for i in range(3):
        h_V, h_E = colabdesign_encoder_layer(h_V, h_E, E_idx, mask, mask_attend, params, i)
        intermediates[f'encoder_{i}_h_V'] = h_V
        intermediates[f'encoder_{i}_h_E'] = h_E

    # Build decoder context
    def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
        h_nodes_gathered = h_nodes[E_idx]
        return jnp.concatenate([h_neighbors, h_nodes_gathered], -1)

    h_EX_encoder = cat_neighbors_nodes(jnp.zeros_like(h_V), h_E, E_idx)
    h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
    intermediates['decoder_context'] = h_EXV_encoder

    # Decoder
    for i in range(3):
        h_V = colabdesign_decoder_layer(h_V, h_EXV_encoder, mask, params, i)
        intermediates[f'decoder_{i}_h_V'] = h_V

    # Output
    w = params['protein_mpnn/~/W_out']['w']
    b = params['protein_mpnn/~/W_out']['b']
    logits = h_V @ w + b
    intermediates['logits'] = logits

    if return_intermediates:
        return logits, E, E_idx, h_EXV_encoder, intermediates
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

    print("\n1. Running ColabDesign (pure JAX) with intermediates...")
    print(f"   Using backbone atoms only: {X_backbone.shape}")
    colab_logits, colab_E, colab_E_idx, colab_context, colab_inter = colabdesign_forward(
        X_backbone,
        protein.mask,
        protein.residue_index,
        protein.chain_index,
        params,
        k_neighbors=48,
        return_intermediates=True,
    )

    print("\n2. Running PrxteinMPNN with intermediates...")

    # Extract PrxteinMPNN intermediates by manually running layers
    prx_edge_features, prx_neighbor_indices, _ = prx_model.features(
        key, protein.coordinates, protein.mask,
        protein.residue_index, protein.chain_index, None,
    )

    # Run encoder manually to extract intermediates
    prx_node_features = jnp.zeros((prx_edge_features.shape[0], prx_model.encoder.node_feature_dim))
    mask_2d = protein.mask[:, None] * protein.mask[None, :]
    prx_mask_attend = jnp.take_along_axis(mask_2d, prx_neighbor_indices, axis=1)

    prx_inter = {}
    prx_inter['h_V_init'] = prx_node_features
    prx_inter['h_E_init'] = prx_edge_features

    for i, layer in enumerate(prx_model.encoder.layers):
        prx_node_features, prx_edge_features = layer(
            prx_node_features, prx_edge_features, prx_neighbor_indices,
            protein.mask, prx_mask_attend
        )
        prx_inter[f'encoder_{i}_h_V'] = prx_node_features
        prx_inter[f'encoder_{i}_h_E'] = prx_edge_features

    # Build decoder context (same as Decoder.__call__)
    zeros_with_edges = concatenate_neighbor_nodes(
        jnp.zeros_like(prx_node_features),
        prx_edge_features,
        prx_neighbor_indices,
    )
    prx_context = concatenate_neighbor_nodes(
        prx_node_features,
        zeros_with_edges,
        prx_neighbor_indices,
    )
    prx_inter['decoder_context'] = prx_context

    # Run decoder manually
    prx_node_features_dec = prx_node_features
    for i, layer in enumerate(prx_model.decoder.layers):
        prx_node_features_dec = layer(prx_node_features_dec, prx_context, protein.mask)
        prx_inter[f'decoder_{i}_h_V'] = prx_node_features_dec

    # Get final logits
    prx_logits = jax.vmap(prx_model.w_out)(prx_node_features_dec)
    prx_inter['logits'] = prx_logits

    print("\n" + "="*80)
    print("3. LAYER-BY-LAYER COMPARISONS")
    print("="*80)

    print("\n📊 NEIGHBOR INDICES:")
    compare("Neighbor indices", colab_E_idx, prx_neighbor_indices)

    print("\n📊 INITIAL EDGE FEATURES (after W_e/w_e_proj):")
    compare("Initial h_E", colab_inter['h_E_init'], prx_inter['h_E_init'])

    # Debug: Check edge features before LayerNorm
    print("\n  🔍 Debugging initial edge features:")
    # Get features before norm from ColabDesign
    colab_E_before_norm, _ = colabdesign_features(X_backbone, protein.mask,
                                                   protein.residue_index, protein.chain_index,
                                                   params, k_neighbors=48)
    w_e = params['protein_mpnn/~/W_e']['w']
    b_e = params['protein_mpnn/~/W_e']['b']
    colab_E_after_we = colab_E_before_norm @ w_e + b_e

    # Get PrxteinMPNN features before the final norm
    from prxteinmpnn.utils.coordinates import compute_backbone_coordinates
    from prxteinmpnn.utils.radial_basis import compute_radial_basis
    from prxteinmpnn.utils.graph import compute_neighbor_offsets

    noised_coords = protein.coordinates  # No noise for this test
    backbone_prx = compute_backbone_coordinates(noised_coords)
    rbf_prx = compute_radial_basis(backbone_prx, prx_neighbor_indices)
    neighbor_offsets = compute_neighbor_offsets(protein.residue_index, prx_neighbor_indices)

    edge_chains = (protein.chain_index[:, None] == protein.chain_index[None, :]).astype(int)
    edge_chains_neighbors = jnp.take_along_axis(edge_chains, prx_neighbor_indices, axis=1)
    neighbor_offset_factor = jnp.clip(neighbor_offsets + 32, 0, 2*32)
    edge_chain_factor = (1 - edge_chains_neighbors) * (2*32 + 1)
    encoded_offset = neighbor_offset_factor * edge_chains_neighbors + edge_chain_factor
    encoded_offset_one_hot = jax.nn.one_hot(encoded_offset, 2*32 + 2)

    # Get w_pos from features module
    w_pos = params['protein_mpnn/~/protein_features/~/positional_encodings/~/embedding_linear']['w']
    b_pos = params['protein_mpnn/~/protein_features/~/positional_encodings/~/embedding_linear']['b']
    encoded_positions = jax.vmap(jax.vmap(lambda x: x @ w_pos + b_pos))(encoded_offset_one_hot)

    edges_prx = jnp.concatenate([encoded_positions, rbf_prx], axis=-1)

    # Get w_e from features
    w_e_feat = params['protein_mpnn/~/protein_features/~/edge_embedding']['w']
    edge_features_prx = jax.vmap(jax.vmap(lambda x: x @ w_e_feat))(edges_prx)

    compare("  Before LayerNorm (after edge embedding)", colab_E_after_we, edge_features_prx)

    # Check LayerNorm computation
    scale_norm = params['protein_mpnn/~/protein_features/~/norm_edges']['scale']
    offset_norm = params['protein_mpnn/~/protein_features/~/norm_edges']['offset']

    # ColabDesign LayerNorm (axis=-1)
    mean_colab = colab_E_after_we.mean(axis=-1, keepdims=True)
    var_colab = colab_E_after_we.var(axis=-1, keepdims=True)
    colab_normed = (colab_E_after_we - mean_colab) / jnp.sqrt(var_colab + 1e-5)
    colab_normed = colab_normed * scale_norm + offset_norm

    # PrxteinMPNN LayerNorm (with vmap)
    def norm_fn(x):
        mean = x.mean()
        var = x.var()
        return ((x - mean) / jnp.sqrt(var + 1e-5)) * scale_norm + offset_norm
    prx_normed = jax.vmap(jax.vmap(norm_fn))(edge_features_prx)

    compare("  After LayerNorm (ColabDesign style)", colab_normed, colab_inter['h_E_init'])
    compare("  After LayerNorm (PrxteinMPNN style)", prx_normed, prx_inter['h_E_init'])

    print("\n📊 INITIAL NODE FEATURES:")
    compare("Initial h_V", colab_inter['h_V_init'], prx_inter['h_V_init'])

    print("\n📊 ENCODER LAYERS:")
    for i in range(3):
        print(f"\n  Layer {i}:")
        compare(f"  Encoder {i} h_V", colab_inter[f'encoder_{i}_h_V'], prx_inter[f'encoder_{i}_h_V'])
        compare(f"  Encoder {i} h_E", colab_inter[f'encoder_{i}_h_E'], prx_inter[f'encoder_{i}_h_E'])

    print("\n📊 DECODER CONTEXT:")
    compare("Decoder context", colab_inter['decoder_context'], prx_inter['decoder_context'])

    print("\n📊 DECODER LAYERS:")
    for i in range(3):
        print(f"\n  Layer {i}:")
        compare(f"  Decoder {i} h_V", colab_inter[f'decoder_{i}_h_V'], prx_inter[f'decoder_{i}_h_V'])

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
