import sys
import os
from pathlib import Path
import numpy as np
import torch
import jax
import jax.numpy as jnp
import equinox as eqx

# Setup paths
PRXTEIN_PATH = Path("/Users/mar/Projects/united_workspace/PrxteinMPNN/src")
REFERENCE_PATH = Path("/Users/mar/Projects/united_workspace/reference_ligandmpnn_clone")
sys.path.insert(0, str(PRXTEIN_PATH))
sys.path.insert(0, str(REFERENCE_PATH))

from prxteinmpnn.model.mpnn import PrxteinMPNN
import model_utils

def test_full_model_parity():
    print("=" * 60)
    print("FULL MODEL LOGITS PARITY TEST")
    print("=" * 60)
    
    # Config
    hidden_dim = 128
    num_layers = 3
    
    # 1. Load PyTorch model
    print("Loading PyTorch model...")
    pt_checkpoint_path = REFERENCE_PATH / "model_params/proteinmpnn_v_48_020.pt"
    checkpoint = torch.load(pt_checkpoint_path, map_location="cpu")
    
    pt_model = model_utils.ProteinMPNN(
        num_letters=21,
        node_features=hidden_dim,
        edge_features=hidden_dim,
        hidden_dim=hidden_dim,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        k_neighbors=48,
    )
    pt_model.load_state_dict(checkpoint["model_state_dict"])
    pt_model.eval()
    
    # 2. Load JAX model
    print("Loading JAX model...")
    jax_checkpoint_path = Path("/Users/mar/Projects/united_workspace/PrxteinMPNN/model_params/proteinmpnn_v_48_020_converted.eqx")
    
    key = jax.random.PRNGKey(0)
    jax_model = PrxteinMPNN(
        node_features=hidden_dim,
        edge_features=hidden_dim,
        hidden_features=hidden_dim,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        k_neighbors=48,
        dropout_rate=0.0,
        key=key,
    )
    jax_model = eqx.tree_deserialise_leaves(jax_checkpoint_path, jax_model)
    
    # Verify some weights
    jax_w1 = jax_model.encoder.layers[0].edge_message_mlp.layers[0].weight
    pt_w1 = pt_model.encoder_layers[0].W1.weight.detach().numpy()
    print(f"JAX W1 weight shape: {jax_w1.shape}, sum: {np.sum(jax_w1):.4f}")
    print(f"PT W1 weight shape: {pt_w1.shape}, sum: {np.sum(pt_w1):.4f}")
    print(f"Weight diff: {np.max(np.abs(jax_w1 - pt_w1)):.2e}")
    
    # 3. Create identical inputs
    print("Creating inputs...")
    batch_size = 1
    seq_len = 20
    np.random.seed(42)
    
    # X: [B, L, 4, 3]
    X_np = np.random.randn(batch_size, seq_len, 4, 3).astype(np.float32)
    # S: [B, L]
    S_np = np.random.randint(0, 21, (batch_size, seq_len)).astype(np.int64)
    # mask: [B, L]
    mask_np = np.ones((batch_size, seq_len), dtype=np.float32)
    # chain_M: [B, L]
    chain_M_np = np.ones((batch_size, seq_len), dtype=np.float32)
    # residue_idx: [B, L]
    residue_idx_np = np.tile(np.arange(seq_len), (batch_size, 1)).astype(np.int64)
    # chain_encoding_all: [B, L]
    chain_encoding_all_np = np.zeros((batch_size, seq_len), dtype=np.int64)
    # randn: [B, L]
    randn_np = np.random.randn(batch_size, seq_len).astype(np.float32)
    
    # Convert JAX inputs (no batch dim)
    # PrxteinMPNN expects Atom37 format (N=0, CA=1, C=2, CB=3, O=4)
    # PyTorch inputs are (N=0, CA=1, C=2, O=3)
    X_jax_37 = jnp.zeros((seq_len, 37, 3))
    # N, CA, C
    X_jax_37 = X_jax_37.at[:, 0, :].set(X_np[0, :, 0, :])
    X_jax_37 = X_jax_37.at[:, 1, :].set(X_np[0, :, 1, :])
    X_jax_37 = X_jax_37.at[:, 2, :].set(X_np[0, :, 2, :])
    # O is at index 4 in Atom37
    X_jax_37 = X_jax_37.at[:, 4, :].set(X_np[0, :, 3, :])
    
    X_jax = X_jax_37
    S_jax = jnp.array(S_np[0])
    mask_jax = jnp.array(mask_np[0])
    chain_M_jax = jnp.array(chain_M_np[0])
    residue_idx_jax = jnp.array(residue_idx_np[0])
    chain_encoding_all_jax = jnp.array(chain_encoding_all_np[0])

    # Convert PyTorch inputs
    X_pt = torch.from_numpy(X_np)
    S_pt = torch.from_numpy(S_np)
    mask_pt = torch.from_numpy(mask_np)
    chain_M_pt = torch.from_numpy(chain_M_np)
    residue_idx_pt = torch.from_numpy(residue_idx_np)
    chain_encoding_all_pt = torch.from_numpy(chain_encoding_all_np)
    
    # Compute decoding order and AR mask (for JAX parity)
    # This must match PyTorch's decoding_order logic
    # decoding_order = torch.argsort((chain_mask + 0.0001) * (torch.abs(randn)))
    # We'll use a fixed order for parity simplicity: range(L)
    decoding_order_np = np.argsort(np.abs(randn_np[0])) # [L]
    
    # order_mask_backward: [L, L] where [i, j] = 1 if j is before i in decoding_order
    ar_mask_np = np.zeros((seq_len, seq_len), dtype=np.int32)
    for i in range(seq_len):
        for j in range(seq_len):
            # i_idx and j_idx are positions in the sequence
            # we want to know if j is before i in decoding_order
            pos_i = np.where(decoding_order_np == i)[0][0]
            pos_j = np.where(decoding_order_np == j)[0][0]
            if pos_j < pos_i:
                ar_mask_np[i, j] = 1
    
    ar_mask_jax = jnp.array(ar_mask_np)

    # Convert PyTorch inputs in feature dict
    # Re-use the same randn for PyTorch
    randn_pt = torch.from_numpy(randn_np)
    feature_dict = {
        "X": X_pt,
        "S": S_pt,
        "mask": mask_pt,
        "chain_mask": chain_M_pt,
        "R_idx": residue_idx_pt,
        "chain_labels": chain_encoding_all_pt,
        "randn": randn_pt,
        "batch_size": batch_size,
        "symmetry_residues": [[]],
        "symmetry_weights": [[]],
    }
    
    # 4. Feature Parity
    print("Checking Feature parity...")
    # PT Features
    with torch.no_grad():
        pt_features_dict = pt_model.features(feature_dict)
        if isinstance(pt_features_dict, tuple):
            E_pt, E_idx_pt = pt_features_dict
        else:
            V_pt, E_pt, E_idx_pt = pt_features_dict
        
    # JAX Features
    key, feat_key = jax.random.split(key)
    
    # 4a. RBF Check
    from prxteinmpnn.utils.radial_basis import compute_radial_basis
    from prxteinmpnn.utils.coordinates import compute_backbone_coordinates
    
    bb_jax = compute_backbone_coordinates(X_jax)
    # PyTorch top_k picks different neighbors if distances are identical, but we used random coords so should be unique.
    # We need to ensure E_idx matches for RBF comparison.
    E_idx_jax_fix = jnp.array(E_idx_pt.numpy()[0], dtype=jnp.int32)
    rbf_jax = compute_radial_basis(bb_jax, E_idx_jax_fix)
    
    # Need to get PT RBF
    # ProteinFeatures.forward:
    # rbf = self._rbf(D_neighbors, E_idx)
    # distance = torch.sqrt(torch.sum((X[:,:,None,:] - X[:,None,:,:])**2,-1) + 1e-6)
    
    # Let's just run the full features and compare weights/outputs
    jax_features_out = jax_model.features(
        feat_key,
        X_jax,
        mask_jax,
        residue_idx_jax,
        chain_encoding_all_jax,
        backbone_noise=0.0,
        neighbor_indices=E_idx_jax_fix # Force neighbor match
    )
    E_jax_feat, E_idx_jax_feat, _, _ = jax_features_out

    E_pt_np = E_pt.numpy()[0]
    E_jax_np = np.array(E_jax_feat)
    
    print(f"Edge Features diff (forced indices): {np.max(np.abs(E_pt_np - E_jax_np)):.2e}")
    
    # Check intermediate linear outputs in features
    # w_pos output
    # w_e output
    # norm_edges output
    
    # Let's check RBF directly if possible
    # We'll need to reach into the PT model or replicate its RBF
    def pt_rbf(D, num_rbf=16):
        D_min, D_max, D_count = 0., 20., num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count)
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        return torch.exp(-((D.unsqueeze(-1) - D_mu) / D_sigma)**2)

    # PT distances
    X_pt_bb = X_pt[:,:,1,:] # CA only for ProteinMPNN
    D_pt = torch.sqrt(torch.sum((X_pt_bb[:,:,None,:] - X_pt_bb[:,None,:,:])**2,-1) + 1e-6)
    D_neighbors_pt = torch.gather(D_pt, 2, E_idx_pt)
    rbf_pt = pt_rbf(D_neighbors_pt).numpy()[0]
    
    # PrxteinMPNN's 400 features = 25 pairs * 16 RBF
    # The first 16 features in JAX rbf should correspond to CA-CA RBF (index [1,1] in BACKBONE_PAIRS)
    # Let's verify BACKBONE_PAIRS[0] is [1,1]
    from prxteinmpnn.utils.radial_basis import BACKBONE_PAIRS
    print(f"First pair: {BACKBONE_PAIRS[0]}") # Should be [1,1]
    
    rbf_jax_caca = np.array(rbf_jax)[:, :, :16]
    
    print(f"CA-CA RBF diff: {np.max(np.abs(rbf_pt - rbf_jax_caca)):.2e}")
    
    # Check Positional Encodings
    # self.w_pos(encoded_offset_one_hot)
    # PyTorch: self.embeddings(E_idx)
    # PositionalEncodings is a bit different

    # 5. Forward pass
    print("Running full forward passes...")
    with torch.no_grad():
        pt_out = pt_model.score(feature_dict, use_sequence=True)
        log_probs_pt = pt_out["log_probs"]
        
        # Intermediate Encoder state
        h_V_pt, h_E_pt, E_idx_pt_enc = pt_model.encode(feature_dict)
    
    # JAX intermediate
    prng_key, encoder_key = jax.random.split(key)
    edge_features_jax, neighbor_indices_jax, node_features_jax, _ = jax_model.features(
      feat_key,
      X_jax,
      mask_jax,
      residue_idx_jax,
      chain_encoding_all_jax,
      backbone_noise=0.0
    )
    h_V_jax_enc, h_E_jax_enc = jax_model.encoder(
      edge_features_jax,
      neighbor_indices_jax,
      mask_jax,
      node_features_jax,
      key=encoder_key,
    )

    print(f"Encoder h_V diff: {np.max(np.abs(h_V_pt.numpy()[0] - np.array(h_V_jax_enc))):.2e}")
    print(f"Encoder h_E diff: {np.max(np.abs(h_E_pt.numpy()[0] - np.array(h_E_jax_enc))):.2e}")

    # JAX model expects: (coords, mask, residue_idx, chain_idx, approach, ...)
    # and returns: (sequence, logits)
    _, logits_jax = jax_model(
        X_jax, 
        mask_jax, 
        residue_idx_jax, 
        chain_encoding_all_jax, 
        "conditional",
        one_hot_sequence=jax.nn.one_hot(S_jax, 21),
        ar_mask=ar_mask_jax,
    )
    
    # Convert JAX logits to log_probs
    import jax.nn as jnn
    log_probs_jax = jnn.log_softmax(logits_jax, axis=-1)
    
    # 5. Compare
    log_probs_pt_np = log_probs_pt.numpy()[0]
    log_probs_jax_np = np.array(log_probs_jax)
    
    print(f"\nLogits shape: PyTorch {log_probs_pt_np.shape}, JAX {log_probs_jax_np.shape}")
    
    max_diff = np.max(np.abs(log_probs_pt_np - log_probs_jax_np))
    mean_diff = np.mean(np.abs(log_probs_pt_np - log_probs_jax_np))
    
    print(f"Max difference: {max_diff:.2e}")
    print(f"Mean difference: {mean_diff:.2e}")
    
    # Calculate Correlation
    from scipy.stats import pearsonr
    corr_logits, _ = pearsonr(log_probs_pt_np.flatten(), log_probs_jax_np.flatten())
    
    probs_pt = np.exp(log_probs_pt_np)
    probs_jax = np.exp(log_probs_jax_np)
    corr_probs, _ = pearsonr(probs_pt.flatten(), probs_jax.flatten())
    
    print(f"Logit Correlation: {corr_logits:.4f}")
    print(f"Probability Correlation: {corr_probs:.4f}")
    
    if corr_logits >= 0.95:
        print(f"\n\u2713 PASSED - Correlation is {corr_logits:.4f} (>= 0.95)")
        return True
    else:
        print(f"\n\u2717 FAILED - Correlation is {corr_logits:.4f} (< 0.95)")
        return False

if __name__ == "__main__":
    success = test_full_model_parity()
    sys.exit(0 if success else 1)
