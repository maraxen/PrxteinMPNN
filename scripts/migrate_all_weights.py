#!/usr/bin/env python3
"""Migrate all PrxteinMPNN weights to dropout-enabled architecture.

Defines old model structure inline to avoid import issues and ensure no dropout layers.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from pathlib import Path
import sys
from huggingface_hub import hf_hub_download
from functools import partial

# Configuration
HF_REPO_ID = "maraxen/prxteinmpnn"
NODE_FEATURES = 128
EDGE_FEATURES = 128
HIDDEN_FEATURES = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
K_NEIGHBORS = 48
VOCAB_SIZE = 21
DROPOUT_RATE = 0.1

WEIGHT_TYPES = ["original", "soluble"]
MODEL_VERSIONS = ["v_48_002", "v_48_010", "v_48_020", "v_48_030"]
OUTPUT_DIR = Path("migrated_weights")

# Import necessary utils
from prxteinmpnn.utils.concatenate import concatenate_neighbor_nodes
# We mock ProteinFeatures to avoid import issues and ensure structure matches
# from prxteinmpnn.model.features import ProteinFeatures

# Define OLD model structure inline
LayerNorm = eqx.nn.LayerNorm
_gelu = partial(jax.nn.gelu, approximate=False)

class OldEncoderLayer(eqx.Module):
    edge_message_mlp: eqx.nn.MLP
    norm1: LayerNorm
    dense: eqx.nn.MLP
    norm2: LayerNorm
    edge_update_mlp: eqx.nn.MLP
    norm3: LayerNorm
    node_features_dim: int = eqx.field(static=True)
    edge_features_dim: int = eqx.field(static=True)

    def __init__(self, node_features, edge_features, hidden_features, *, key):
        self.node_features_dim = node_features
        self.edge_features_dim = edge_features
        keys = jax.random.split(key, 7)
        embed_input_size = edge_features + node_features * 2
        
        self.edge_message_mlp = eqx.nn.MLP(
            in_size=embed_input_size,
            out_size=edge_features,
            width_size=hidden_features,
            depth=2,
            activation=_gelu,
            key=keys[3],
        )
        self.norm1 = LayerNorm(node_features)
        self.dense = eqx.nn.MLP(
            in_size=node_features,
            out_size=node_features,
            width_size=embed_input_size + hidden_features,
            depth=1,
            activation=_gelu,
            key=keys[4],
        )
        self.norm2 = LayerNorm(node_features)
        self.edge_update_mlp = eqx.nn.MLP(
            in_size=embed_input_size,
            out_size=edge_features,
            width_size=edge_features,
            depth=2,
            activation=_gelu,
            key=keys[5],
        )
        self.norm3 = LayerNorm(edge_features)

    def __call__(self, node_features, edge_features, neighbor_indices, mask, mask_attend=None, scale=30.0, *, key=None):
        pass

class OldEncoder(eqx.Module):
    layers: tuple[OldEncoderLayer, ...]
    node_feature_dim: int = eqx.field(static=True)

    def __init__(self, node_features, edge_features, hidden_features, num_layers=3, _physics_feature_dim=None, *, key):
        self.node_feature_dim = node_features
        keys = jax.random.split(key, num_layers)
        self.layers = tuple(
            OldEncoderLayer(node_features, edge_features, hidden_features, key=k)
            for k in keys
        )

    def __call__(self, *args, **kwargs):
        pass

class OldDecoderLayer(eqx.Module):
    message_mlp: eqx.nn.MLP
    norm1: LayerNorm
    dense: eqx.nn.MLP
    norm2: LayerNorm

    def __init__(self, node_features, edge_context_features, _hidden_features, *, key):
        keys = jax.random.split(key, 4)
        mlp_input_dim = node_features + edge_context_features
        
        self.message_mlp = eqx.nn.MLP(
            in_size=mlp_input_dim,
            out_size=node_features,
            width_size=node_features,
            depth=2,
            activation=_gelu,
            key=keys[2],
        )
        self.norm1 = LayerNorm(node_features)
        self.dense = eqx.nn.MLP(
            in_size=node_features,
            out_size=node_features,
            width_size=mlp_input_dim,
            depth=1,
            activation=_gelu,
            key=keys[3],
        )
        self.norm2 = LayerNorm(node_features)

    def __call__(self, *args, **kwargs):
        pass

class OldDecoder(eqx.Module):
    layers: tuple[OldDecoderLayer, ...]
    node_features_dim: int = eqx.field(static=True)
    edge_features_dim: int = eqx.field(static=True)

    def __init__(self, node_features, edge_features, hidden_features, num_layers=3, *, key):
        self.node_features_dim = node_features
        self.edge_features_dim = edge_features
        keys = jax.random.split(key, num_layers)
        edge_context_features = 2 * node_features + edge_features
        self.layers = tuple(
            OldDecoderLayer(node_features, edge_context_features, hidden_features, key=k)
            for k in keys
        )

    def __call__(self, *args, **kwargs):
        pass

# Mock ProteinFeatures to ensure exact structure match
class ProteinFeatures(eqx.Module):
    w_pos: eqx.nn.Linear
    w_e: eqx.nn.Linear
    norm_edges: LayerNorm
    w_e_proj: eqx.nn.Linear
    k_neighbors: int = eqx.field(static=True)
    rbf_dim: int = eqx.field(static=True)
    pos_embed_dim: int = eqx.field(static=True)
    
    def __init__(self, node_features, edge_features, k_neighbors, *, key):
        keys = jax.random.split(key, 4)
        self.k_neighbors = k_neighbors
        self.rbf_dim = 16
        self.pos_embed_dim = 16
        # 16+16+1+1+1+1+1+1+1+1+26 = 66
        self.w_pos = eqx.nn.Linear(self.pos_embed_dim + self.rbf_dim + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 26, 16, key=keys[0]) # 66 inputs
        self.w_e = eqx.nn.Linear(edge_features + 16 + 16 + 128 + 128, edge_features, use_bias=False, key=keys[1]) # 416 inputs
        self.norm_edges = LayerNorm(edge_features)
        self.w_e_proj = eqx.nn.Linear(edge_features, edge_features, key=keys[3])

class OldPrxteinMPNN(eqx.Module):
    features: ProteinFeatures
    encoder: OldEncoder
    decoder: OldDecoder
    w_s_embed: eqx.nn.Embedding
    w_out: eqx.nn.Linear
    node_features_dim: int = eqx.field(static=True)
    edge_features_dim: int = eqx.field(static=True)
    num_decoder_layers: int = eqx.field(static=True)

    def __init__(self, node_features, edge_features, hidden_features, num_encoder_layers, num_decoder_layers, k_neighbors, vocab_size=21, *, key):
        self.node_features_dim = node_features
        self.edge_features_dim = edge_features
        self.num_decoder_layers = num_decoder_layers
        keys = jax.random.split(key, 5)
        
        self.features = ProteinFeatures(node_features, edge_features, k_neighbors, key=keys[0])
        self.encoder = OldEncoder(node_features, edge_features, hidden_features, num_encoder_layers, key=keys[1])
        self.decoder = OldDecoder(node_features, edge_features, hidden_features, num_decoder_layers, key=keys[2])
        self.w_s_embed = eqx.nn.Embedding(num_embeddings=vocab_size, embedding_size=node_features, key=keys[3])
        self.w_out = eqx.nn.Linear(node_features, vocab_size, key=keys[4])


def migrate_single_weight(weight_type: str, version: str, output_dir: Path) -> tuple[str, bool]:
    """Migrate a single weight file."""
    filename = f"{weight_type}_{version}.eqx"
    output_path = output_dir / filename
    
    print(f"\n{'='*60}")
    print(f"Migrating: {filename}")
    print(f"{'='*60}")
    
    try:
        # Download
        print(f"Downloading...")
        weights_file = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=f"eqx/{filename}",
            repo_type="model",
        )
        print(f"✓ Downloaded to: {weights_file}")
        
        # Create old model skeleton
        print("Creating old model skeleton...")
        key = jax.random.PRNGKey(0)
        old_model = OldPrxteinMPNN(
            node_features=NODE_FEATURES,
            edge_features=EDGE_FEATURES,
            hidden_features=HIDDEN_FEATURES,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            num_decoder_layers=NUM_DECODER_LAYERS,
            k_neighbors=K_NEIGHBORS,
            vocab_size=VOCAB_SIZE,
            key=key
        )
        
        # Load old weights
        print("Loading old weights...")
        print(f"DEBUG: Structure before loading:\n{eqx.tree_pformat(old_model)}")
        old_model = eqx.tree_deserialise_leaves(weights_file, old_model)
        print(f"✓ Loaded")
        
        # Import NEW model (with dropout)
        print("Creating new model with dropout...")
        from prxteinmpnn.model.mpnn import PrxteinMPNN as NewModel
        new_model = NewModel(
            node_features=NODE_FEATURES,
            edge_features=EDGE_FEATURES,
            hidden_features=HIDDEN_FEATURES,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            num_decoder_layers=NUM_DECODER_LAYERS,
            k_neighbors=K_NEIGHBORS,
            vocab_size=VOCAB_SIZE,
            dropout_rate=DROPOUT_RATE,
            key=key
        )
        
        # Copy parameters by transferring leaves
        print("Copying parameters...")
        
        # Get parameters from old model
        old_params = eqx.filter(old_model, eqx.is_inexact_array)
        old_leaves = jax.tree_util.tree_leaves(old_params)
        
        # Get parameters structure from new model
        new_params = eqx.filter(new_model, eqx.is_inexact_array)
        new_leaves = jax.tree_util.tree_leaves(new_params)
        
        if len(old_leaves) != len(new_leaves):
            raise ValueError(f"Parameter mismatch! Old: {len(old_leaves)}, New: {len(new_leaves)}")
            
        # Reconstruct new parameters with old values
        new_params_filled = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(new_params), 
            old_leaves
        )
        
        # Combine with new static parts
        new_static = eqx.filter(new_model, eqx.is_inexact_array, inverse=True)
        migrated_model = eqx.combine(new_params_filled, new_static)
        print(f"✓ Copied")
        
        # Save
        print(f"Saving...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        eqx.tree_serialise_leaves(str(output_path), migrated_model)
        print(f"✓ Saved")
        
        return str(output_path), True
        
    except Exception as e:
        import traceback
        print(f"✗ Error: {e}")
        traceback.print_exc()
        return str(output_path), False


def main():
    print("="*60)
    print("PrxteinMPNN Weight Migration Tool")
    print("="*60)
    print(f"\nRepo: {HF_REPO_ID}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Total files: {len(WEIGHT_TYPES) * len(MODEL_VERSIONS)}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    results = []
    successful = 0
    failed = 0
    
    for weight_type in WEIGHT_TYPES:
        for version in MODEL_VERSIONS:
            output_path, success = migrate_single_weight(weight_type, version, OUTPUT_DIR)
            results.append((weight_type, version, output_path, success))
            if success:
                successful += 1
            else:
                failed += 1
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✓ Successful: {successful}/{len(results)}")
    print(f"✗ Failed: {failed}/{len(results)}")
    
    if successful > 0:
        print(f"\n✓ Migrated weights in: {OUTPUT_DIR}/")
        for wt, ver, _, success in results:
            if success:
                print(f"  ✓ {wt}_{ver}.eqx")
    
    if failed > 0:
        print("\n✗ Failed:")
        for wt, ver, _, success in results:
            if not success:
                print(f"  ✗ {wt}_{ver}.eqx")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("✓ ALL WEIGHTS MIGRATED SUCCESSFULLY!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Run: python scripts/push_all_to_hub.py")
    print("  2. Clean up: rm src/prxteinmpnn/model/*_old.py")
    print("="*60)


if __name__ == "__main__":
    main()
