#!/usr/bin/env python3
"""Migrate all PrxteinMPNN weights to dropout-enabled architecture.

This script downloads all weight variants from HuggingFace, migrates them to
the new dropout-enabled architecture, and prepares them for upload.
"""

import equinox as eqx
import jax
from pathlib import Path
import sys
from huggingface_hub import hf_hub_download
from prxteinmpnn.model.mpnn import PrxteinMPNN

# Configuration from weights.py
HF_REPO_ID = "maraxen/prxteinmpnn"
NODE_FEATURES = 128
EDGE_FEATURES = 128
HIDDEN_FEATURES = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
K_NEIGHBORS = 48
VOCAB_SIZE = 21
DROPOUT_RATE = 0.1

# All weight variants to migrate
WEIGHT_TYPES = ["original", "soluble"]
MODEL_VERSIONS = ["v_48_002", "v_48_010", "v_48_020", "v_48_030"]

OUTPUT_DIR = Path("migrated_weights")


def migrate_single_weight(
    weight_type: str,
    version: str,
    output_dir: Path,
) -> tuple[str, bool]:
    """Migrate a single weight file.
    
    Returns:
        Tuple of (output_path, success)
    """
    filename = f"{weight_type}_{version}.eqx"
    output_path = output_dir / filename
    
    print(f"\n{'='*60}")
    print(f"Migrating: {filename}")
    print(f"{'='*60}")
    
    try:
        # Download from HuggingFace
        print(f"Downloading from {HF_REPO_ID}/eqx/{filename}...")
        weights_file = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=f"eqx/{filename}",
            repo_type="model",
        )
        print(f"✓ Downloaded to: {weights_file}")
        
        # Create model skeleton
        print("Creating model skeleton...")
        key = jax.random.PRNGKey(0)
        model = PrxteinMPNN(
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
        
        # Load weights into new skeleton
        print("Loading weights into dropout-enabled skeleton...")
        model = eqx.tree_deserialise_leaves(weights_file, model)
        print("✓ Weights loaded successfully")
        
        # Save migrated weights
        print(f"Saving to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        eqx.tree_serialise_leaves(str(output_path), model)
        print(f"✓ Saved migrated weights")
        
        return str(output_path), True
        
    except Exception as e:
        print(f"✗ Error migrating {filename}: {e}")
        return str(output_path), False


def main():
    print("="*60)
    print("PrxteinMPNN Weight Migration Tool")
    print("="*60)
    print(f"\nMigrating all weights from {HF_REPO_ID}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"\nWeight types: {', '.join(WEIGHT_TYPES)}")
    print(f"Model versions: {', '.join(MODEL_VERSIONS)}")
    print(f"Total files to migrate: {len(WEIGHT_TYPES) * len(MODEL_VERSIONS)}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Track results
    results = []
    successful = 0
    failed = 0
    
    # Migrate all combinations
    for weight_type in WEIGHT_TYPES:
        for version in MODEL_VERSIONS:
            output_path, success = migrate_single_weight(
                weight_type,
                version,
                OUTPUT_DIR,
            )
            results.append((weight_type, version, output_path, success))
            if success:
                successful += 1
            else:
                failed += 1
    
    # Print summary
    print("\n" + "="*60)
    print("MIGRATION SUMMARY")
    print("="*60)
    print(f"Total: {len(results)}")
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")
    
    if successful > 0:
        print(f"\n✓ Migrated weights saved to: {OUTPUT_DIR}/")
        print("\nSuccessfully migrated files:")
        for weight_type, version, path, success in results:
            if success:
                print(f"  - {weight_type}_{version}.eqx")
    
    if failed > 0:
        print("\n✗ Failed migrations:")
        for weight_type, version, path, success in results:
            if not success:
                print(f"  - {weight_type}_{version}.eqx")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("Next steps:")
    print("  1. Review migrated weights in migrated_weights/")
    print("  2. Run: python scripts/push_all_to_hub.py")
    print("="*60)


if __name__ == "__main__":
    main()
