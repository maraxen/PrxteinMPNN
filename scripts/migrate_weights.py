import equinox as eqx
import jax
import jax.numpy as jnp
import argparse
import sys
import os
from pathlib import Path
from huggingface_hub import hf_hub_download
from prxteinmpnn.model.mpnn import PrxteinMPNN

DEFAULT_OUTPUT_PATH = "prxteinmpnn_dropout_weights.eqx"
DEFAULT_REPO_ID = "maraxen/PrxteinMPNN"
DEFAULT_FILENAME = "prxteinmpnn_v1.eqx"

def main():
    parser = argparse.ArgumentParser(
        description="Migrate old PrxteinMPNN weights to new dropout-enabled architecture.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download from HuggingFace and migrate
  python migrate_weights.py
  
  # Use local weights file
  python migrate_weights.py --input local_weights.eqx
  
  # Specify custom output path
  python migrate_weights.py --output custom_output.eqx
        """
    )
    parser.add_argument(
        "--input", 
        type=str, 
        default=None,
        help=f"Path to input .eqx weights file. If not provided, downloads from HuggingFace ({DEFAULT_REPO_ID})"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path to output .eqx weights file (default: {DEFAULT_OUTPUT_PATH})"
    )
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID, help="HuggingFace repo ID")
    parser.add_argument("--filename", type=str, default=DEFAULT_FILENAME, help="Filename in HuggingFace repo")
    parser.add_argument("--node-features", type=int, default=128, help="Node feature dimension")
    parser.add_argument("--edge-features", type=int, default=128, help="Edge feature dimension")
    parser.add_argument("--hidden-features", type=int, default=128, help="Hidden feature dimension")
    parser.add_argument("--num-encoder-layers", type=int, default=3, help="Number of encoder layers")
    parser.add_argument("--num-decoder-layers", type=int, default=3, help="Number of decoder layers")
    parser.add_argument("--k-neighbors", type=int, default=30, help="Number of neighbors")
    parser.add_argument("--dropout-rate", type=float, default=0.1, help="Dropout rate (doesn't affect weights)")
    
    args = parser.parse_args()

    # Download from HuggingFace if no input provided
    if args.input is None:
        print(f"Downloading weights from HuggingFace: {args.repo_id}/{args.filename}")
        try:
            args.input = hf_hub_download(repo_id=args.repo_id, filename=args.filename)
            print(f"Downloaded to: {args.input}")
        except Exception as e:
            print(f"Error downloading from HuggingFace: {e}")
            sys.exit(1)

    print(f"\nInitializing new model skeleton with config:")
    print(f"  Node features: {args.node_features}")
    print(f"  Edge features: {args.edge_features}")
    print(f"  Hidden features: {args.hidden_features}")
    print(f"  Encoder layers: {args.num_encoder_layers}")
    print(f"  Decoder layers: {args.num_decoder_layers}")
    print(f"  K neighbors: {args.k_neighbors}")
    print(f"  Dropout rate: {args.dropout_rate}")

    key = jax.random.PRNGKey(0)
    model = PrxteinMPNN(
        node_features=args.node_features,
        edge_features=args.edge_features,
        hidden_features=args.hidden_features,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        k_neighbors=args.k_neighbors,
        dropout_rate=args.dropout_rate,
        key=key
    )

    print(f"\nLoading weights from {args.input}...")
    try:
        model = eqx.tree_deserialise_leaves(args.input, model)
        print("✓ Successfully loaded weights into new model structure.")
    except Exception as e:
        print(f"✗ Error loading weights: {e}")
        print("The model configuration (layers, dimensions) might not match the checkpoint.")
        sys.exit(1)

    print(f"\nSaving migrated weights to {args.output}...")
    try:
        # Create output directory if needed
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        eqx.tree_serialise_leaves(args.output, model)
        print(f"✓ Done! Migrated weights saved to: {args.output}")
    except Exception as e:
        print(f"✗ Error saving weights: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
