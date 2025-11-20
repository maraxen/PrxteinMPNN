import equinox as eqx
import jax
import jax.numpy as jnp
import argparse
import sys
from prxteinmpnn.model.mpnn import PrxteinMPNN

def main():
    parser = argparse.ArgumentParser(description="Migrate old PrxteinMPNN weights to new dropout-enabled architecture.")
    parser.add_argument("--input", type=str, required=True, help="Path to input .eqx weights file")
    parser.add_argument("--output", type=str, required=True, help="Path to output .eqx weights file")
    parser.add_argument("--node-features", type=int, default=128, help="Node feature dimension")
    parser.add_argument("--edge-features", type=int, default=128, help="Edge feature dimension")
    parser.add_argument("--hidden-features", type=int, default=128, help="Hidden feature dimension")
    parser.add_argument("--num-encoder-layers", type=int, default=3, help="Number of encoder layers")
    parser.add_argument("--num-decoder-layers", type=int, default=3, help="Number of decoder layers")
    
    args = parser.parse_args()

    print(f"Initializing new model skeleton with config:")
    print(f"  Node features: {args.node_features}")
    print(f"  Edge features: {args.edge_features}")
    print(f"  Hidden features: {args.hidden_features}")
    print(f"  Encoder layers: {args.num_encoder_layers}")
    print(f"  Decoder layers: {args.num_decoder_layers}")

    key = jax.random.PRNGKey(0)
    # Initialize with default dropout_rate (it doesn't affect weights)
    model = PrxteinMPNN(
        node_features=args.node_features,
        edge_features=args.edge_features,
        hidden_features=args.hidden_features,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        key=key
    )

    print(f"Loading weights from {args.input}...")
    try:
        # Attempt to load leaves into the new model structure
        # Since Dropout layers have no leaves (parameters), this should work if the rest of the structure is identical.
        model = eqx.tree_deserialise_leaves(args.input, model)
        print("Successfully loaded weights into new model structure.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("The model configuration (layers, dimensions) might not match the checkpoint, or the structure is incompatible.")
        sys.exit(1)

    print(f"Saving migrated weights to {args.output}...")
    try:
        eqx.tree_serialise_leaves(args.output, model)
        print("Done.")
    except Exception as e:
        print(f"Error saving weights: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
