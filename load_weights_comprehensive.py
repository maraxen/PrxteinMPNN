"""
Comprehensive weight loading from ColabDesign to PrxteinMPNN.
Maps ALL encoder and decoder layer weights correctly.
"""

import jax
import jax.numpy as jnp
import numpy as np
import joblib
import equinox as eqx
from prxteinmpnn.model import PrxteinMPNN

def load_prxteinmpnn_with_colabdesign_weights(weights_path, key=None):
    """Load PrxteinMPNN model with ColabDesign weights."""
    if key is None:
        key = jax.random.PRNGKey(0)

    # Load ColabDesign weights
    print(f"Loading ColabDesign weights...")
    checkpoint = joblib.load(weights_path)
    params = checkpoint['model_state_dict']
    k_neighbors = checkpoint['num_edges']

    # Create PrxteinMPNN skeleton
    print("Creating PrxteinMPNN skeleton...")
    model = PrxteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_features=512,
        num_encoder_layers=3,
        num_decoder_layers=3,
        vocab_size=21,
        k_neighbors=k_neighbors,
        key=key,
    )

    print("Injecting weights...")

    # Helper to update linear layer
    def update_linear_layer(layer, w, b):
        """Update a Linear layer with weights and bias."""
        layer = eqx.tree_at(lambda l: l.weight, layer, jnp.array(w))
        if b is not None:
            layer = eqx.tree_at(lambda l: l.bias, layer, jnp.array(b))
        return layer

    # Helper to update layer norm
    def update_layer_norm(layer, scale, offset):
        """Update a LayerNorm layer."""
        layer = eqx.tree_at(lambda l: l.weight, layer, jnp.array(scale))
        layer = eqx.tree_at(lambda l: l.bias, layer, jnp.array(offset))
        return layer

    # 1. W_out
    print("  1. W_out...")
    w_out = params['protein_mpnn/~/W_out']['w'].T  # (21, 128)
    b_out = params['protein_mpnn/~/W_out']['b']
    model = eqx.tree_at(
        lambda m: m.w_out,
        model,
        update_linear_layer(model.w_out, w_out, b_out)
    )

    # 2. W_s_embed
    print("  2. W_s_embed...")
    w_s = params['protein_mpnn/~/embed_token']['W_s']  # (21, 128)
    model = eqx.tree_at(
        lambda m: m.w_s_embed.weight,
        model,
        jnp.array(w_s)
    )

    # 3. Feature layers
    print("  3. Feature layers...")

    # w_pos
    w = params['protein_mpnn/~/protein_features/~/positional_encodings/~/embedding_linear']['w'].T
    b = params['protein_mpnn/~/protein_features/~/positional_encodings/~/embedding_linear']['b']
    model = eqx.tree_at(
        lambda m: m.features.w_pos,
        model,
        update_linear_layer(model.features.w_pos, w, b)
    )

    # w_e (edge embedding)
    w = params['protein_mpnn/~/protein_features/~/edge_embedding']['w'].T
    model = eqx.tree_at(
        lambda m: m.features.w_e.weight,
        model,
        jnp.array(w)
    )

    # norm_edges
    scale = params['protein_mpnn/~/protein_features/~/norm_edges']['scale']
    offset = params['protein_mpnn/~/protein_features/~/norm_edges']['offset']
    model = eqx.tree_at(
        lambda m: m.features.norm_edges,
        model,
        update_layer_norm(model.features.norm_edges, scale, offset)
    )

    # w_e_proj (edge projection) - this is W_e in ColabDesign!
    w = params['protein_mpnn/~/W_e']['w'].T  # (128, 128)
    b = params['protein_mpnn/~/W_e']['b']
    model = eqx.tree_at(
        lambda m: m.features.w_e_proj,
        model,
        update_linear_layer(model.features.w_e_proj, w, b)
    )

    # 4. Encoder layers
    print("  4. Encoder layers...")
    for i in range(3):
        prefix = "protein_mpnn/~/enc_layer" if i == 0 else f"protein_mpnn/~/enc_layer_{i}"
        print(f"    Layer {i}...")

        enc_layer = model.encoder.layers[i]

        # edge_message_mlp: MLP(384->128, width=128, depth=2) -> 3 layers
        # Maps to enc{i}_W1, enc{i}_W2, enc{i}_W3
        w1 = params[f'{prefix}/~/enc{i}_W1']['w'].T  # (128, 384)
        b1 = params[f'{prefix}/~/enc{i}_W1']['b']
        w2 = params[f'{prefix}/~/enc{i}_W2']['w'].T  # (128, 128)
        b2 = params[f'{prefix}/~/enc{i}_W2']['b']
        w3 = params[f'{prefix}/~/enc{i}_W3']['w'].T  # (128, 128)
        b3 = params[f'{prefix}/~/enc{i}_W3']['b']

        enc_layer = eqx.tree_at(
            lambda l: l.edge_message_mlp.layers[0],
            enc_layer,
            update_linear_layer(enc_layer.edge_message_mlp.layers[0], w1, b1)
        )
        enc_layer = eqx.tree_at(
            lambda l: l.edge_message_mlp.layers[1],
            enc_layer,
            update_linear_layer(enc_layer.edge_message_mlp.layers[1], w2, b2)
        )
        enc_layer = eqx.tree_at(
            lambda l: l.edge_message_mlp.layers[2],
            enc_layer,
            update_linear_layer(enc_layer.edge_message_mlp.layers[2], w3, b3)
        )

        # norm1
        scale = params[f'{prefix}/~/enc{i}_norm1']['scale']
        offset = params[f'{prefix}/~/enc{i}_norm1']['offset']
        enc_layer = eqx.tree_at(
            lambda l: l.norm1,
            enc_layer,
            update_layer_norm(enc_layer.norm1, scale, offset)
        )

        # dense: MLP(128->128, width=512, depth=1) -> 2 layers
        # Maps to enc{i}_dense_W_in, enc{i}_dense_W_out
        w_in = params[f'{prefix}/~/position_wise_feed_forward/~/enc{i}_dense_W_in']['w'].T  # (512, 128)
        b_in = params[f'{prefix}/~/position_wise_feed_forward/~/enc{i}_dense_W_in']['b']
        w_out = params[f'{prefix}/~/position_wise_feed_forward/~/enc{i}_dense_W_out']['w'].T  # (128, 512)
        b_out = params[f'{prefix}/~/position_wise_feed_forward/~/enc{i}_dense_W_out']['b']

        enc_layer = eqx.tree_at(
            lambda l: l.dense.layers[0],
            enc_layer,
            update_linear_layer(enc_layer.dense.layers[0], w_in, b_in)
        )
        enc_layer = eqx.tree_at(
            lambda l: l.dense.layers[1],
            enc_layer,
            update_linear_layer(enc_layer.dense.layers[1], w_out, b_out)
        )

        # norm2
        scale = params[f'{prefix}/~/enc{i}_norm2']['scale']
        offset = params[f'{prefix}/~/enc{i}_norm2']['offset']
        enc_layer = eqx.tree_at(
            lambda l: l.norm2,
            enc_layer,
            update_layer_norm(enc_layer.norm2, scale, offset)
        )

        # edge_update_mlp: MLP(384->128, width=128, depth=2) -> 3 layers
        # Maps to enc{i}_W11, enc{i}_W12, enc{i}_W13
        w11 = params[f'{prefix}/~/enc{i}_W11']['w'].T  # (128, 384)
        b11 = params[f'{prefix}/~/enc{i}_W11']['b']
        w12 = params[f'{prefix}/~/enc{i}_W12']['w'].T  # (128, 128)
        b12 = params[f'{prefix}/~/enc{i}_W12']['b']
        w13 = params[f'{prefix}/~/enc{i}_W13']['w'].T  # (128, 128)
        b13 = params[f'{prefix}/~/enc{i}_W13']['b']

        enc_layer = eqx.tree_at(
            lambda l: l.edge_update_mlp.layers[0],
            enc_layer,
            update_linear_layer(enc_layer.edge_update_mlp.layers[0], w11, b11)
        )
        enc_layer = eqx.tree_at(
            lambda l: l.edge_update_mlp.layers[1],
            enc_layer,
            update_linear_layer(enc_layer.edge_update_mlp.layers[1], w12, b12)
        )
        enc_layer = eqx.tree_at(
            lambda l: l.edge_update_mlp.layers[2],
            enc_layer,
            update_linear_layer(enc_layer.edge_update_mlp.layers[2], w13, b13)
        )

        # norm3
        scale = params[f'{prefix}/~/enc{i}_norm3']['scale']
        offset = params[f'{prefix}/~/enc{i}_norm3']['offset']
        enc_layer = eqx.tree_at(
            lambda l: l.norm3,
            enc_layer,
            update_layer_norm(enc_layer.norm3, scale, offset)
        )

        # Update encoder layer in model
        layers_list = list(model.encoder.layers)
        layers_list[i] = enc_layer
        model = eqx.tree_at(
            lambda m: m.encoder.layers,
            model,
            tuple(layers_list)
        )

    # 5. Decoder layers
    print("  5. Decoder layers...")
    for i in range(3):
        prefix = "protein_mpnn/~/dec_layer" if i == 0 else f"protein_mpnn/~/dec_layer_{i}"
        print(f"    Layer {i}...")

        dec_layer = model.decoder.layers[i]

        # message_mlp: MLP(512->128, width=128, depth=2) -> 3 layers
        # Maps to dec{i}_W1, dec{i}_W2, dec{i}_W3
        w1 = params[f'{prefix}/~/dec{i}_W1']['w'].T  # (128, 512)
        b1 = params[f'{prefix}/~/dec{i}_W1']['b']
        w2 = params[f'{prefix}/~/dec{i}_W2']['w'].T  # (128, 128)
        b2 = params[f'{prefix}/~/dec{i}_W2']['b']
        w3 = params[f'{prefix}/~/dec{i}_W3']['w'].T  # (128, 128)
        b3 = params[f'{prefix}/~/dec{i}_W3']['b']

        dec_layer = eqx.tree_at(
            lambda l: l.message_mlp.layers[0],
            dec_layer,
            update_linear_layer(dec_layer.message_mlp.layers[0], w1, b1)
        )
        dec_layer = eqx.tree_at(
            lambda l: l.message_mlp.layers[1],
            dec_layer,
            update_linear_layer(dec_layer.message_mlp.layers[1], w2, b2)
        )
        dec_layer = eqx.tree_at(
            lambda l: l.message_mlp.layers[2],
            dec_layer,
            update_linear_layer(dec_layer.message_mlp.layers[2], w3, b3)
        )

        # norm1
        scale = params[f'{prefix}/~/dec{i}_norm1']['scale']
        offset = params[f'{prefix}/~/dec{i}_norm1']['offset']
        dec_layer = eqx.tree_at(
            lambda l: l.norm1,
            dec_layer,
            update_layer_norm(dec_layer.norm1, scale, offset)
        )

        # dense: MLP(128->128, width=512, depth=1) -> 2 layers
        w_in = params[f'{prefix}/~/position_wise_feed_forward/~/dec{i}_dense_W_in']['w'].T
        b_in = params[f'{prefix}/~/position_wise_feed_forward/~/dec{i}_dense_W_in']['b']
        w_out = params[f'{prefix}/~/position_wise_feed_forward/~/dec{i}_dense_W_out']['w'].T
        b_out = params[f'{prefix}/~/position_wise_feed_forward/~/dec{i}_dense_W_out']['b']

        dec_layer = eqx.tree_at(
            lambda l: l.dense.layers[0],
            dec_layer,
            update_linear_layer(dec_layer.dense.layers[0], w_in, b_in)
        )
        dec_layer = eqx.tree_at(
            lambda l: l.dense.layers[1],
            dec_layer,
            update_linear_layer(dec_layer.dense.layers[1], w_out, b_out)
        )

        # norm2
        scale = params[f'{prefix}/~/dec{i}_norm2']['scale']
        offset = params[f'{prefix}/~/dec{i}_norm2']['offset']
        dec_layer = eqx.tree_at(
            lambda l: l.norm2,
            dec_layer,
            update_layer_norm(dec_layer.norm2, scale, offset)
        )

        # Update decoder layer in model
        layers_list = list(model.decoder.layers)
        layers_list[i] = dec_layer
        model = eqx.tree_at(
            lambda m: m.decoder.layers,
            model,
            tuple(layers_list)
        )

    print("✅ All weights loaded!")
    return model

if __name__ == "__main__":
    model = load_prxteinmpnn_with_colabdesign_weights(
        "/tmp/ColabDesign/colabdesign/mpnn/weights/v_48_020.pkl"
    )
    print(f"\nModel ready with {sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))} parameters")
