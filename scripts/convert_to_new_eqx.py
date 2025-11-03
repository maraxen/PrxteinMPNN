"""Convert ALL old ProteinMPNNWrapped weights to the new PrxteinMPNN format.

This script loops through all models on the Hugging Face Hub,
downloads them, maps their weights to the new, fully modular
`prxteinmpnn.eqx.PrxteinMPNN` structure, and saves them to a
local output directory.
"""

import argparse
from pathlib import Path
from typing import cast

import equinox as eqx
import jax

# from prxteinmpnn.eqx_old import PrxteinMPNNOld # No longer used for loading
# from prxteinmpnn.conversion import create_prxteinmpnn as create_prxteinmpnn_from_pytree # No longer used
from prxteinmpnn.eqx_new import PrxteinMPNN
from prxteinmpnn.io.weights import (
  ALL_MODEL_VERSIONS,
  ALL_MODEL_WEIGHTS,
  load_weights,
)

# --- Define Model Hyperparameters ---
NODE_FEATURES = 128
EDGE_FEATURES = 128
HIDDEN_FEATURES = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
K_NEIGHBORS = 48
VOCAB_SIZE = 21
NUM_AMINO_ACIDS = 21
# SCALE = 30.0 # This might be needed if create_prxteinmpnn is used

# --- Helper functions to build new eqx modules from old weight dicts ---


def create_linear_from_dict(w_b_dict: dict, *, key: jax.Array) -> eqx.nn.Linear:
  """Creates an eqx.nn.Linear layer from a {'w': ..., 'b': ...} dict.

  Handles cases where 'b' (bias) may not be present.
  """
  w = w_b_dict["w"]
  # Check if bias exists
  has_bias = "b" in w_b_dict
  b = w_b_dict["b"] if has_bias else None

  in_features, out_features = w.shape

  # Create a dummy Linear layer to get the structure
  linear = eqx.nn.Linear(in_features, out_features, use_bias=has_bias, key=key)

  # Replace weights
  # Note: Old weights are (in, out), Equinox Linear is (out, in)
  linear = eqx.tree_at(lambda m: m.weight, linear, w.T)

  # Replace bias only if it exists
  if has_bias:
    linear = eqx.tree_at(lambda m: m.bias, linear, b)

  return linear


def create_layernorm_from_dict(s_o_dict: dict) -> eqx.nn.LayerNorm:
  """Creates an eqx.nn.LayerNorm from a {'scale': ..., 'offset': ...} dict."""
  scale = s_o_dict["scale"]
  offset = s_o_dict["offset"]
  shape = scale.shape
  layer_norm = eqx.nn.LayerNorm(shape, use_weight=True, use_bias=True)
  layer_norm = eqx.tree_at(lambda m: m.weight, layer_norm, scale)
  return eqx.tree_at(lambda m: m.bias, layer_norm, offset)


def convert_from_old(
  old_weights_dict: dict,
  key: jax.Array,
) -> PrxteinMPNN:
  """Manually maps weights from the raw flat PyTree dictionary
  to the new PrxteinMPNN structure.
  """
  print("  Building new model skeleton...")

  # The actual weights are in the 'model_state_dict' sub-dictionary
  param_dict = old_weights_dict["model_state_dict"]

  print("\n--- Printing loaded parameter shapes ---")
  for k, v in param_dict.items():
    if isinstance(v, dict):
      for kk, vv in v.items():
        print(f"{k}/{kk}: {getattr(vv, 'shape', type(vv))}")
    else:
      print(f"{k}: {getattr(v, 'shape', type(v))}")
  print("--- End parameter shapes ---\n")

  keys = jax.random.split(key, 10)  # Split key for all components

  # 1. Create the new skeleton
  new_model = PrxteinMPNN(
    node_features=NODE_FEATURES,
    edge_features=EDGE_FEATURES,
    hidden_features=HIDDEN_FEATURES,
    num_encoder_layers=NUM_ENCODER_LAYERS,
    num_decoder_layers=NUM_DECODER_LAYERS,
    k_neighbors=K_NEIGHBORS,
    num_amino_acids=NUM_AMINO_ACIDS,
    vocab_size=VOCAB_SIZE,
    key=keys[0],
  )

  print("  Mapping Encoder layers...")
  # 2. Create new Encoder layers from old ones
  new_encoder_layers = []
  enc_layer_keys = jax.random.split(keys[1], NUM_ENCODER_LAYERS)
  for i in range(NUM_ENCODER_LAYERS):
    # Get the skeleton of the new layer
    new_layer_skel = new_model.encoder.layers[i]

    # Define the flat keys based on the layer index
    key_prefix = f"enc{i}"
    layer_prefix = f"enc_layer_{i}" if i > 0 else "enc_layer"
    base_path = f"protein_mpnn/~/{layer_prefix}/~/"
    ff_path = f"{base_path}position_wise_feed_forward/~/"

    layer_keys = jax.random.split(enc_layer_keys[i], 11)

    # Map MLPs (w1, w2, w3 -> edge_message_mlp.layers[0,1,2])
    mlp = new_layer_skel.edge_message_mlp
    mlp = eqx.tree_at(
      lambda l: l.layers[0],
      mlp,
      create_linear_from_dict(param_dict[f"{base_path}{key_prefix}_W1"], key=layer_keys[0]),
    )
    mlp = eqx.tree_at(
      lambda l: l.layers[1],
      mlp,
      create_linear_from_dict(param_dict[f"{base_path}{key_prefix}_W2"], key=layer_keys[1]),
    )
    mlp = eqx.tree_at(
      lambda l: l.layers[2],
      mlp,
      create_linear_from_dict(param_dict[f"{base_path}{key_prefix}_W3"], key=layer_keys[2]),
    )

    # Map Dense (now an MLP with dense_W_in and dense_W_out)
    dense = new_layer_skel.dense
    dense = eqx.tree_at(
      lambda d: d.layers[0],
      dense,
      create_linear_from_dict(param_dict[f"{ff_path}{key_prefix}_dense_W_in"], key=layer_keys[3]),
    )
    dense = eqx.tree_at(
      lambda d: d.layers[1],
      dense,
      create_linear_from_dict(param_dict[f"{ff_path}{key_prefix}_dense_W_out"], key=layer_keys[4]),
    )

    # Map Update MLPs (w11, w12, w13 -> edge_update_mlp.layers[0,1,2])
    up_mlp = new_layer_skel.edge_update_mlp
    up_mlp = eqx.tree_at(
      lambda l: l.layers[0],
      up_mlp,
      create_linear_from_dict(param_dict[f"{base_path}{key_prefix}_W11"], key=layer_keys[6]),
    )
    up_mlp = eqx.tree_at(
      lambda l: l.layers[1],
      up_mlp,
      create_linear_from_dict(param_dict[f"{base_path}{key_prefix}_W12"], key=layer_keys[7]),
    )
    up_mlp = eqx.tree_at(
      lambda l: l.layers[2],
      up_mlp,
      create_linear_from_dict(param_dict[f"{base_path}{key_prefix}_W13"], key=layer_keys[8]),
    )

    # Re-assemble the new layer by replacing components
    new_layer = new_layer_skel
    new_layer = eqx.tree_at(lambda nl: nl.edge_message_mlp, new_layer, mlp)
    new_layer = eqx.tree_at(lambda nl: nl.dense, new_layer, dense)
    new_layer = eqx.tree_at(lambda nl: nl.edge_update_mlp, new_layer, up_mlp)
    # Map LayerNorms
    new_layer = eqx.tree_at(
      lambda nl: nl.norm1,
      new_layer,
      create_layernorm_from_dict(param_dict[f"{base_path}{key_prefix}_norm1"]),
    )
    new_layer = eqx.tree_at(
      lambda nl: nl.norm2,
      new_layer,
      create_layernorm_from_dict(param_dict[f"{base_path}{key_prefix}_norm2"]),
    )
    new_layer = eqx.tree_at(
      lambda nl: nl.norm3,
      new_layer,
      create_layernorm_from_dict(param_dict[f"{base_path}{key_prefix}_norm3"]),
    )

    new_encoder_layers.append(new_layer)

  # Replace the entire tuple of layers in the encoder
  where_enc_layers = lambda m: m.encoder.layers
  new_model = eqx.tree_at(where_enc_layers, new_model, tuple(new_encoder_layers))

  print("  Mapping Decoder layers...")
  # 3. Create new Decoder layers from old ones
  new_decoder_layers = []
  dec_layer_keys = jax.random.split(keys[2], NUM_DECODER_LAYERS)
  for i in range(NUM_DECODER_LAYERS):
    new_layer_skel = new_model.decoder.layers[i]

    # Define the flat keys based on the layer index
    key_prefix = f"dec{i}"
    layer_prefix = f"dec_layer_{i}" if i > 0 else "dec_layer"
    base_path = f"protein_mpnn/~/{layer_prefix}/~/"
    ff_path = f"{base_path}position_wise_feed_forward/~/"

    layer_keys = jax.random.split(dec_layer_keys[i], 5)

    # Map MLPs (w1, w2, w3 -> message_mlp.layers[0,1,2])
    mlp = new_layer_skel.message_mlp
    mlp = eqx.tree_at(
      lambda l: l.layers[0],
      mlp,
      create_linear_from_dict(param_dict[f"{base_path}{key_prefix}_W1"], key=layer_keys[0]),
    )
    mlp = eqx.tree_at(
      lambda l: l.layers[1],
      mlp,
      create_linear_from_dict(param_dict[f"{base_path}{key_prefix}_W2"], key=layer_keys[1]),
    )
    mlp = eqx.tree_at(
      lambda l: l.layers[2],
      mlp,
      create_linear_from_dict(param_dict[f"{base_path}{key_prefix}_W3"], key=layer_keys[2]),
    )

    # Map Dense (dense.linear_in/out -> dense.layers[0,1])
    dense = new_layer_skel.dense
    dense = eqx.tree_at(
      lambda d: d.layers[0],
      dense,
      create_linear_from_dict(param_dict[f"{ff_path}{key_prefix}_dense_W_in"], key=layer_keys[3]),
    )
    dense = eqx.tree_at(
      lambda d: d.layers[1],
      dense,
      create_linear_from_dict(param_dict[f"{ff_path}{key_prefix}_dense_W_out"], key=layer_keys[4]),
    )

    # Re-assemble the new layer
    new_layer = new_layer_skel
    new_layer = eqx.tree_at(lambda nl: nl.message_mlp, new_layer, mlp)
    new_layer = eqx.tree_at(lambda nl: nl.dense, new_layer, dense)
    new_layer = eqx.tree_at(
      lambda nl: nl.norm1,
      new_layer,
      create_layernorm_from_dict(param_dict[f"{base_path}{key_prefix}_norm1"]),
    )
    new_layer = eqx.tree_at(
      lambda nl: nl.norm2,
      new_layer,
      create_layernorm_from_dict(param_dict[f"{base_path}{key_prefix}_norm2"]),
    )

    new_decoder_layers.append(new_layer)

  # Replace the entire tuple of layers in the decoder
  where_dec_layers = lambda m: m.decoder.layers
  new_model = eqx.tree_at(where_dec_layers, new_model, tuple(new_decoder_layers))

  print("  Mapping feature weights...")
  # 4. Map Feature Extraction weights
  w_pos_dict = param_dict[
    "protein_mpnn/~/protein_features/~/positional_encodings/~/embedding_linear"
  ]
  w_e_dict = param_dict["protein_mpnn/~/protein_features/~/edge_embedding"]
  norm_edges_dict = param_dict["protein_mpnn/~/protein_features/~/norm_edges"]

  # The old model state dict also has 'protein_mpnn/~/W_e' which seems to be
  # a *projection* of the edge_embedding. This maps to `new.features.w_e_proj`
  w_e_proj_dict = param_dict["protein_mpnn/~/W_e"]

  where_w_pos = lambda m: m.features.w_pos
  new_model = eqx.tree_at(where_w_pos, new_model, create_linear_from_dict(w_pos_dict, key=keys[3]))

  where_w_e = lambda m: m.features.w_e
  new_model = eqx.tree_at(where_w_e, new_model, create_linear_from_dict(w_e_dict, key=keys[4]))

  where_norm_edges = lambda m: m.features.norm_edges
  new_model = eqx.tree_at(where_norm_edges, new_model, create_layernorm_from_dict(norm_edges_dict))

  where_w_e_proj = lambda m: m.features.w_e_proj
  new_model = eqx.tree_at(
    where_w_e_proj,
    new_model,
    create_linear_from_dict(w_e_proj_dict, key=keys[5]),
  )

  print("  Mapping output layer weights...")
  # 5. Map Output Layer (w_out and w_s_embed)
  w_out_dict = param_dict["protein_mpnn/~/W_out"]
  w_s_dict = param_dict["protein_mpnn/~/embed_token"]  # This is the sequence embedding

  # Map w_out
  where_w_out = lambda m: m.w_out
  new_model = eqx.tree_at(where_w_out, new_model, create_linear_from_dict(w_out_dict, key=keys[6]))

  # Map w_s_embed
  w_s_embed_skel = new_model.w_s_embed
  # Fix: The old weight is not a dict, but the raw tensor itself.
  w_s_embed_new = eqx.tree_at(lambda e: e.weight, w_s_embed_skel, w_s_dict)
  # Fix: The attribute name is w_s_embed, not s_embed
  where_w_s = lambda m: m.w_s_embed
  new_model = eqx.tree_at(where_w_s, new_model, w_s_embed_new)

  print("  Weight mapping complete.")

  # Ensure the return type is correct for downstream type checking
  return cast("PrxteinMPNN", new_model)


def main(output_dir: str):
  """Main script function: loops through all models and converts them."""
  out_path = Path(output_dir)
  if not out_path.exists():
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Created output directory: {out_path}")

  # 1. Create a dummy key
  key = jax.random.PRNGKey(0)
  keys = jax.random.split(key, 2)

  # --- Main Conversion Loop ---
  for model_weights in ALL_MODEL_WEIGHTS:
    for model_version in ALL_MODEL_VERSIONS:
      print(f"\n--- Processing {model_weights} / {model_version} ---")

      # 1. Define output path
      new_model_path = out_path / f"{model_weights}_{model_version}.eqx"

      # Overwrite existing files if present
      if new_model_path.exists():
        print(f"Overwriting existing file: {new_model_path}")

      # 2. Load the old weights (PyTree dictionary) from HF Hub
      try:
        # We load WITHOUT a skeleton, expecting a raw PyTree dictionary of weights.
        old_weights_dict = load_weights(
          model_version=model_version,
          model_weights=model_weights,
          skeleton=None,  # Load as raw PyTree dict
        )
      except Exception as e:
        print(f"Failed to load raw PyTree weights for {model_weights}/{model_version}: {e}")
        continue

      # Ensure the loaded module is of the expected type
      if not isinstance(old_weights_dict, dict):
        print("Loaded weights are not a dictionary. Skipping.")
        continue

      # 3. Perform the conversion from PyTree to the new PrxteinMPNN model
      new_model = convert_from_old(old_weights_dict, keys[0])

      # 4. Save the new model
      print(f"  Saving new model to {new_model_path}...")
      eqx.tree_serialise_leaves(new_model_path, new_model)

  print("\nDone.")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Convert all old (PyTree) ProteinMPNN weights from HF Hub to new PrxteinMPNN format.",
  )
  parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Directory to save the new model .eqx files.",
  )
  args = parser.parse_args()
  main(args.output_dir)
