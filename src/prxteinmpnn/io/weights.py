"""Unified loader for PrxteinMPNN weights from Hugging Face Hub or local path."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import equinox as eqx
import jax
from huggingface_hub import hf_hub_download

if TYPE_CHECKING:
  from jaxtyping import PyTree

  from prxteinmpnn.model import PrxteinMPNN

MODEL_WEIGHTS = Literal["original", "soluble"]
MODEL_VERSION = Literal["v_48_002", "v_48_010", "v_48_020", "v_48_030"]
HF_REPO_ID = "maraxen/prxteinmpnn"
ALL_MODEL_WEIGHTS: list[MODEL_WEIGHTS] = ["original", "soluble"]
ALL_MODEL_VERSIONS: list[MODEL_VERSION] = [
  "v_48_002",
  "v_48_010",
  "v_48_020",
  "v_48_030",
]

# Model hyperparameters (same for all models)
NODE_FEATURES = 128
EDGE_FEATURES = 128
HIDDEN_FEATURES = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
K_NEIGHBORS = 48
VOCAB_SIZE = 21


def load_weights(
  model_version: MODEL_VERSION = "v_48_020",
  model_weights: MODEL_WEIGHTS = "original",
  local_path: str | None = None,
  skeleton: eqx.Module | None = None,
) -> PyTree | eqx.Module:
  """Load PrxteinMPNN weights from Hugging Face Hub or a local path.

  Args:
      model_version: The model version (e.g., "v_48_020").
      model_weights: The weight type ("original" or "soluble").
      local_path: Optional. If provided, loads from this local file.
      skeleton: Optional. If provided, loads weights *into* this
                eqx.Module skeleton. Required for .eqx format.

  Returns:
      The loaded model (as a raw PyTree or populated eqx.Module).

  Example:
      >>> # Load from HuggingFace (.eqx format, recommended)
      >>> model = load_model(model_version="v_48_020", model_weights="original")

  """
  if local_path:
    weights_file_path = local_path
  else:
    filename = f"{model_weights}_{model_version}.eqx"
    weights_file_path = hf_hub_download(
      repo_id=HF_REPO_ID,
      filename=f"eqx/{filename}",
      repo_type="model",
    )
  return eqx.tree_deserialise_leaves(weights_file_path, skeleton)


def load_model(
  model_version: MODEL_VERSION = "v_48_020",
  model_weights: MODEL_WEIGHTS = "original",
  local_path: str | None = None,
  key: jax.Array | None = None,
) -> PrxteinMPNN:
  """Load a fully instantiated PrxteinMPNN model with pre-trained weights.

  This is the recommended high-level API for loading models.

  Args:
      model_version: The model version (e.g., "v_48_020") or full name
                     (e.g., "original_v_48_020"). If full name is provided,
                     it will be parsed to extract weights and version.
      model_weights: The weight type ("original" or "soluble").
                     Ignored if model_version contains the full name.
      local_path: Optional. If provided, loads from this local .eqx file.
      key: Optional JAX random key. If None, uses default PRNGKey(0).

  Returns:
      A fully loaded PrxteinMPNN model ready for inference.

  Example:
      >>> from prxteinmpnn.io.weights import load_model
      >>> # Either specify separately:
      >>> model = load_model(model_version="v_48_020", model_weights="original")
      >>> # Or use combined name:
      >>> model = load_model("original_v_48_020")
      >>> # Model is ready for inference
      >>> seq, logits = model(coords, mask, res_idx, chain_idx, "unconditional")

  """
  # Import here to avoid circular dependency
  # This is intentional to break import cycles
  from prxteinmpnn.model import PrxteinMPNN  # noqa: PLC0415

  if key is None:
    key = jax.random.PRNGKey(0)

  # Parse model_version if it contains the full name (e.g., "original_v_48_020")
  if "_v_" in model_version:
    # Extract weights type and version from full name
    parts = model_version.split("_v_", 1)
    model_weights = parts[0]  # type: ignore[assignment]
    model_version = f"v_{parts[1]}"  # type: ignore[assignment]
  # Create skeleton with correct hyperparameters
  skeleton = PrxteinMPNN(
    node_features=NODE_FEATURES,
    edge_features=EDGE_FEATURES,
    hidden_features=HIDDEN_FEATURES,
    num_encoder_layers=NUM_ENCODER_LAYERS,
    num_decoder_layers=NUM_DECODER_LAYERS,
    vocab_size=VOCAB_SIZE,
    k_neighbors=K_NEIGHBORS,
    key=key,
  )

  # Load weights - returns the skeleton populated with weights
  loaded = load_weights(
    model_version=model_version,
    model_weights=model_weights,
    local_path=local_path,
    skeleton=skeleton,
  )

  # Type check for safety
  if not isinstance(loaded, PrxteinMPNN):
    msg = f"Expected PrxteinMPNN, got {type(loaded)}"
    raise TypeError(msg)
  return loaded
