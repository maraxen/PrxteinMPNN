"""Unified loader for PrxteinMPNN weights from Hugging Face Hub or local path."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import equinox as eqx
import jax
import joblib
from huggingface_hub import hf_hub_download

if TYPE_CHECKING:
  from jaxtyping import PyTree

  from prxteinmpnn.eqx_new import PrxteinMPNN

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
HIDDEN_FEATURES = 512
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
K_NEIGHBORS = 48
VOCAB_SIZE = 21


def load_weights(
  model_version: MODEL_VERSION = "v_48_020",
  model_weights: MODEL_WEIGHTS = "original",
  local_path: str | None = None,
  skeleton: eqx.Module | None = None,
  use_eqx_format: bool = True,
) -> PyTree | eqx.Module:
  """Load PrxteinMPNN weights from Hugging Face Hub or a local path.

  Args:
      model_version: The model version (e.g., "v_48_020").
      model_weights: The weight type ("original" or "soluble").
      local_path: Optional. If provided, loads from this local file.
      skeleton: Optional. If provided, loads weights *into* this
                eqx.Module skeleton. Required for .eqx format.
      use_eqx_format: If True, loads from .eqx files (default).
                      If False, loads legacy .pkl files.

  Returns:
      The loaded model (as a raw PyTree or populated eqx.Module).

  Example:
      >>> # Load from HuggingFace (.eqx format, recommended)
      >>> model = load_model(model_version="v_48_020", model_weights="original")
      >>>
      >>> # Load legacy .pkl format
      >>> weights = load_weights(use_eqx_format=False, skeleton=None)

  """
  if local_path:
    weights_file_path = local_path
  elif use_eqx_format:
    # New .eqx format (recommended)
    filename = f"{model_weights}_{model_version}.eqx"
    weights_file_path = hf_hub_download(
      repo_id=HF_REPO_ID,
      filename=f"eqx/{filename}",
      repo_type="model",
    )
  else:
    # Legacy .pkl format
    filename = f"{model_version.replace('.pkl', '')}.pkl"
    weights_file_path = hf_hub_download(
      repo_id=HF_REPO_ID,
      filename=f"{model_weights}_{filename}",
    )

  # Load based on format
  if use_eqx_format or (local_path and local_path.endswith(".eqx")):
    if skeleton is None:
      msg = "skeleton must be provided when loading .eqx format"
      raise ValueError(msg)
    return eqx.tree_deserialise_leaves(weights_file_path, skeleton)

  # Legacy .pkl format
  if skeleton is None:
    skeleton = eqx.Module()
  return joblib.load(weights_file_path)


def load_model(
  model_version: MODEL_VERSION = "v_48_020",
  model_weights: MODEL_WEIGHTS = "original",
  local_path: str | None = None,
  key: jax.Array | None = None,
) -> PrxteinMPNN:
  """Load a fully instantiated PrxteinMPNN model with pre-trained weights.

  This is the recommended high-level API for loading models.

  Args:
      model_version: The model version (e.g., "v_48_020").
      model_weights: The weight type ("original" or "soluble").
      local_path: Optional. If provided, loads from this local .eqx file.
      key: Optional JAX random key. If None, uses default PRNGKey(0).

  Returns:
      A fully loaded PrxteinMPNN model ready for inference.

  Example:
      >>> from prxteinmpnn.io.weights import load_model
      >>> model = load_model(model_version="v_48_020", model_weights="original")
      >>> # Model is ready for inference
      >>> logits = model(X, E, mask)

  """
  # Import here to avoid circular dependency
  # This is intentional to break import cycles
  from prxteinmpnn.eqx_new import PrxteinMPNN  # noqa: PLC0415

  if key is None:
    key = jax.random.PRNGKey(0)

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
    use_eqx_format=True,
  )

  # Type check for safety
  if not isinstance(loaded, PrxteinMPNN):
    msg = f"Expected PrxteinMPNN, got {type(loaded)}"
    raise TypeError(msg)
  return loaded
