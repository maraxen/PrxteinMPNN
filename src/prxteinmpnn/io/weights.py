"""Unified loader for PrxteinMPNN weights from Hugging Face Hub or local path."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import equinox as eqx
import jax
import jax.nn.initializers as init
from huggingface_hub import hf_hub_download

from prxteinmpnn.model import PrxteinMPNN

if TYPE_CHECKING:
  from jaxtyping import PyTree

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

NODE_FEATURES = 128
EDGE_FEATURES = 128
HIDDEN_FEATURES = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
K_NEIGHBORS = 48
VOCAB_SIZE = 21


def load_weights(
  model_version: MODEL_VERSION = "v_48_020",
  model_weights: MODEL_WEIGHTS | None = "original",
  local_path: str | None = None,
  skeleton: eqx.Module | None = None,
  key: jax.Array | None = None,
) -> PyTree | eqx.Module:
  """Load PrxteinMPNN weights from Hugging Face Hub or a local path.

  Args:
      model_version: The model version (e.g., "v_48_020").
      model_weights: The weight type ("original" or "soluble").
                     If None, reinitializes skeleton with Glorot normal.
      local_path: Optional. If provided, loads from this local file.
      skeleton: Optional. If provided, loads weights *into* this
                eqx.Module skeleton. Required for .eqx format.
      key: Optional JAX random key for reinitialization when model_weights is None.

  Returns:
      The loaded model (as a raw PyTree or populated eqx.Module).

  Example:
      >>> # Load from HuggingFace (.eqx format, recommended)
      >>> model = load_model(model_version="v_48_020", model_weights="original")
      >>> # Or reinitialize with Glorot normal
      >>> key = jax.random.PRNGKey(42)
      >>> model = load_model(model_version="v_48_020", model_weights=None, key=key)

  """
  if model_weights is None:
    if skeleton is None:
      msg = "skeleton is required when model_weights is None"
      raise ValueError(msg)
    if key is None:
      key = jax.random.PRNGKey(0)

    params, static = eqx.partition(skeleton, eqx.is_inexact_array)

    param_leaves = jax.tree_util.tree_leaves(params)
    keys = jax.random.split(key, len(param_leaves))

    min_weight_dims = 2

    def initialize_param(param: jax.Array, key: jax.Array) -> jax.Array:
      """Initialize parameter with appropriate distribution based on shape."""
      shape = param.shape
      if len(shape) >= min_weight_dims:
        return init.glorot_normal()(key, shape, param.dtype)
      return init.normal(stddev=0.01)(key, shape, param.dtype)

    initialized_leaves = [initialize_param(p, k) for p, k in zip(param_leaves, keys, strict=True)]
    new_params = jax.tree_util.tree_unflatten(
      jax.tree_util.tree_structure(params),
      initialized_leaves,
    )

    return eqx.combine(new_params, static)

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
  *,
  use_electrostatics: bool = False,
  use_vdw: bool = False,
  dropout_rate: float = 0.1,
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
      use_electrostatics: bool = False, Whether to include electrostatic features.
      use_vdw: bool = False, Whether to include van der Waals features.
      dropout_rate: Dropout rate (default: 0.1).

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
  if key is None:
    key = jax.random.PRNGKey(0)

  if "_v_" in model_version:
    parts = model_version.split("_v_", 1)
    model_weights = parts[0]
    model_version = f"v_{parts[1]}"

  physics_feature_dim = 0 + (5 if use_electrostatics else 0) + (0 if not use_vdw else 5)
  skeleton = PrxteinMPNN(
    node_features=NODE_FEATURES,
    edge_features=EDGE_FEATURES,
    hidden_features=HIDDEN_FEATURES,
    physics_feature_dim=physics_feature_dim if physics_feature_dim > 0 else None,
    num_encoder_layers=NUM_ENCODER_LAYERS,
    num_decoder_layers=NUM_DECODER_LAYERS,
    vocab_size=VOCAB_SIZE,
    k_neighbors=K_NEIGHBORS,
    dropout_rate=dropout_rate,
    key=key,
  )

  loaded = load_weights(
    model_version=model_version,
    model_weights=model_weights,
    local_path=local_path,
    skeleton=skeleton,
  )

  if not isinstance(loaded, PrxteinMPNN):
    msg = f"Expected PrxteinMPNN, got {type(loaded)}"
    raise TypeError(msg)
  return loaded
