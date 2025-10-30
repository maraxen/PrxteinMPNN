"""Main model API (functional legacy API).

prxteinmpnn.functional.model
"""

from typing import Literal

import jax
import jax.numpy as jnp
import joblib
from huggingface_hub import hf_hub_download
from jaxtyping import PyTree

ModelVersion = Literal[
  "v_48_002",
  "v_48_010",
  "v_48_020",
  "v_48_030",
]

ModelWeights = Literal["original", "soluble"]


def get_functional_model(
  model_version: ModelVersion = "v_48_020",
  model_weights: ModelWeights = "original",
  model_path: str | None = None,
) -> PyTree:
  """Create a ProteinMPNN model with specified configuration and weights.

  This is the legacy functional API for loading model parameters.

  Args:
    model_version: The model configuration to use.
    model_weights: The weights to load for the model.
    model_path: Optional path to a local model file. If None, download from HuggingFace Hub.

  Returns:
    A PyTree containing the model parameters.

  Raises:
    FileNotFoundError: If the model file does not exist.

  Example:
    >>> params = get_functional_model()

  """
  if model_path is not None:
    # Load from local path
    checkpoint_state = joblib.load(model_path)
    checkpoint_state = checkpoint_state["model_state_dict"]
    return jax.tree_util.tree_map(jnp.array, checkpoint_state)

  # Map weights and version to HF filenames
  if model_weights == "original":
    filename = f"original_{model_version}"
  elif model_weights == "soluble":
    filename = f"soluble_{model_version}"
  else:
    msg = f"Unknown model_weights: {model_weights}"
    raise ValueError(msg)

  # Download from HuggingFace Hub
  repo_id = "maraxen/prxteinmpnn"
  # Try .pkl extension (legacy functional)
  hf_filename = filename if filename.endswith(".pkl") else f"{filename}.pkl"
  try:
    hf_path = hf_hub_download(repo_id=repo_id, filename=hf_filename)
  except FileNotFoundError as err:
    msg = f"Could not download model weights from HuggingFace Hub: {hf_filename}\n{err}"
    raise FileNotFoundError(msg) from err

  checkpoint_state = joblib.load(hf_path)
  checkpoint_state = checkpoint_state["model_state_dict"]
  return jax.tree_util.tree_map(jnp.array, checkpoint_state)
