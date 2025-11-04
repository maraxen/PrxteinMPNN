"""Main model API (functional legacy API).

prxteinmpnn.functional.model
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import joblib
from huggingface_hub import hf_hub_download

if TYPE_CHECKING:
  from jaxtyping import PyTree

  from prxteinmpnn.eqx_new import PrxteinMPNN

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
  *,
  use_new_architecture: bool = True,
) -> PyTree | PrxteinMPNN:
  """Create a ProteinMPNN model with specified configuration and weights.

  This function serves as an adapter between the legacy functional API and the new
  Equinox architecture. It allows gradual migration by supporting both implementations.

  Args:
    model_version: The model configuration to use.
    model_weights: The weights to load for the model.
    model_path: Optional path to a local model file. If None, download from HuggingFace Hub.
    use_new_architecture: If True (default), return a PrxteinMPNN Equinox module.
                          If False, return legacy PyTree parameters.

  Returns:
    A PyTree containing the model parameters (legacy) or a PrxteinMPNN module (new).

  Raises:
    FileNotFoundError: If the model file does not exist.

  Example:
    >>> # New Equinox API (default)
    >>> model = get_functional_model()
    >>>
    >>> # Legacy functional API (if needed)
    >>> params = get_functional_model(use_new_architecture=False)

  """
  # NEW ARCHITECTURE: Use Equinox PrxteinMPNN
  if use_new_architecture:
    # Import here to avoid circular dependency
    from prxteinmpnn.io.weights import load_model  # noqa: PLC0415

    # Note: model_path is ignored in new architecture (use load_model's local_path instead)
    return load_model(
      model_version=model_version,
      model_weights=model_weights,
      local_path=model_path,
    )

  # LEGACY ARCHITECTURE: Use functional PyTree parameters
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
