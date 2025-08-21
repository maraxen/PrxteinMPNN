"""Core module for the PrxteinMPNN model.

prxteinmpnn.mpnn
"""

import pathlib
from typing import Literal

import jax
import jax.numpy as jnp
import joblib
from jaxtyping import PyTree

ModelVersion = Literal[
  "v_48_002.pkl",
  "v_48_010.pkl",
  "v_48_020.pkl",
  "v_48_030.pkl",
]

ModelWeights = Literal["original", "soluble"]


def get_mpnn_model(
  model_version: ModelVersion = "v_48_020.pkl",
  model_weights: ModelWeights = "original",
) -> PyTree:
  """Create a ProteinMPNN model with specified configuration and weights.

  Args:
    model_version: The model configuration to use.
    model_weights: The weights to load for the model.

  Returns:
    A PyTree containing the model parameters.

  Raises:
    FileNotFoundError: If the model file does not exist.

  Example:
    >>> params = get_mpnn_model()

  """
  base_dir = pathlib.Path(__file__).parent
  model_path = base_dir / "model" / model_weights / model_version
  if not model_path.exists():
    msg = f"Model file not found: {model_path}"
    raise FileNotFoundError(msg)
  checkpoint_state = joblib.load(model_path)
  checkpoint_state = checkpoint_state["model_state_dict"]
  return jax.tree_util.tree_map(jnp.array, checkpoint_state)
