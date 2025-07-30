"""Core module for the PrxteinMPNN model.

prxteinmpnn.mpnn
"""

import enum
import pathlib

import jax
import jax.numpy as jnp
import joblib
from jaxtyping import PyTree


class ProteinMPNNModelVersion(enum.Enum):
  """Enum for different ProteinMPNN model configurations."""

  V_48_002 = "v_48_002.pkl"
  V_48_010 = "v_48_010.pkl"
  V_48_020 = "v_48_020.pkl"
  V_48_030 = "v_48_030.pkl"


class ModelWeights(enum.Enum):
  """Enum for different sets of model weights."""

  DEFAULT = "original"
  SOLUBLE = "soluble"


def get_mpnn_model(
  model_version: ProteinMPNNModelVersion = ProteinMPNNModelVersion.V_48_002,
  model_weights: ModelWeights = ModelWeights.DEFAULT,
) -> PyTree:
  """Create a ProteinMPNN model with specified configuration and weights.

  Args:
    model_version: The model configuration to use.
    model_weights: The weights to load for the model.

  Returns:
    A dictionary containing the model parameters.

  """
  model_path = pathlib.Path("models") / model_weights.value / model_version.value
  checkpoint_state = joblib.load(model_path)
  return jax.tree_util.tree_map(jnp.array, checkpoint_state)
