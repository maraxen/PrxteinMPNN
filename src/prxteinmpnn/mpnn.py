"""Core module for the PrxteinMPNN model."""

import enum
import pathlib
import warnings

import jax
import jax.numpy as jnp
import joblib
from flax.struct import dataclass, field
from jaxtyping import PyTree

from prxteinmpnn.utils.types import Sequence, StructureAtomicCoordinates


@dataclass(frozen=True)
class ModelInputs:
  """Dataclass for model inputs.

  Note that any of these can be stacked together to form a batch of inputs.
  """

  sequence_str: str = field(pytree_node=False, default="")
  """A string representation of the amino acid sequence."""

  structure_coordinates: StructureAtomicCoordinates = field(default_factory=lambda: jnp.array([]))
  """Structure atomic coordinates for the model input."""
  sequence: jax.Array = field(default_factory=lambda: jnp.array([]))
  """A sequence of amino acids for the model input. As MPNN-alphabet based array of integers."""
  structure: jax.Array = field(default_factory=lambda: jnp.array([]))
  """3D structure representation for the model input."""
  mask: jax.Array = field(default_factory=lambda: jnp.array([]))
  """Mask for the model input, indicating valid atoms structure."""
  residue_index: jax.Array = field(default_factory=lambda: jnp.array([]))
  """Index of residues in the structure, used for mapping atoms in structures to their residues."""
  chain_index: jax.Array = field(default_factory=lambda: jnp.array([]))
  """Index of chains in the structure, used for mapping atoms in structures to their chains."""
  lengths: jax.Array = field(default_factory=lambda: jnp.array([]))
  """Lengths of the sequences in the batch, used for padding and batching."""

  @property
  def sequence(self) -> Sequence:
    """Generate default sequence based on lengths.

    Returns:
      Zero sequence array with shape (sum(lengths),).

    """
    sequence

  @property
  def bias(self) -> jax.Array:
    """Generate default bias based on lengths.

    Returns:
      Zero bias array with shape (sum(lengths), 20).

    """
    if self.lengths.size == 0:
      return jnp.array([])
    return jnp.zeros((jnp.sum(self.lengths), 20))


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
