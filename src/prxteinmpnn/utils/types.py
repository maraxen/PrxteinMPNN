"""Type definitions for the PrxteinMPNN project."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, PyTree
from optax import GradientTransformation

if TYPE_CHECKING:
  from prxteinmpnn.model.mpnn import PrxteinLigandMPNN, PrxteinMPNN

ArrayLike = Union[Array, np.ndarray]

NodeFeatures = Float[ArrayLike, "num_atoms num_features"]  # Node features
Scalar = Int[ArrayLike, ""]
ScalarFloat = Float[ArrayLike, ""]
EdgeFeatures = Float[ArrayLike, "num_atoms num_neighbors num_features"]  # Edge features
Message = Float[ArrayLike, "num_atoms num_neighbors num_features"]  # Message passing features
AtomicCoordinate = Float[ArrayLike, "3"]  # Atomic coordinates (x, y, z)
NeighborIndices = Int[ArrayLike, "num_atoms num_neighbors"]  # Indices of neighboring nodes
BackboneCoordinates = Float[ArrayLike, "4 3"]  # Residue coordinates (x, y, z)
StructureAtomicCoordinates = Float[
  ArrayLike,
  "num_residues num_atoms 3",
]  # Atomic coordinates of the structure
AtomMask = Int[ArrayLike, "num_residues num_atoms"]  # Masks for atoms in the structure
AtomResidueIndex = Int[ArrayLike, "num_residues num_atoms"]  # Residue indices for atoms
AtomChainIndex = Int[ArrayLike, "num_residues num_atoms"]  # Chain indices for atoms
Parameters = Float[ArrayLike, "num_parameters"]  # Model parameters
ModelParameters = PyTree[str, "P"]
# Type union for migration: supports both legacy PyTree and new Equinox model
# Using Union with string annotation to avoid runtime import
Model = Union["PrxteinMPNN", "PrxteinLigandMPNN", ModelParameters]
AlphaCarbonDistance = Float[
  ArrayLike,
  "num_atoms num_atoms",
]  # Distances between alpha carbon atoms
Distances = Float[ArrayLike, "num_atoms num_neighbors"]  # Distances between nodes
AtomIndexPair = Int[ArrayLike, "2"]  # Pairs of atom indices for edges
AttentionMask = Bool[ArrayLike, "num_atoms num_atoms"]  # Attention mask for nodes
Logits = Float[ArrayLike, "num_residues num_classes"]  # Logits for classification
DecodingOrder = Int[ArrayLike, "num_residues"]  # Order of residues for autoregressive decoding
ProteinSequence = Int[ArrayLike, "num_residues"]  # Sequence of residues
OneHotProteinSequence = Float[
  ArrayLike,
  "num_residues num_classes",
]  # One-hot encoded protein sequence
NodeEdgeFeatures = Float[
  ArrayLike,
  "num_atoms num_neighbors num_features",
]  # Combined node and edge features
SequenceEdgeFeatures = Float[
  ArrayLike,
  "num_residues num_neighbors num_features",
]  # Sequence edge features
AutoRegressiveMask = Bool[
  ArrayLike,
  "num_residues num_residues",
]  # Mask for autoregressive decoding
InputBias = Float[ArrayLike, "num_residues num_classes"]  # Input bias for classification
InputLengths = Int[ArrayLike, "num_sequences"]  # Lengths of input sequences
BFactors = Float[ArrayLike, "num_residues num_atom_types"]  # B-factors for residues
ResidueIndex = Int[ArrayLike, "num_residues"]  # Index of residues in the structure
ChainIndex = Int[ArrayLike, "num_residues"]  # Index of chains in the structure

DecodingOrderInputs = tuple[PRNGKeyArray, int]
DecodingOrderOutputs = tuple[DecodingOrder, PRNGKeyArray]
CEELoss = Float[ArrayLike, ""]  # Cross-entropy loss
SamplingHyperparameters = tuple[float | int | Array | GradientTransformation, ...]

AlphaCarbonMask = Int[ArrayLike, "num_residues"]
BackboneDihedrals = Float[ArrayLike, "num_residues 3"]  # Dihedral angles for backbone atoms
BackboneNoise = Float[ArrayLike, "n"]  # Noise added to backbone coordinates
BackboneAtomCoordinates = Float[ArrayLike, "num_residues 4 3"]  # Backbone atom coordinates
GroupMask = Bool[ArrayLike, "num_residues"]
LinkMask = Float[ArrayLike, "num_residues num_neighbors"]
TieGroupMap = Int[ArrayLike, "num_residues"]

Temperature = Float[ArrayLike, ""]  # Temperature for sampling
CategoricalJacobian = Float[ArrayLike, "num_residues num_classes num_residues num_classes"]
InterproteinMapping = Int[ArrayLike, "num_pairs max_length 2"]  # Mapping between protein pairs

EnsembleData = (
  Float[ArrayLike, "num_samples num_features"] | Float[ArrayLike, "n_batches n_samples n_features"]
)
Centroids = Float[ArrayLike, "num_clusters num_features"]
Labels = Int[ArrayLike, "num_samples"]

Means = Float[ArrayLike, "n_components n_features"]
Covariances = Float[ArrayLike, "n_components n_features n_features"]
Weights = Float[ArrayLike, "n_components"]
Responsibilities = Float[ArrayLike, "n_samples n_components"]
Converged = Bool[ArrayLike, ""]
LogLikelihood = Float[ArrayLike, ""]
ComponentCounts = Int[ArrayLike, "n_components"]
BIC = Float[ArrayLike, ""]
PCAInputData = Float[ArrayLike, "num_samples num_features"]


class TrainingMetrics(dict):
  """Dictionary containing training metrics."""

  loss: float
  accuracy: float
  perplexity: float
  learning_rate: float
