"""Type definitions for the PrxteinMPNN project."""

from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, PyTree
from optax import GradientTransformation

NodeFeatures = Int[Array, "num_atoms num_features"]  # Node features
EdgeFeatures = Float[Array, "num_atoms num_neighbors num_features"]  # Edge features
Message = Float[Array, "num_atoms num_neighbors num_features"]  # Message passing features
AtomicCoordinate = Float[Array, "3"]  # Atomic coordinates (x, y, z)
NeighborIndices = Int[Array, "num_atoms num_neighbors"]  # Indices of neighboring nodes
BackboneCoordinates = Float[AtomicCoordinate, "4 3"]  # Residue coordinates (x, y, z)
StructureAtomicCoordinates = Float[
  Array,
  "num_residues num_atoms 3",
]  # Atomic coordinates of the structure
AtomMask = Int[Array, "num_residues num_atoms"]  # Masks for atoms in the structure
AtomResidueIndex = Int[Array, "num_residues num_atoms"]  # Residue indices for atoms
AtomChainIndex = Int[Array, "num_residues num_atoms"]  # Chain indices for atoms
Parameters = Float[Array, "num_parameters"]  # Model parameters
ModelParameters = PyTree[str, "P"]
AlphaCarbonDistance = Float[Array, "num_atoms num_atoms"]  # Distances between alpha carbon atoms
Distances = Float[Array, "num_atoms num_neighbors"]  # Distances between nodes
AtomIndexPair = Int[Array, "2"]  # Pairs of atom indices for edges
AttentionMask = Bool[Array, "num_atoms num_atoms"]  # Attention mask for nodes
Logits = Float[Array, "num_residues num_classes"]  # Logits for classification
DecodingOrder = Int[Array, "num_residues"]  # Order of residues for autoregressive decoding
ProteinSequence = Int[Array, "num_residues"]  # Sequence of residues
OneHotProteinSequence = Float[Array, "num_residues num_classes"]  # One-hot encoded protein sequence
NodeEdgeFeatures = Float[
  Array,
  "num_atoms num_neighbors num_features",
]  # Combined node and edge features
SequenceEdgeFeatures = Float[
  Array,
  "num_residues num_neighbors num_features",
]  # Sequence edge features
AutoRegressiveMask = Bool[Array, "num_residues num_residues"]  # Mask for autoregressive decoding
InputBias = Float[Array, "num_residues num_classes"]  # Input bias for classification
InputLengths = Int[Array, "num_sequences"]  # Lengths of input sequences
BFactors = Float[Array, "num_residues num_atom_types"]  # B-factors for residues
ResidueIndex = Int[Array, "num_residues"]  # Index of residues in the structure
ChainIndex = Int[Array, "num_residues"]  # Index of chains in the structure

DecodingOrderInputs = tuple[PRNGKeyArray, int]
DecodingOrderOutputs = tuple[DecodingOrder, PRNGKeyArray]
CEELoss = Float[Array, ""]  # Cross-entropy loss
SamplingHyperparameters = tuple[float | int | Array | GradientTransformation, ...]

AlphaCarbonMask = Int[Array, "num_residues 3"]
BackboneDihedrals = Float[Array, "num_residues 2"]  # Dihedral angles for backbone atoms
BackboneNoise = Float[Array, "n"]  # Noise added to backbone coordinates
BackboneAtomCoordinates = Float[Array, "num_residues 4 3"]  # Backbone atom coordinates

Temperature = Float[Array, ""]  # Temperature for sampling
CategoricalJacobian = Float[Array, "num_residues num_classes num_residues num_classes"]
InterproteinMapping = Int[Array, "num_pairs max_length 2"]  # Mapping between protein pairs

EnsembleData = (
  Float[Array, "num_samples num_features"] | Float[Array, "n_batches n_samples n_features"]
)
Centroids = Float[Array, "num_clusters num_features"]
Labels = Int[Array, "num_samples"]

Means = Float[Array, "n_components n_features"]
Covariances = Float[Array, "n_components n_features n_features"]
Weights = Float[Array, "n_components"]
Responsibilities = Float[Array, "n_samples n_components"]
Converged = Bool[Array, ""]
LogLikelihood = Float[Array, ""]
ComponentCounts = Int[Array, "n_components"]
BIC = Float[Array, ""]
PCAInputData = Float[Array, "num_samples num_features"]
