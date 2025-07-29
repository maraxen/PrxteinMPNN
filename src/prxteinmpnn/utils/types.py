"""Type definitions for the PrxteinMPNN project."""

from jaxtyping import Array, Bool, Float, Int, PyTree

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
ModelParameters = PyTree[
  str,
  Float[Array, "parameters"],
]  # Model parameters, e.g., weights
Distances = Float[Array, "num_atoms num_neighbors"]  # Distances between nodes
AtomIndexPair = Int[Array, "2"]  # Pairs of atom indices for edges
AttentionMask = Bool[Array, "num_atoms num_atoms"]  # Attention mask for nodes
Logits = Float[Array, "num_residues num_classes"]  # Logits for classification
DecodingOrder = Int[Array, "num_residues"]  # Order of residues for autoregressive decoding
Sequence = Int[Array, "num_residues"]  # Sequence of residues
NodeEdgeFeatures = Float[
  Array,
  "num_atoms num_neighbors num_features",
]  # Combined node and edge features
SequenceEdgeFeatures = Float[
  Array,
  "num_residues num_neighbors num_features",
]  # Sequence edge features
AutoRegressiveMask = Bool[Array, "num_residues num_residues"]  # Mask for autoregressive decoding
