"""
Debug: Compare the context tensor construction between PrxteinMPNN and ColabDesign.
"""

import jax
import jax.numpy as jnp
import numpy as np
import joblib
from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import Protein
from prxteinmpnn.utils.concatenate import concatenate_neighbor_nodes
from load_weights_comprehensive import load_prxteinmpnn_with_colabdesign_weights

# Load test data
pdb_path = "tests/data/1ubq.pdb"
protein_tuple = next(parse_input(pdb_path))
protein = Protein.from_tuple(protein_tuple)

# Load model
key = jax.random.PRNGKey(42)
prx_model = load_prxteinmpnn_with_colabdesign_weights(
    "/tmp/ColabDesign/colabdesign/mpnn/weights/v_48_020.pkl",
    key=key
)

# Extract features
edge_features, neighbor_indices, _ = prx_model.features(
    key,
    protein.coordinates,
    protein.mask,
    protein.residue_index,
    protein.chain_index,
    None,
)

# Run encoder
node_features, processed_edge_features = prx_model.encoder(
    edge_features,
    neighbor_indices,
    protein.mask,
)

print("After encoder:")
print(f"  node_features shape: {node_features.shape}")
print(f"  node_features[0, :5]: {node_features[0, :5]}")
print(f"  processed_edge_features shape: {processed_edge_features.shape}")
print(f"  processed_edge_features[0, 0, :5]: {processed_edge_features[0, 0, :5]}")

# Build context tensor (unconditional)
zeros_with_edges = concatenate_neighbor_nodes(
    jnp.zeros_like(node_features),
    processed_edge_features,
    neighbor_indices,
)
print("\nzeros_with_edges shape:", zeros_with_edges.shape)
print("zeros_with_edges[0, 0, :5]:", zeros_with_edges[0, 0, :5])
print("zeros_with_edges[0, 0, 128:133]:", zeros_with_edges[0, 0, 128:133])  # Should be zeros

layer_edge_features = concatenate_neighbor_nodes(
    node_features,
    zeros_with_edges,
    neighbor_indices,
)
print("\nlayer_edge_features shape:", layer_edge_features.shape)
print("layer_edge_features[0, 0, :5]:", layer_edge_features[0, 0, :5])  # h_E
print("layer_edge_features[0, 0, 128:133]:", layer_edge_features[0, 0, 128:133])  # zeros
print("layer_edge_features[0, 0, 256:261]:", layer_edge_features[0, 0, 256:261])  # h_V_neighbor

# Check neighbor_indices
print(f"\nneighbor_indices[0, 0] = {neighbor_indices[0, 0]}")
print(f"node_features[{neighbor_indices[0, 0]}, :5] = {node_features[neighbor_indices[0, 0], :5]}")
print("Expected in layer_edge_features[0, 0, 256:261]:", node_features[neighbor_indices[0, 0], :5])
