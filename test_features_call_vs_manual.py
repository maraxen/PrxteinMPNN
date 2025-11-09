"""Compare features.__call__() vs manual step-through with same inputs."""

import jax
import jax.numpy as jnp
import joblib
from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import Protein
from prxteinmpnn.utils.residue_constants import atom_order
from load_weights_comprehensive import load_prxteinmpnn_with_colabdesign_weights
from compare_pure_jax import compare

# Import internals
from prxteinmpnn.utils.coordinates import compute_backbone_coordinates, compute_backbone_distance
from prxteinmpnn.utils.radial_basis import compute_radial_basis
from prxteinmpnn.utils.graph import compute_neighbor_offsets


def main():
    print("="*80)
    print("FEATURES __call__() vs MANUAL WITH SAME INPUTS")
    print("="*80)

    # Load protein
    pdb_path = "tests/data/1ubq.pdb"
    protein_tuple = next(parse_input(pdb_path))
    protein = Protein.from_tuple(protein_tuple)

    # Load model
    key = jax.random.PRNGKey(42)
    colab_weights_path = "/tmp/ColabDesign/colabdesign/mpnn/weights/v_48_020.pkl"
    model = load_prxteinmpnn_with_colabdesign_weights(colab_weights_path, key=key)

    print("\n1. Calling features.__call__() directly...")
    prx_edge_call, prx_idx_call, _ = model.features(
        key, protein.coordinates, protein.mask,
        protein.residue_index, protein.chain_index, None
    )

    print("\n2. Manual step-through (replicating __call__ exactly)...")

    # This should be EXACTLY what __call__ does
    prng_key = key
    backbone_noise = jnp.array(0.0, dtype=jnp.float32)

    # Same as features.__call__
    from prxteinmpnn.utils.coordinates import apply_noise_to_coordinates
    noised_coordinates, prng_key = apply_noise_to_coordinates(
        prng_key, protein.coordinates, backbone_noise=backbone_noise
    )

    backbone_atom_coordinates = compute_backbone_coordinates(noised_coordinates)
    distances = compute_backbone_distance(backbone_atom_coordinates)

    distances_masked = jnp.array(
        jnp.where(
            (protein.mask[:, None] * protein.mask[None, :]).astype(bool),
            distances,
            jnp.inf,
        ),
    )

    top_k = jax.jit(jax.lax.top_k, static_argnames=("k",))
    k = min(48, protein.coordinates.shape[0])
    _, neighbor_indices = top_k(-distances_masked, k)
    neighbor_indices = jnp.array(neighbor_indices, dtype=jnp.int32)

    rbf = compute_radial_basis(backbone_atom_coordinates, neighbor_indices)
    neighbor_offsets = compute_neighbor_offsets(protein.residue_index, neighbor_indices)

    edge_chains = (protein.chain_index[:, None] == protein.chain_index[None, :]).astype(int)
    edge_chains_neighbors = jnp.take_along_axis(
        edge_chains, neighbor_indices, axis=1,
    )

    neighbor_offset_factor = jnp.clip(neighbor_offsets + 32, 0, 64)
    edge_chain_factor = (1 - edge_chains_neighbors) * 65
    encoded_offset = neighbor_offset_factor * edge_chains_neighbors + edge_chain_factor
    encoded_offset_one_hot = jax.nn.one_hot(encoded_offset, 66)

    # vmap over (N, K)
    encoded_positions = jax.vmap(jax.vmap(model.features.w_pos))(encoded_offset_one_hot)

    # Embed edges
    edges = jnp.concatenate([encoded_positions, rbf], axis=-1)
    edge_features = jax.vmap(jax.vmap(model.features.w_e))(edges)
    edge_features = jax.vmap(jax.vmap(model.features.norm_edges))(edge_features)

    # Project features
    edge_features = jax.vmap(jax.vmap(model.features.w_e_proj))(edge_features)

    prx_edge_manual = edge_features
    prx_idx_manual = neighbor_indices

    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)

    compare("\nNeighbor indices: __call__ vs manual", prx_idx_call, prx_idx_manual)
    corr = compare("\nEdge features: __call__ vs manual", prx_edge_call, prx_edge_manual)

    if corr > 0.9999:
        print("\n✅ PERFECT MATCH! __call__ and manual are identical")
        print("The 0.971 issue must be elsewhere!")
    else:
        print(f"\n❌ FOUND THE BUG! __call__ gives different results than manual stepping!")
        print(f"Correlation: {corr:.6f}")

if __name__ == "__main__":
    main()
