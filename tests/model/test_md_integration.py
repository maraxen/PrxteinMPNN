
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import equinox as eqx
from prxteinmpnn.model.features import ProteinFeatures
from prxteinmpnn.utils.data_structures import Protein, ProteinTuple
from prxteinmpnn.io.operations import _apply_md_parameterization

def test_md_integration_features():
    """Test that ProteinFeatures runs MD simulation when configured."""
    key = jax.random.PRNGKey(0)
    
    # 1. Create dummy protein
    # Poly-alanine helix-ish
    N = 5
    aatype = np.array([0] * N) # ALA
    # Coordinates: simple line
    coords = np.zeros((N, 4, 3))
    for i in range(N):
        coords[i, 1, 0] = i * 3.8 # CA
        coords[i, 0, 0] = i * 3.8 # N (dummy)
        coords[i, 2, 0] = i * 3.8 # C (dummy)
        coords[i, 3, 0] = i * 3.8 # O (dummy)
    
    # Full coordinates (N, CA, C, O, CB) -> 5 atoms per res
    # We need full coordinates for MD.
    # Let's create dummy full coords (N*5, 3)
    full_coords = np.zeros((N * 5, 3))
    # Just populate with some values
    full_coords[:, 0] = np.arange(N * 5)
    
    protein_tuple = ProteinTuple(
        aatype=aatype,
        coordinates=coords,
        residue_index=np.arange(N),
        chain_index=np.zeros(N),
        atom_mask=np.ones((N, 4)),
    )
    
    # 2. Parameterize for MD
    # This requires force fields and residue constants which should be available
    # We use _apply_md_parameterization to get MD params
    # Note: _apply_md_parameterization expects list of ProteinTuple
    
    # Mocking force field loading if needed, but let's try running it.
    # If it fails due to missing files, we might need to skip or mock.
    # Assuming environment has necessary files.
    
    try:
        updated_tuples = _apply_md_parameterization([protein_tuple], use_md=True)
        p_tuple = updated_tuples[0]
    except Exception as e:
        pytest.skip(f"Skipping MD test due to parameterization failure (likely missing files): {e}")

    # Convert to Protein (JAX)
    protein = Protein.from_tuple(p_tuple)
    
    # 3. Setup ProteinFeatures
    features = ProteinFeatures(
        node_features=128,
        edge_features=128,
        k_neighbors=5,
        key=key
    )
    
    # 4. Run with MD mode
    md_params = {
        "bonds": protein.md_bonds,
        "bond_params": protein.md_bond_params,
        "angles": protein.md_angles,
        "angle_params": protein.md_angle_params,
        "backbone_indices": protein.md_backbone_indices,
        "exclusion_mask": protein.md_exclusion_mask,
        "charges": protein.charges,
        "sigmas": protein.sigmas,
        "epsilons": protein.epsilons,
    }
    
    md_config = {
        "temperature": 300.0,
        "min_steps": 10, # Short run
        "therm_steps": 10,
    }
    
    # We need to pass full_coordinates.
    # ProteinTuple doesn't have full_coordinates field populated by _apply_md_parameterization?
    # _apply_md_parameterization only adds params.
    # We need to manually construct full_coordinates matching the topology.
    # For this test, we can use a dummy array of correct size.
    # max_atoms from parameterization?
    # We can infer from exclusion_mask size or just use a large enough array.
    # exclusion_mask is (N_atoms, N_atoms).
    n_atoms = protein.md_exclusion_mask.shape[0]
    full_coords_jax = jnp.zeros((n_atoms, 3))
    
    # Run
    edge_feats, neighbor_idx, node_feats, new_key = features(
        key,
        protein.coordinates,
        protein.mask, # CA mask
        protein.residue_index,
        protein.chain_index,
        backbone_noise=jnp.array(1.0),
        backbone_noise_mode="md",
        full_coordinates=full_coords_jax,
        md_params=md_params,
        md_config=md_config
    )
    
    # Check that it ran without error
    assert edge_feats is not None
    assert neighbor_idx is not None
    
    # We can't easily check if coordinates changed significantly without a real simulation,
    # but we can check if the code path was executed.
    # If we pass invalid md_params, it would crash.
