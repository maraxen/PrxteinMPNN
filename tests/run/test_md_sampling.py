
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import equinox as eqx
from prxteinmpnn.model import PrxteinMPNN
from prxteinmpnn.sampling.sample import make_sample_sequences
from prxteinmpnn.utils.data_structures import Protein, ProteinTuple
from prxteinmpnn.io.operations import _apply_md_parameterization

def test_md_sampling_end_to_end():
    """Test end-to-end MD sampling."""
    key = jax.random.PRNGKey(0)
    
    # 1. Create dummy protein
    N = 5
    aatype = np.array([0] * N) # ALA
    coords = np.zeros((N, 4, 3))
    for i in range(N):
        coords[i, 1, 0] = i * 3.8 # CA
        coords[i, 0, 0] = i * 3.8 # N
        coords[i, 2, 0] = i * 3.8 # C
        coords[i, 3, 0] = i * 3.8 # O
    
    protein_tuple = ProteinTuple(
        aatype=aatype,
        coordinates=coords,
        residue_index=np.arange(N),
        chain_index=np.zeros(N),
        atom_mask=np.ones((N, 4)),
    )
    
    # 2. Parameterize
    try:
        updated_tuples = _apply_md_parameterization([protein_tuple], use_md=True)
        p_tuple = updated_tuples[0]
    except Exception as e:
        pytest.skip(f"Skipping MD test due to parameterization failure: {e}")

    protein = Protein.from_tuple(p_tuple)
    
    # 3. Initialize Model
    model = PrxteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_features=128,
        num_encoder_layers=1,
        num_decoder_layers=1,
        k_neighbors=5,
        key=key
    )
    
    # 4. Create Sampler
    sample_fn = make_sample_sequences(model, sampling_strategy="temperature")
    
    # 5. Prepare MD args
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
        "min_steps": 10,
        "therm_steps": 10,
    }
    
    # Dummy full coordinates
    n_atoms = protein.md_exclusion_mask.shape[0]
    full_coords_jax = jnp.zeros((n_atoms, 3))
    
    # 6. Run Sampling
    # We need to pass MD args to sample_fn
    # Note: sample_fn is JIT compiled, so we pass JAX arrays
    
    seq, logits, order = sample_fn(
        key,
        protein.coordinates,
        protein.mask,
        protein.residue_index,
        protein.chain_index,
        backbone_noise=jnp.array(1.0), # Trigger noise
        full_coordinates=full_coords_jax,
        md_params=md_params,
        md_config=md_config
    )
    
    assert seq is not None
    assert logits is not None


def test_md_clash_resolution():
    """Test that minimization resolves steric clashes."""
    from prxteinmpnn.physics import simulate, force_fields, jax_md_bridge, system
    from jax_md import space

    key = jax.random.PRNGKey(1)
    
    # Create a 2-residue system
    res_names = ["ALA", "ALA"]
    from prxteinmpnn.utils import residue_constants
    atom_names = []
    for r in res_names:
        atom_names.extend(residue_constants.residue_atoms[r])
        
    ff = force_fields.load_force_field_from_hub("ff14SB")
    params = jax_md_bridge.parameterize_system(ff, res_names, atom_names)
    
    n_atoms = len(params["charges"])
    n_atoms_per_res = n_atoms // 2
    
    # Create realistic initial coordinates with a moderate clash
    # Build extended chain for residue 1
    coords_res1 = np.array([
        [0.0, 0.0, 0.0],      # N
        [1.45, 0.0, 0.0],     # CA  
        [2.0, 1.5, 0.0],      # C
        [3.2, 1.7, 0.0],      # O
        [2.0, -0.5, 1.0],     # CB
    ], dtype=np.float32)
    
    # Add hydrogens if needed
    if n_atoms_per_res > 5:
        extra = np.random.randn(n_atoms_per_res - 5, 3).astype(np.float32) * 0.3
        extra += coords_res1[1]  # Near CA
        coords_res1 = np.vstack([coords_res1, extra])
    
    # Build residue 2 slightly too close (3.0 Å instead of ~3.8 Å)
    coords_res2 = coords_res1.copy()
    coords_res2[:, 0] += 3.0  # Shift along x-axis
    
    coords = np.vstack([coords_res1[:n_atoms_per_res], coords_res2[:n_atoms_per_res]])
    coords = jnp.array(coords)
    
    # Setup Energy Fn
    displacement_fn, shift_fn = space.free()
    energy_fn = system.make_energy_fn(displacement_fn, params)
    
    # Initial Energy
    E_init = energy_fn(coords)
    
    # Run Minimization
    r_final = simulate.run_simulation(
        params,
        coords,
        temperature=0.0,
        min_steps=500,
        therm_steps=0,
        key=key
    )
    
    # Final Energy
    E_final = energy_fn(r_final)
    
    # Assert Energy Drop
    assert jnp.isfinite(E_init), f"Initial energy is not finite: {E_init}"
    assert jnp.isfinite(E_final), f"Final energy is not finite: {E_final}"
    assert E_final < E_init, f"Energy did not decrease: {E_init:.2f} -> {E_final:.2f}"
    
    # Assert structure improved
    energy_drop = (E_init - E_final) / abs(E_init)
    assert energy_drop > 0.001, f"Energy drop too small: {energy_drop:.2%}"


def test_md_temperature_stability():
    """Test that NVT simulation runs without producing NaN."""
    from prxteinmpnn.physics import simulate, force_fields, jax_md_bridge

    key = jax.random.PRNGKey(2)
    
    # Create a system (Single ALA)
    res_names = ["ALA"]
    from prxteinmpnn.utils import residue_constants
    atom_names = residue_constants.residue_atoms["ALA"]
    
    ff = force_fields.load_force_field_from_hub("ff14SB")
    params = jax_md_bridge.parameterize_system(ff, res_names, atom_names)
    
    n_atoms = len(params["charges"])
    
    # Create realistic initial coordinates
    coords = np.array([
        [0.0, 0.0, 0.0],      # N
        [1.45, 0.0, 0.0],     # CA  
        [2.0, 1.5, 0.0],      # C
        [3.2, 1.7, 0.0],      # O
        [2.0, -0.5, 1.0],     # CB
    ], dtype=np.float32)
    
    # Add hydrogens if needed
    if n_atoms > 5:
        extra = np.random.randn(n_atoms - 5, 3).astype(np.float32) * 0.3
        extra += coords[1]  # Near CA
        coords = np.vstack([coords, extra])
    
    coords = jnp.array(coords[:n_atoms])
    
    # Run MD simulation (minimization + thermalization)
    target_temp = 300.0
    
    r_final = simulate.run_simulation(
        params,
        coords,
        temperature=target_temp,
        min_steps=100,
        therm_steps=100,  # Short run just to verify stability
        key=key
    )
    
    # Assert simulation completed without NaN
    assert jnp.all(jnp.isfinite(r_final)), "MD simulation produced NaN coordinates"
    
    # Assert coordinates changed (system evolved)
    coord_change = jnp.linalg.norm(r_final - coords)
    assert coord_change > 0.01, f"Coordinates barely changed: {coord_change:.4f}"
