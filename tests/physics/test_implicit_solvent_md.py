"""End-to-end test for implicit solvent MD."""

import jax
import jax.numpy as jnp
import pytest
from prxteinmpnn.physics import force_fields, jax_md_bridge, system, simulate, generalized_born
from jax_md import space

def test_implicit_solvent_md_stability():
    """Test that implicit solvent MD runs without NaNs and maintains physical constraints."""
    
    # 1. Setup small system (Alanine Dipeptide or similar small peptide)
    # Using a simple sequence: ALA-ALA-ALA
    residues = ["ALA", "ALA", "ALA"]
    
    # Minimal atoms for ALA-ALA-ALA (N, CA, C, O, CB + Hydrogens if full atom)
    # For this test, we'll construct a minimal valid set of atoms that parameterize_system accepts.
    # parameterize_system needs atom names and residues.
    # We can mock the atoms.
    
    # ALA atoms: N, H, CA, HA, CB, HB1, HB2, HB3, C, O
    # We'll just use heavy atoms + backbone H for simplicity if allowed, 
    # but mbondi2 needs specific atoms.
    # Let's use a mock function or just manually define a small valid structure.
    
    # Better: Use a known valid small structure or construct one.
    # Let's construct a linear chain of 3 ALAs.
    
    # ALA atoms in residue_constants order: ["C", "CA", "CB", "N", "O"]
    
    atom_names = []
    res_names = []
    coords_list = []
    
    # Simple geometry generation (Zig-Zag to avoid clashes)
    for i, res in enumerate(residues):
        offset_x = i * 3.8
        
        # N
        atom_names.append("N")
        res_names.append(res)
        coords_list.append([offset_x, 0.0, 0.0])
        
        # CA
        atom_names.append("CA")
        res_names.append(res)
        coords_list.append([offset_x + 1.46, 0.0, 0.0])
        
        # C
        atom_names.append("C")
        res_names.append(res)
        coords_list.append([offset_x + 2.0, 1.2, 0.0])
        
        # O
        atom_names.append("O")
        res_names.append(res)
        coords_list.append([offset_x + 1.8, 2.4, 0.0])
        
        # CB
        atom_names.append("CB")
        res_names.append(res)
        coords_list.append([offset_x + 1.8, -1.0, 1.0]) # Sidechain out of plane
        
    coords = jnp.array(coords_list)
    
    # 2. Parameterize
    ff = force_fields.load_force_field_from_hub("ff14SB")
    # parameterize_system expects list of residue names (sequence), not per-atom residue names
    params = jax_md_bridge.parameterize_system(ff, residues, atom_names)
    
    # Check if gb_radii are assigned
    assert "gb_radii" in params
    assert params["gb_radii"] is not None
    assert params["gb_radii"].shape == (len(atom_names),)
    
    # Check specific radii values
    for i, (name, radius) in enumerate(zip(atom_names, params["gb_radii"])):
        if name == "N":
            assert jnp.isclose(radius, 1.55, atol=1e-3)
        elif name == "C" or name == "CA" or name == "CB":
            assert jnp.isclose(radius, 1.70, atol=1e-3)
        elif name == "O":
            assert jnp.isclose(radius, 1.50, atol=1e-3)
            
    # 3. Run MD (Minimization only first)
    key = jax.random.PRNGKey(42)
    
    displacement_fn, _ = space.free()
    energy_fn = system.make_energy_fn(displacement_fn, params, implicit_solvent=True)
    
    print("Running Minimization...")
    r_min = simulate.run_minimization(energy_fn, coords, steps=100, dt_start=1e-4)
    
    print(f"Minimized Coords: {r_min}")
    assert jnp.all(jnp.isfinite(r_min))
    
    min_energy = energy_fn(r_min)
    print(f"Minimized Energy: {min_energy}")
    assert jnp.isfinite(min_energy)
    
    # 4. Run short thermalization
    print("Running Thermalization...")
    r_final = simulate.run_thermalization(
        energy_fn, 
        r_min, 
        temperature=300.0, 
        steps=50, 
        dt=1e-4, # Small timestep for stability
        key=key
    )
    
    final_energy = energy_fn(r_final)
    print(f"Final Energy: {final_energy}")
    print(f"Final Coords: {r_final}")
    
    assert jnp.all(jnp.isfinite(r_final))
    assert jnp.isfinite(final_energy)

if __name__ == "__main__":
    test_implicit_solvent_md_stability()
