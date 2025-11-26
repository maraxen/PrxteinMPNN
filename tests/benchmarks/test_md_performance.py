"""Benchmarking suite for MD integration performance."""
import time
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from prxteinmpnn.physics import simulate, force_fields, jax_md_bridge, system
from jax_md import space


@pytest.fixture(scope="module")
def force_field():
    """Load force field once for all tests."""
    return force_fields.load_force_field_from_hub("ff14SB")


@pytest.fixture
def ala_system(force_field):
    """Create a single ALA residue system."""
    from prxteinmpnn.utils import residue_constants
    
    res_names = ["ALA"]
    atom_names = residue_constants.residue_atoms["ALA"]
    params = jax_md_bridge.parameterize_system(force_field, res_names, atom_names)
    
    # Realistic initial coordinates
    n_atoms = len(params["charges"])
    coords = np.array([
        [0.0, 0.0, 0.0],      # N
        [1.45, 0.0, 0.0],     # CA  
        [2.0, 1.5, 0.0],      # C
        [3.2, 1.7, 0.0],      # O
        [2.0, -0.5, 1.0],     # CB
    ], dtype=np.float32)
    
    if n_atoms > 5:
        extra = np.random.randn(n_atoms - 5, 3).astype(np.float32) * 0.3
        extra += coords[1]
        coords = np.vstack([coords, extra])
    
    coords = jnp.array(coords[:n_atoms])
    
    return params, coords


@pytest.fixture
def dipeptide_system(force_field):
    """Create a 2-residue dipeptide system."""
    from prxteinmpnn.utils import residue_constants
    
    res_names = ["ALA", "ALA"]
    atom_names = []
    for r in res_names:
        atom_names.extend(residue_constants.residue_atoms[r])
    
    params = jax_md_bridge.parameterize_system(force_field, res_names, atom_names)
    
    n_atoms = len(params["charges"])
    n_atoms_per_res = n_atoms // 2
    
    # Build realistic dipeptide
    coords_res1 = np.array([
        [0.0, 0.0, 0.0],
        [1.45, 0.0, 0.0],
        [2.0, 1.5, 0.0],
        [3.2, 1.7, 0.0],
        [2.0, -0.5, 1.0],
    ], dtype=np.float32)
    
    if n_atoms_per_res > 5:
        extra = np.random.randn(n_atoms_per_res - 5, 3).astype(np.float32) * 0.3
        extra += coords_res1[1]
        coords_res1 = np.vstack([coords_res1, extra])
    
    coords_res2 = coords_res1.copy()
    coords_res2[:, 0] += 3.8  # Realistic peptide bond distance
    
    coords = np.vstack([coords_res1[:n_atoms_per_res], coords_res2[:n_atoms_per_res]])
    coords = jnp.array(coords)
    
    return params, coords


def test_benchmark_energy_evaluation(ala_system, benchmark):
    """Benchmark energy function evaluation."""
    params, coords = ala_system
    
    displacement_fn, _ = space.free()
    energy_fn = system.make_energy_fn(displacement_fn, params)
    
    # JIT compile
    energy_fn(coords)
    
    # Benchmark
    result = benchmark(lambda: energy_fn(coords).block_until_ready())
    
    print(f"\nEnergy evaluation: {result:.4f} kcal/mol")

def test_benchmark_minimization(ala_system, benchmark):
    """Benchmark minimization performance."""
    params, coords = ala_system
    key = jax.random.PRNGKey(0)
    
    # Warmup
    simulate.run_simulation(params, coords, temperature=0.0, min_steps=10, therm_steps=0, key=key)
    
    # Benchmark
    def run_min():
        result = simulate.run_simulation(
            params, coords, temperature=0.0, min_steps=100, therm_steps=0, key=key
        )
        return result.block_until_ready()
    
    result = benchmark(run_min)
    
    print(f"\nMinimization (100 steps) completed")


def test_benchmark_thermalization(ala_system, benchmark):
    """Benchmark NVT thermalization performance."""
    params, coords = ala_system
    key = jax.random.PRNGKey(1)
    
    # Warmup
    simulate.run_simulation(params, coords, temperature=300.0, min_steps=10, therm_steps=10, key=key)
    
    # Benchmark
    def run_nvt():
        result = simulate.run_simulation(
            params, coords, temperature=300.0, min_steps=0, therm_steps=100, key=key
        )
        return result.block_until_ready()
    
    result = benchmark(run_nvt)
    
    print(f"\nThermalization (100 steps) completed")


def test_benchmark_full_md_pipeline(dipeptide_system, benchmark):
    """Benchmark full MD pipeline (minimization + thermalization)."""
    params, coords = dipeptide_system
    key = jax.random.PRNGKey(2)
    
    # Warmup
    simulate.run_simulation(params, coords, temperature=300.0, min_steps=10, therm_steps=10, key=key)
    
    # Benchmark
    def run_full():
        result = simulate.run_simulation(
            params, coords, temperature=300.0, min_steps=200, therm_steps=500, key=key
        )
        return result.block_until_ready()
    
    result = benchmark(run_full)
    
    print(f"\nFull MD pipeline (200 min + 500 NVT steps) completed")


def test_energy_conservation(ala_system):
    """Test energy conservation during NVE simulation."""
    params, coords = ala_system
    
    displacement_fn, shift_fn = space.free()
    energy_fn = system.make_energy_fn(displacement_fn, params)
    
    # Run short NVE (no thermostat)
    from jax_md import simulate as jax_simulate
    
    init_fn, apply_fn = jax_simulate.nve(energy_fn, shift_fn, dt=2e-3)
    state = init_fn(jax.random.PRNGKey(3), coords, mass=1.0)
    
    energies = []
    for _ in range(100):
        state = apply_fn(state)
        E = energy_fn(state.position)
        KE = jax_simulate.quantity.kinetic_energy(velocity=state.velocity, mass=1.0)
        energies.append(E + KE)
    
    energies = np.array(energies)
    
    # Check energy drift
    energy_drift = np.std(energies) / np.abs(np.mean(energies))
    
    print(f"\nEnergy drift (NVE): {energy_drift:.2%}")
    print(f"Mean total energy: {np.mean(energies):.2f} kcal/mol")
    print(f"Std total energy: {np.std(energies):.2f} kcal/mol")
    
    # Energy should be conserved within 5% for short NVE
    assert energy_drift < 0.05, f"Energy drift too large: {energy_drift:.2%}"


def test_force_field_loading_performance(benchmark):
    """Benchmark force field loading (should be cached)."""
    
    def load_ff():
        ff = force_fields.load_force_field_from_hub("ff14SB")
        return ff
    
    result = benchmark(load_ff)
    
    print(f"\nForce field loading completed (should be fast due to caching)")


def test_parameterization_performance(force_field, benchmark):
    """Benchmark system parameterization."""
    from prxteinmpnn.utils import residue_constants
    
    res_names = ["ALA", "GLY", "VAL", "LEU", "ILE"]
    atom_names = []
    for r in res_names:
        atom_names.extend(residue_constants.residue_atoms[r])
    
    def parameterize():
        params = jax_md_bridge.parameterize_system(force_field, res_names, atom_names)
        return params
    
    result = benchmark(parameterize)
    
    print(f"\nParameterization of 5-residue system completed")
    print(f"Total atoms: {len(result['charges'])}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only", "--benchmark-sort=mean"])
