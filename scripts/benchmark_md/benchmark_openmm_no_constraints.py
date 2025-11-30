import os
import sys
import numpy as np
import pandas as pd
import jax.numpy as jnp
from termcolor import colored

# OpenMM Imports
try:
    import openmm
    import openmm.app as app
    import openmm.unit as unit
    # from pdbfixer import PDBFixer
except ImportError:
    print(colored("Error: OpenMM not found. Please install it.", "red"))
    sys.exit(1)

# PrxteinMPNN Imports (for Dihedral Calculation)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from prxteinmpnn.io.parsing import biotite as parsing_biotite
import biotite.structure.io.pdb as pdb
# We need compute_dihedrals_jax from benchmark_conformation.py, but it's not in a module.
# Let's copy the dihedral logic or import it if possible.
# Actually, let's just use mdtraj if available, or copy the numpy version.
# Since we are in the same repo, we can import from scripts.benchmark_md.benchmark_conformation?
# No, scripts are not usually modules.
# Let's just copy the dihedral calculation logic (numpy version).

def compute_dihedral_numpy(p1, p2, p3, p4):
    b0 = -1.0 * (p2 - p1)
    b1 = p3 - p2
    b2 = p4 - p3

    b1 /= np.linalg.norm(b1, axis=-1, keepdims=True)

    v = b0 - np.sum(b0 * b1, axis=-1, keepdims=True) * b1
    w = b2 - np.sum(b2 * b1, axis=-1, keepdims=True) * b1

    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1, v) * w, axis=-1)

    return np.degrees(np.arctan2(y, x))

def compute_phi_psi(coords, n_idx, ca_idx, c_idx):
    # coords: (N_atoms, 3)
    # Phi
    c_prev = coords[c_idx[:-1]]
    n_curr = coords[n_idx[1:]]
    ca_curr = coords[ca_idx[1:]]
    c_curr = coords[c_idx[1:]]
    
    phi_vals = compute_dihedral_numpy(c_prev, n_curr, ca_curr, c_curr)
    phi = np.concatenate([[0.0], phi_vals])
    
    # Psi
    n_curr = coords[n_idx[:-1]]
    ca_curr = coords[ca_idx[:-1]]
    c_curr = coords[c_idx[:-1]]
    n_next = coords[n_idx[1:]]
    
    psi_vals = compute_dihedral_numpy(n_curr, ca_curr, c_curr, n_next)
    psi = np.concatenate([psi_vals, [0.0]])
    
    return phi, psi

def is_allowed(phi, psi):
    is_alpha = (phi > -160) & (phi < -20) & (psi > -100) & (psi < 50)
    is_beta = (phi > -180) & (phi < -20) & (psi > 50) & (psi < 180)
    is_left_alpha = (phi > 20) & (phi < 100) & (psi > 0) & (psi < 100)
    return is_alpha | is_beta | is_left_alpha

def run_openmm_benchmark(pdb_id="1UAO", constraints=None, steps=100000, dt_fs=2.0):
    print(colored(f"Benchmarking OpenMM on {pdb_id} | Constraints: {constraints} | dt: {dt_fs} fs", "cyan"))
    
    # Setup
    pdb_path = f"data/pdb/{pdb_id}.pdb"
    if not os.path.exists(pdb_path):
        # Fetch if needed (using biotite or just assume it exists from previous runs)
        # For now assume it exists
        pass
        
    # Load with Hydride
    print("Loading with Hydride...")
    atom_array = parsing_biotite.load_structure_with_hydride(pdb_path, model=1)
    
    # Convert to OpenMM Topology/Positions via temporary PDB
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w+") as tmp:
        pdb_file_bio = pdb.PDBFile()
        pdb_file_bio.set_structure(atom_array)
        pdb_file_bio.write(tmp)
        tmp.flush()
        tmp.seek(0)
        pdb_file = app.PDBFile(tmp.name)
        topology = pdb_file.topology
        positions = pdb_file.positions
    
    # Force Field
    ff = app.ForceField('amber14-all.xml', 'implicit/obc2.xml')
    
    # System
    system = ff.createSystem(
        topology,
        nonbondedMethod=app.NoCutoff,
        constraints=constraints, # app.HBonds or None
        rigidWater=False,
        removeCMMotion=True
    )
    
    # Integrator
    integrator = openmm.LangevinIntegrator(300*unit.kelvin, 1.0/unit.picosecond, dt_fs*unit.femtoseconds)
    
    # Simulation
    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)
    simulation.minimizeEnergy()
    
    # Run
    print("  Running simulation...")
    # Collect samples
    phis = []
    psis = []
    
    # Extract indices
    atom_names = [a.name for a in topology.atoms()]
    res_indices = [r.index for r in topology.residues()]
    
    # Map residue index to N, CA, C indices
    n_idx = []
    ca_idx = []
    c_idx = []
    
    atoms = list(topology.atoms())
    for r in topology.residues():
        n = next((a.index for a in r.atoms() if a.name == "N"), None)
        ca = next((a.index for a in r.atoms() if a.name == "CA"), None)
        c = next((a.index for a in r.atoms() if a.name == "C"), None)
        if n is not None and ca is not None and c is not None:
            n_idx.append(n)
            ca_idx.append(ca)
            c_idx.append(c)
            
    n_idx = np.array(n_idx)
    ca_idx = np.array(ca_idx)
    c_idx = np.array(c_idx)
    
    # Sampling loop
    n_frames = 100
    stride = steps // n_frames
    
    valid_counts = []
    
    for i in range(n_frames):
        simulation.step(stride)
        state = simulation.context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        
        phi, psi = compute_phi_psi(pos, n_idx, ca_idx, c_idx)
        valid = is_allowed(phi, psi)
        # Exclude termini
        if len(valid) > 2:
            pct = np.mean(valid[1:-1]) * 100
            valid_counts.append(pct)
            
    avg_valid = np.mean(valid_counts)
    print(f"  Average Validity: {avg_valid:.2f}%")
    return avg_valid

if __name__ == "__main__":
    # 1. With Constraints (HBonds) - Should be ~100%
    # Note: OpenMM default dt is usually 2fs with constraints.
    res_constrained = run_openmm_benchmark("1UAO", constraints=app.HBonds, dt_fs=2.0)
    
    # 2. Without Constraints - Should be lower?
    # Note: Without constraints, 2fs might be unstable. Usually 1fs or 0.5fs is needed.
    # But JAX MD was running at what timestep?
    # benchmark_conformation.py uses simulate.run_simulation.
    # simulate.py: run_thermalization uses dt=1e-3 (1fs) by default?
    # Let's check simulate.py again.
    # Yes: dt: float = 1e-3,  # 1 fs - safer without constraints
    # So JAX MD is running at 1fs without constraints.
    
    res_unconstrained_1fs = run_openmm_benchmark("1UAO", constraints=None, dt_fs=1.0)
    
    # 3. Without Constraints at 2fs (Unstable?)
    # res_unconstrained_2fs = run_openmm_benchmark("1UAO", constraints=None, dt_fs=2.0)
    
    print("\nSummary:")
    print(f"OpenMM (HBonds, 2fs): {res_constrained:.2f}%")
    print(f"OpenMM (No Constraints, 1fs): {res_unconstrained_1fs:.2f}%")
