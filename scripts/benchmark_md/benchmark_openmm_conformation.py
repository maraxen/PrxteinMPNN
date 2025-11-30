"""Benchmark Conformational Validity using OpenMM."""
import os
import sys
import numpy as np
import pandas as pd
import biotite.structure.io.pdb as pdb
import biotite.structure as struc
import biotite.database.rcsb as rcsb
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

# PrxteinMPNN Imports (for residue constants and dihedral calc)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from prxteinmpnn.utils import residue_constants
from prxteinmpnn.io.parsing import biotite as parsing_biotite

# Constants
PDB_ID = "1UAO"
NUM_SAMPLES = 8
MD_STEPS = 100
MD_THERM = 500
TEMP_KELVIN = 300.0

def download_and_load_pdb(pdb_id, output_dir="data/pdb"):
    os.makedirs(output_dir, exist_ok=True)
    pdb_path = os.path.join(output_dir, f"{pdb_id.lower()}.pdb")
    if not os.path.exists(pdb_path):
        try:
            pdb_path = rcsb.fetch(pdb_id, "pdb", output_dir)
        except Exception:
            return None
    return pdb_path

def compute_dihedrals_numpy(coords, n_idx, ca_idx, c_idx):
    """Compute Phi and Psi angles using NumPy."""
    def compute_dihedral(p1, p2, p3, p4):
        b0 = -1.0 * (p2 - p1)
        b1 = p3 - p2
        b2 = p4 - p3

        b1 /= np.linalg.norm(b1, axis=-1, keepdims=True)

        v = b0 - np.sum(b0 * b1, axis=-1, keepdims=True) * b1
        w = b2 - np.sum(b2 * b1, axis=-1, keepdims=True) * b1

        x = np.sum(v * w, axis=-1)
        y = np.sum(np.cross(b1, v) * w, axis=-1)

        return np.degrees(np.arctan2(y, x))

    # Phi
    c_prev = coords[c_idx[:-1]]
    n_curr = coords[n_idx[1:]]
    ca_curr = coords[ca_idx[1:]]
    c_curr_phi = coords[c_idx[1:]]
    
    phi_vals = compute_dihedral(c_prev, n_curr, ca_curr, c_curr_phi)
    phi = np.concatenate([np.array([0.0]), phi_vals])

    # Psi
    n_curr_psi = coords[n_idx[:-1]]
    ca_curr_psi = coords[ca_idx[:-1]]
    c_curr_psi = coords[c_idx[:-1]]
    n_next = coords[n_idx[1:]]
    
    psi_vals = compute_dihedral(n_curr_psi, ca_curr_psi, c_curr_psi, n_next)
    psi = np.concatenate([psi_vals, np.array([0.0])])
    
    return phi, psi

def is_allowed_numpy(phi, psi):
    """Check Ramachandran validity."""
    is_alpha = (phi > -160) & (phi < -20) & (psi > -100) & (psi < 50)
    is_beta = (phi > -180) & (phi < -20) & (psi > 50) & (psi < 180)
    is_left_alpha = (phi > 20) & (phi < 100) & (psi > 0) & (psi < 100)
    return is_alpha | is_beta | is_left_alpha

def get_atom_indices(atom_array):
    """Get indices for N, CA, C."""
    n_indices = []
    ca_indices = []
    c_indices = []
    
    current_atom_idx = 0
    
    # Filter to protein only first
    atom_array = atom_array[struc.filter_amino_acids(atom_array)]
    chains = struc.get_chains(atom_array)
    if len(chains) > 0:
        atom_array = atom_array[atom_array.chain_id == chains[0]]

    for res in struc.residue_iter(atom_array):
        res_name = res[0].res_name
        if res_name not in residue_constants.restype_3to1: continue
        
        # Find indices relative to this residue start
        res_n_idx = -1
        res_ca_idx = -1
        res_c_idx = -1
        
        for i, atom_name in enumerate(res.atom_name):
            if atom_name == "N": res_n_idx = current_atom_idx + i
            if atom_name == "CA": res_ca_idx = current_atom_idx + i
            if atom_name == "C": res_c_idx = current_atom_idx + i
            
        if res_n_idx != -1 and res_ca_idx != -1 and res_c_idx != -1:
            n_indices.append(res_n_idx)
            ca_indices.append(res_ca_idx)
            c_indices.append(res_c_idx)
            
        current_atom_idx += len(res)
        
    return np.array(n_indices), np.array(ca_indices), np.array(c_indices)

def run_openmm_benchmark():
    print(f"Benchmarking OpenMM Validity on {PDB_ID}...")
    
    # 1. Setup System
    pdb_path = download_and_load_pdb(PDB_ID)
    if pdb_path is None:
        print("Error: Could not download PDB.")
        return

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

    # Load Force Field
    # Using amber14-all + obc2 to match JAX MD setup
    try:
        ff = app.ForceField('amber14-all.xml', 'implicit/obc2.xml')
    except Exception:
        print("Error loading force field.")
        return

    system = ff.createSystem(
        topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None, # No constraints to match JAX MD
        rigidWater=False,
        removeCMMotion=False
    )

    integrator = openmm.LangevinIntegrator(
        TEMP_KELVIN * unit.kelvin,
        1.0 / unit.picosecond, # Friction 1/ps
        2.0 * unit.femtoseconds # 2 fs step
    )
    
    platform = openmm.Platform.getPlatformByName('Reference') # CPU Reference
    simulation = app.Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions)

    # Get Indices for Dihedrals
    # We use the atom_array we already loaded
    n_idx, ca_idx, c_idx = get_atom_indices(atom_array)

    # 2. Run Simulation
    print("Minimizing...")
    simulation.minimizeEnergy(maxIterations=MD_STEPS) # Match JAX MD min steps roughly
    
    print("Thermalizing...")
    simulation.step(MD_THERM) # Match JAX MD therm steps
    
    # 3. Sample
    print(f"Sampling {NUM_SAMPLES} conformations...")
    valid_counts = []
    
    for i in range(NUM_SAMPLES):
        simulation.step(100) # Step between samples
        state = simulation.context.getState(getPositions=True)
        coords = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        
        # Extract backbone coords for dihedral calc
        # Note: OpenMM coords include hydrogens, JAX MD usually does too if all-atom.
        # But our indices are based on the full atom array, so it should work.
        
        phi, psi = compute_dihedrals_numpy(coords, n_idx, ca_idx, c_idx)
        valid = is_allowed_numpy(phi, psi)
        
        if len(valid) > 2:
            pct = np.mean(valid[1:-1]) * 100
            valid_counts.append(pct)
            
    mean_validity = np.mean(valid_counts)
    print(f"\nOpenMM Results for {PDB_ID}:")
    print(f"  Mean Validity: {mean_validity:.2f}%")
    print(f"  Samples: {valid_counts}")

if __name__ == "__main__":
    run_openmm_benchmark()
