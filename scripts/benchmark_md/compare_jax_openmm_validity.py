"""Compare JAX MD and OpenMM Validity Trajectories."""
import os
import sys
import time
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import biotite.structure.io.pdb as pdb
import biotite.structure as struc
import biotite.database.rcsb as rcsb
import argparse

# OpenMM Imports
import openmm as mm
import openmm.app as app
import openmm.unit as unit
# from pdbfixer import PDBFixer

# PrxteinMPNN imports
from prxteinmpnn.physics import simulate, force_fields, jax_md_bridge, system, constants
from prxteinmpnn.utils import residue_constants
from prxteinmpnn.io.parsing import biotite as parsing_biotite
from jax_md import space

# Enable x64 for physics
jax.config.update("jax_enable_x64", True)

# Constants
PDB_ID = "1UAO"
THERM_STEPS = 2000 # 4ps
PROD_STEPS = 5000  # 10ps
REPORT_INTERVAL = 100
DT_FS = 1.0
KB = 0.0019872041 # kcal/mol/K

def download_and_load_pdb(pdb_id, output_dir="data/pdb"):
    os.makedirs(output_dir, exist_ok=True)
    pdb_path = os.path.join(output_dir, f"{pdb_id.lower()}.pdb")
    if not os.path.exists(pdb_path):
        try:
            pdb_path = rcsb.fetch(pdb_id, "pdb", output_dir)
        except Exception:
            return None, None
    
    return pdb_path

def extract_system_with_hydride(pdb_path):
    print(f"Loading {pdb_path} with Hydride...")
    
    # Load with hydride (adds H if missing, removes solvent)
    # We must specify model=1 to get an AtomArray, not Stack, and to ensure Hydride works.
    atom_array = parsing_biotite.load_structure_with_hydride(pdb_path, model=1)
    
    # Extract data for JAX MD
    res_names = []
    atom_names = []
    atom_counts = []
    
    # Indices
    n_idx = []
    ca_idx = []
    c_idx = []
    
    idx_counter = 0
    for res_atoms in struc.residue_iter(atom_array):
        res_names.append(res_atoms[0].res_name)
        atom_counts.append(len(res_atoms))
        
        for atom in res_atoms:
            atom_names.append(atom.atom_name)
            
            if atom.atom_name == "N": n_idx.append(idx_counter)
            elif atom.atom_name == "CA": ca_idx.append(idx_counter)
            elif atom.atom_name == "C": c_idx.append(idx_counter)
            
            idx_counter += 1
            
    positions_np = atom_array.coord # (N, 3) in Angstroms
    
    # Convert to OpenMM Topology/Positions via temporary PDB
    # This ensures OpenMM gets exactly what we have in Biotite/Hydride
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w+") as tmp:
        pdb_file = pdb.PDBFile()
        pdb_file.set_structure(atom_array)
        pdb_file.write(tmp)
        tmp.flush()
        tmp.seek(0)
        pdb_file_omm = app.PDBFile(tmp.name)
        topology = pdb_file_omm.topology
        positions = pdb_file_omm.positions
            
    return positions_np, res_names, atom_names, np.array(n_idx), np.array(ca_idx), np.array(c_idx), atom_counts, topology, positions

def compute_dihedrals_jax(coords, n_idx, ca_idx, c_idx):
    """Compute Phi and Psi angles using JAX."""
    def compute_dihedral(p1, p2, p3, p4):
        b0 = -1.0 * (p2 - p1)
        b1 = p3 - p2
        b2 = p4 - p3

        b1 /= jnp.linalg.norm(b1, axis=-1, keepdims=True)

        v = b0 - jnp.sum(b0 * b1, axis=-1, keepdims=True) * b1
        w = b2 - jnp.sum(b2 * b1, axis=-1, keepdims=True) * b1

        x = jnp.sum(v * w, axis=-1)
        y = jnp.sum(jnp.cross(b1, v) * w, axis=-1)

        return jnp.degrees(jnp.arctan2(y, x))

    # Phi
    c_prev = coords[c_idx[:-1]]
    n_curr = coords[n_idx[1:]]
    ca_curr = coords[ca_idx[1:]]
    c_curr_phi = coords[c_idx[1:]]
    
    phi_vals = compute_dihedral(c_prev, n_curr, ca_curr, c_curr_phi)
    phi = jnp.concatenate([jnp.array([0.0]), phi_vals])

    # Psi
    n_curr_psi = coords[n_idx[:-1]]
    ca_curr_psi = coords[ca_idx[:-1]]
    c_curr_psi = coords[c_idx[:-1]]
    n_next = coords[n_idx[1:]]
    
    psi_vals = compute_dihedral(n_curr_psi, ca_curr_psi, c_curr_psi, n_next)
    psi = jnp.concatenate([psi_vals, jnp.array([0.0])])
    
    return phi, psi

def is_allowed_jax(phi, psi):
    """Check Ramachandran validity."""
    is_alpha = (phi > -160) & (phi < -20) & (psi > -100) & (psi < 50)
    is_beta = (phi > -180) & (phi < -20) & (psi > 50) & (psi < 180)
    is_left_alpha = (phi > 20) & (phi < 100) & (psi > 0) & (psi < 100)
    return is_alpha | is_beta | is_left_alpha

def run_openmm_simulation(topology, positions, n_idx, ca_idx, c_idx):
    print(f"Running OpenMM Simulation...")
    
    # pdb = app.PDBFile(pdb_path) # Removed
    forcefield = app.ForceField('amber14-all.xml', 'implicit/obc2.xml')
    
    # Add missing hydrogens if needed (though our PDB should have them)
    modeller = app.Modeller(topology, positions)
    # modeller.addHydrogens(forcefield) # Skip, we trust our PDB has H
    
    try:
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.CutoffNonPeriodic,
            nonbondedCutoff=2.0*unit.nanometer,
            constraints=app.HBonds,
        )
    except Exception as e:
        print(f"OpenMM System Creation Failed: {e}")
        # Try adding hydrogens if it failed?
        try:
            print("  Retrying with addHydrogens...")
            modeller.addHydrogens(forcefield)
            system = forcefield.createSystem(
                modeller.topology,
                nonbondedMethod=app.CutoffNonPeriodic,
                nonbondedCutoff=2.0*unit.nanometer,
                constraints=app.HBonds,
            )
        except Exception as e2:
            print(f"  Retry Failed: {e2}")
            return []

    integrator = mm.LangevinIntegrator(
        300*unit.kelvin, 
        1.0/unit.picosecond, 
        2.0*unit.femtosecond
    )
    
    simulation = app.Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    
    # Minimize
    print("  Minimizing...")
    simulation.minimizeEnergy()
    
    # Thermalize & Production
    results = []
    
    total_steps = THERM_STEPS + PROD_STEPS
    
    for step in range(0, total_steps, REPORT_INTERVAL):
        simulation.step(REPORT_INTERVAL)
        
        # Get State
        state = simulation.context.getState(getPositions=True, getEnergy=True)
        pos = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        pe = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        ke = state.getKineticEnergy().value_in_unit(unit.kilocalories_per_mole)
        
        # Temp
        n_atoms = pos.shape[0]
        temp = 2 * ke / (3 * n_atoms * KB)
        
        # Compute Validity
        phi, psi = compute_dihedrals_jax(jnp.array(pos), n_idx, ca_idx, c_idx)
        valid = is_allowed_jax(phi, psi)
        pct = float(jnp.mean(valid[1:-1]) * 100)
        
        phase = "Therm" if step < THERM_STEPS else "Prod"
        results.append({
            "step": step + REPORT_INTERVAL,
            "engine": "OpenMM",
            "phase": phase,
            "validity": pct,
            "phi_mean": float(jnp.mean(phi)),
            "psi_mean": float(jnp.mean(psi)),
            "pe": pe,
            "ke": ke,
            "temp": temp
        })
        
    # Get Minimized Positions for JAX MD
    state_min = simulation.context.getState(getPositions=True)
    minimized_pos = state_min.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    
    return results, minimized_pos

def run_jax_simulation(coords, params, n_idx, ca_idx, c_idx):
    print("Running JAX MD Simulation...")
    
    displacement_fn, shift_fn = space.free()
    
    # Setup
    key = jax.random.PRNGKey(0)
    
    # Energy Fn
    energy_fn = system.make_energy_fn(
        displacement_fn=displacement_fn,
        system_params=params,
        implicit_solvent=True,
        solvent_dielectric=78.5,
        solute_dielectric=1.0,
        dielectric_offset=constants.DIELECTRIC_OFFSET, 
        surface_tension=constants.SURFACE_TENSION # kcal/mol/A^2 (Matches OpenMM default)
    )
    
    # Init
    init_fn, apply_fn = simulate.rattle_langevin(
        energy_fn,
        shift_fn=shift_fn,
        dt=DT_FS * 1e-3, # fs to ps
        kT=KB * 300.0,
        gamma=1.0, # 1/ps
        constraints=(params["constrained_bonds"], params["constrained_bond_lengths"])
    )
    
    # Check Initial PE
    pe_initial = float(energy_fn(coords))
    print(f"  Initial PE (OpenMM Minimized): {pe_initial:.4f} kcal/mol")

    # Minimize (Refinement)
    print("  Minimizing (Refinement)...")
    r_min = simulate.run_minimization(energy_fn, coords, steps=100)
    
    # Project constraints after minimization to avoid shock
    if params["constrained_bonds"] is not None:
        print("  Projecting constraints after minimization...")
        # We need mass in correct shape (N, 1)
        masses = params["masses"]
        if masses.ndim == 1: masses = masses[:, None]
        
        # Need shift_fn
        displacement_fn, shift_fn = space.free()
        
        r_min = simulate.project_positions(
            r_min, 
            params["constrained_bonds"], 
            params["constrained_bond_lengths"], 
            masses, 
            shift_fn
        )
        
    # Check PE after minimization
    pe_min = float(energy_fn(r_min))
    print(f"  PE after minimization (and projection): {pe_min:.4f} kcal/mol")

    # Initialize State
    state = init_fn(key, r_min, mass=params["masses"])
    
    # Run Loop
    results = []
    
    total_steps = THERM_STEPS + PROD_STEPS
    
    # JIT the step function
    @jax.jit
    def step_fn(state, key):
        return apply_fn(state, key=key)
        
    current_state = state
    current_key = key
    
    for step in range(0, total_steps, REPORT_INTERVAL):
        # Run chunk
        for _ in range(REPORT_INTERVAL):
            current_key, subkey = jax.random.split(current_key)
            current_state = step_fn(current_state, subkey)
            
        pos = current_state.position
        
        # Compute Energy/Temp
        pe = float(energy_fn(pos))
        
        # KE = 0.5 * sum(p^2 / m)
        # Mass in simulate.py is passed as m_amu / 418.4
        # But NVTLangevinState stores mass.
        # Let's check if state.mass is the scaled mass.
        # Yes, init_fn in simulate.py sets it.
        
        # KE in kcal/mol
        ke = 0.5 * jnp.sum(current_state.momentum**2 / current_state.mass)
        ke = float(ke)
        
        # Temp
        # T = 2 * KE / (3 * N * KB)
        n_atoms = pos.shape[0]
        temp = 2 * ke / (3 * n_atoms * KB)
        
        phi, psi = compute_dihedrals_jax(pos, n_idx, ca_idx, c_idx)
        valid = is_allowed_jax(phi, psi)
        pct = float(jnp.mean(valid[1:-1]) * 100)
        
        phase = "Therm" if step < THERM_STEPS else "Prod"
        results.append({
            "step": step + REPORT_INTERVAL,
            "engine": "JAXMD",
            "phase": phase,
            "validity": pct,
            "phi_mean": float(jnp.mean(phi)),
            "psi_mean": float(jnp.mean(psi)),
            "pe": pe,
            "ke": ke,
            "temp": temp
        })
        
    return results

def main():
    print(f"Comparing JAX MD vs OpenMM on {PDB_ID}...")
    
    # Load Data
    pdb_path = download_and_load_pdb(PDB_ID)
    coords, res_names, atom_names, n_idx, ca_idx, c_idx, atom_counts, topology, positions = extract_system_with_hydride(pdb_path)
    
    # Rename terminals for Amber FF
    if len(res_names) > 0:
        res_names[0] = "N" + res_names[0]
        res_names[-1] = "C" + res_names[-1]
    
    # Load FF
    ff = force_fields.load_force_field("src/prxteinmpnn/physics/force_fields/eqx/protein19SB.eqx")
    
    print(f"DEBUG: FF Residue Templates Count: {len(ff.residue_templates) if ff.residue_templates else 0}")
    if ff.residue_templates:
        print(f"DEBUG: Sample Template Keys: {list(ff.residue_templates.keys())[:5]}")
    
    print(f"DEBUG: First 5 Res Names: {res_names[:5]}")
    print(f"DEBUG: First Residue Atoms: {atom_names[:10]}")
    
    # Debug ASP mismatch
    if 'ASP' in ff.residue_templates:
        template_bonds = ff.residue_templates['ASP']
        template_atoms = set()
        for a1, a2 in template_bonds:
            template_atoms.add(a1)
            template_atoms.add(a2)
        print(f"DEBUG: ASP Template Atoms: {sorted(list(template_atoms))}")
        
        # Find first ASP in res_names
        try:
            # Let's just print the first 100 atoms and look manually.
            print(f"DEBUG: First 100 Atom Names: {atom_names[:100]}")
        except ValueError:
            pass
            
    params = jax_md_bridge.parameterize_system(ff, res_names, atom_names, atom_counts=atom_counts)
    
    print(f"DEBUG: Coords Shape: {coords.shape}")
    print(f"DEBUG: Params Sigmas Shape: {params['sigmas'].shape}")
    print(f"DEBUG: Params Charges Shape: {params['charges'].shape}")
    
    # Run OpenMM
    omm_results, minimized_coords = run_openmm_simulation(topology, positions, n_idx, ca_idx, c_idx)
    
    # Run JAX MD (using minimized coords from OpenMM)
    # Note: We need to ensure coords shape matches. OpenMM might have added hydrogens?
    # But we loaded PDB with hydrogens already.
    # OpenMM Modeller might re-order?
    # Usually Modeller keeps order if we don't addHydrogens.
    # We skipped addHydrogens in run_openmm_simulation (commented out).
    # So order should be preserved.
    
    print(f"DEBUG: Using OpenMM Minimized Coords for JAX MD. Shape: {minimized_coords.shape}")
    jax_results = run_jax_simulation(jnp.array(minimized_coords), params, n_idx, ca_idx, c_idx)
    
    # Save
    all_results = omm_results + jax_results
    df = pd.DataFrame(all_results)
    df.to_csv("comparison_validity.csv", index=False)
    print("Saved to comparison_validity.csv")
    
    # Summary
    print("\nSummary (Last 5 steps):")
    print(df.tail(10))

if __name__ == "__main__":
    main()
