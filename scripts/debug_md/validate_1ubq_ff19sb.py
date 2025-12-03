import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
from jax_md import space, energy
import pandas as pd
from termcolor import colored

# OpenMM Imports
try:
    import openmm
    import openmm.app as app
    import openmm.unit as unit
    import openmm.unit as unit
except ImportError:
    print(colored("Error: OpenMM not found. Please install it.", "red"))
    sys.exit(1)

# PrxteinMPNN Imports
# Assuming the script is run from the root or scripts/debug_md
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from prxteinmpnn.physics import force_fields, jax_md_bridge, cmap, system, generalized_born, bonded
print(f"DEBUG: Loaded system module from: {system.__file__}")
print(f"DEBUG: Loaded generalized_born module from: {generalized_born.__file__}")
from prxteinmpnn.utils import residue_constants
from prxteinmpnn.io.parsing import biotite as parsing_biotite
import biotite.structure.io.pdb as pdb

# =============================================================================
# Configuration
# =============================================================================
PDB_PATH = "data/pdb/1UBQ.pdb"
FF_EQX_PATH = "src/prxteinmpnn/physics/force_fields/eqx/protein19SB.eqx"
OPENMM_FF_XMLS = ['amber14-all.xml', 'implicit/obc2.xml'] # Using amber14-all which includes ff19SB usually or similar
# Note: OpenMM might not have 'amber19sb.xml' by default in older versions, checking...
# If amber19sb.xml is not found, we might need to use amber14/protein.ff14SB.xml as fallback or ensure 19SB is available.
# Let's try 'amber14-all.xml' first as it is standard, but the user asked for ff19SB.
# If the user provided protein19SB.eqx, they likely want exact match.
# Let's try to load 'amber14/protein.ff14SB.xml' if 19SB isn't there, but let's stick to user request if possible.
# Actually, standard OpenMM often has 'amber14-all.xml'. Let's check if we can find 19SB.
# For now, we will use 'amber14-all.xml' which is robust, but if we strictly need 19SB we might need a specific file.
# However, the user prompt says: "ForceField('amber19sb.xml', 'implicit/obc2.xml')"
# So we will try that.

# Constants
KCAL_TO_KJ = 4.184
KJ_TO_KCAL = 1.0 / 4.184

def run_validation():
    print(colored("===========================================================", "cyan"))
    print(colored("   JAX MD vs OpenMM Validation: 1UBQ + ff19SB + OBC2", "cyan"))
    print(colored("===========================================================", "cyan"))

    # -------------------------------------------------------------------------
    # 1. Setup OpenMM System (Ground Truth)
    # -------------------------------------------------------------------------
    print(colored("\n[1] Setting up OpenMM System...", "yellow"))
    
    # Fix PDB (add missing hydrogens, etc if needed, but 1UAO might be clean)
    # Actually 1UAO is an NMR structure, usually has H.
    # But to be safe and consistent, let's use PDBFixer to ensure topology is standard.
    # Use Hydride to load and prep structure    # 1. Load Structure
    print(colored("\n[1] Loading Structure...", "yellow"))
    atom_array = parsing_biotite.load_structure_with_hydride(PDB_PATH, model=1)
    
    # Perturb coordinates to avoid singularities (linear angles from Hydride)
    # print(colored("    Perturbing coordinates by 0.01 A to avoid singularities...", "yellow"))
    # np.random.seed(0)
    # atom_array.coord += np.random.normal(0, 0.01, atom_array.coord.shape)


    
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

    # DEBUG: Print atoms of first residue to check Hydride output
    print(colored("DEBUG: Checking residue atoms from Hydride...", "yellow"))
    for res in topology.residues():
        if res.index == 0:
            print(f"Residue: {res.name} (Index 0)")
            for atom in res.atoms():
                print(f"  Atom: {atom.name}")
        if res.index == 41:
            print(f"Residue: {res.name} (Index 41)")
            for atom in res.atoms():
                print(f"  Atom: {atom.name}")
        if res.index == 75:
            print(f"Residue: {res.name} (Index 75)")
            for atom in res.atoms():
                print(f"  Atom: {atom.name}")
            break

    # Create ForceField
    try:
        # Load local protein.ff19SB.xml AND implicit/obc2.xml
        ff19sb_xml = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../openmmforcefields/openmmforcefields/ffxml/amber/protein.ff19SB.xml"))
        if not os.path.exists(ff19sb_xml):
             # Fallback to standard location or error
             print(colored(f"Warning: Could not find local protein.ff19SB.xml at {ff19sb_xml}. Trying standard...", "yellow"))
             omm_ff = app.ForceField('amber14-all.xml', 'implicit/obc2.xml')
        else:
             omm_ff = app.ForceField(ff19sb_xml, 'implicit/obc2.xml')
    except Exception as e:
        print(colored(f"Error loading ff19SB: {e}", "red"))
        sys.exit(1)

    # Create ForceField
    try:
        # Load local protein.ff19SB.xml AND implicit/obc2.xml
        ff19sb_xml = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../openmmforcefields/openmmforcefields/ffxml/amber/protein.ff19SB.xml"))
        if not os.path.exists(ff19sb_xml):
             raise FileNotFoundError(f"Could not find protein.ff19SB.xml at {ff19sb_xml}")
             
        omm_ff = app.ForceField(ff19sb_xml, 'implicit/obc2.xml')
    except Exception as e:
        print(colored(f"Error loading ff19SB: {e}", "red"))
        sys.exit(1)

    # Create System
    omm_system = omm_ff.createSystem(
        topology,
        nonbondedMethod=app.NoCutoff, # Infinite cutoff
        constraints=None, # No constraints for validation!
        rigidWater=False,
        removeCMMotion=False
    )
    
    # GBSAOBCForce setup moved to after parameterization to use JAX radii
    
    # Create Simulation context
    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    platform = openmm.Platform.getPlatformByName('Reference') # CPU Reference for precision
    simulation = app.Simulation(topology, omm_system, integrator, platform)
    simulation.context.setPositions(positions)
    
    # Get State
    state = simulation.context.getState(getEnergy=True, getForces=True, getPositions=True)
    omm_energy_total = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    omm_forces = state.getForces(asNumpy=True).value_in_unit(unit.kilocalories_per_mole / unit.angstrom)
    omm_positions = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    
    print(f"OpenMM Total Energy (Vacuum): {omm_energy_total:.4f} kcal/mol")
    
    # -------------------------------------------------------------------------
    # 2. Setup JAX MD System
    # -------------------------------------------------------------------------
    print(colored("\n[2] Setting up JAX MD System...", "yellow"))
    
    # Load Force Field
    ff = force_fields.load_force_field(FF_EQX_PATH)
    
    # Parse Topology from OpenMM (to ensure identical atoms/residues)
    # We need to convert OpenMM topology to what parameterize_system expects
    # residues: list of names, atom_names: list of names
    
    residues = []
    atom_names = []
    atom_counts = []
    
    for i, chain in enumerate(topology.chains()):
        for j, res in enumerate(chain.residues()):
            residues.append(res.name)
            count = 0
            for atom in res.atoms():
                name = atom.name
                # Fix N-term H -> H1 for Amber
                if i == 0 and j == 0 and name == "H":
                    name = "H1"
                atom_names.append(name)
                count += 1
            atom_counts.append(count)
    print("Creating JAX MD System...")
    # Parameterize JAX first to get Radii
    system_params = jax_md_bridge.parameterize_system(
        ff, residues, atom_names, atom_counts
    )
    
    # GBSAOBCForce setup removed - using implicit/obc2.xml
            
    # DEBUG: Print atoms of first residue
    print(colored(f"\n[DEBUG] First Residue ({residues[0]}) Atoms:", "cyan"))
    print(atom_names[:atom_counts[0]])
    
    # Parameterize
    system_params = jax_md_bridge.parameterize_system(
        ff, residues, atom_names, atom_counts
    )
    
    # Create Energy Function
    # Displacement function (no PBC for this test)
    displacement_fn, shift_fn = space.free()
    
    # DEBUG: Inspect System Params
    print(colored("\n[DEBUG] System Params Keys:", "yellow"))
    print(list(system_params.keys()))
    
    if "exclusion_mask" in system_params:
        em = system_params["exclusion_mask"]
        print(f"exclusion_mask shape: {em.shape}")
        print(f"exclusion_mask[0,1] (1-2): {em[0,1]}")
        print(f"exclusion_mask[0,2] (1-3): {em[0,2]}")
        print(f"exclusion_mask[0,3] (1-4): {em[0,3]}")
    else:
        print("exclusion_mask NOT FOUND!")

    if "scale_matrix_vdw" in system_params:
        sm = system_params["scale_matrix_vdw"]
        print(f"scale_matrix_vdw shape: {sm.shape}")
        # Check 1-2 (Bond) - e.g., atoms 0 and 1
        print(f"scale_matrix_vdw[0,1] (1-2): {sm[0,1]}")
        # Check 1-3 (Angle) - e.g., atoms 0 and 2
        print(f"scale_matrix_vdw[0,2] (1-3): {sm[0,2]}")
        # Check 1-4 (Dihedral) - e.g., atoms 0 and 4 (N-CA-C-N... wait, N-CA-C-O is 0-1-2-3)
        # 0(N)-1(CA)-2(C)-3(O) is 1-4? Yes.
        print(f"scale_matrix_vdw[0,3] (1-4): {sm[0,3]}")
        # Check 1-5 - e.g., atoms 0 and 5
        print(f"scale_matrix_vdw[0,5] (1-4?): {sm[0,5]}")
        # Check 1-6 - e.g., atoms 0 and 6 (N-CA-CB-CG-SD) -> 1-5 interaction
        print(f"scale_matrix_vdw[0,6] (1-5): {sm[0,6]}")
    else:
        print(colored("scale_matrix_vdw NOT FOUND!", "red"))
        
    if "gb_radii" in system_params:
        print(f"gb_radii present. Mean: {jnp.mean(system_params['gb_radii']):.4f}")
    else:
        print("gb_radii NOT FOUND!")

    # Define jax_positions early
    jax_positions = jnp.array(omm_positions)

    # DEBUG: Compare Bonds immediately
    jax_bonds = np.array(system_params['bonds'])
    print(f"DEBUG: JAX MD Generated {len(jax_bonds)} bonds.")
    
    # We need OpenMM bonds to compare. Extract them now.
    omm_bonds_check = []
    for force in omm_system.getForces():
        if isinstance(force, openmm.HarmonicBondForce):
            for i in range(force.getNumBonds()):
                p1, p2, length, k = force.getBondParameters(i)
                omm_bonds_check.append(tuple(sorted((p1, p2))))
    
    omm_bonds_set = set(omm_bonds_check)
    jax_bonds_set = set([tuple(sorted((int(b[0]), int(b[1])))) for b in jax_bonds])
    
    print(f"DEBUG: OpenMM has {len(omm_bonds_set)} bonds.")
    
    missing_in_jax = omm_bonds_set - jax_bonds_set
    extra_in_jax = jax_bonds_set - omm_bonds_set
    
    if missing_in_jax:
        print(colored(f"FAIL: JAX MD is missing {len(missing_in_jax)} bonds!", "red"))
        for b in list(missing_in_jax)[:10]:
            print(f"  Missing: {b} ({atom_names[b[0]]}-{atom_names[b[1]]})")
            
    if extra_in_jax:
        print(colored(f"FAIL: JAX MD has {len(extra_in_jax)} extra bonds!", "red"))
        for b in list(extra_in_jax)[:10]:
            print(f"  Extra: {b} ({atom_names[b[0]]}-{atom_names[b[1]]})")
            
    if not missing_in_jax and not extra_in_jax:
        print(colored("PASS: Topology (Bonds) matches.", "green"))
        
    # DEBUG: Compare Angles
    jax_angles = len(system_params['angles'])
    omm_angles = 0
    for force in omm_system.getForces():
        if isinstance(force, openmm.HarmonicAngleForce):
            omm_angles = force.getNumAngles()
    print(f"DEBUG: Angles: JAX={jax_angles}, OpenMM={omm_angles}")
    
    # DEBUG: Compare Torsions
    jax_dihedrals = len(system_params['dihedrals'])
    omm_dihedrals = 0
    for force in omm_system.getForces():
        if isinstance(force, openmm.PeriodicTorsionForce):
            omm_dihedrals = force.getNumTorsions()
    print(f"DEBUG: Torsions: JAX={jax_dihedrals}, OpenMM={omm_dihedrals}")

    # DEBUG: Print Initial Energy Components
    print(colored("\n[DEBUG] Initial JAX MD Energy Components (Generated Params):", "yellow"))
    energy_fn_initial = system.make_energy_fn(
        displacement_fn,
        system_params,
        implicit_solvent=True,
        dielectric_constant=1.0,
        solvent_dielectric=78.5,
        surface_tension=system.constants.SURFACE_TENSION,
        dielectric_offset=system.constants.DIELECTRIC_OFFSET
    )
    e_total_initial = energy_fn_initial(jax_positions)
    print(f"Total Energy: {e_total_initial:.4f} kcal/mol")
    
    # Breakdown
    e_bond_fn = system.bonded.make_bond_energy_fn(displacement_fn, system_params['bonds'], system_params['bond_params'])
    print(f"Bond Energy: {e_bond_fn(jax_positions):.4f}")
    
    e_angle_fn = system.bonded.make_angle_energy_fn(displacement_fn, system_params['angles'], system_params['angle_params'])
    print(f"Angle Energy: {e_angle_fn(jax_positions):.4f}")
    
    e_dih_fn = system.bonded.make_dihedral_energy_fn(displacement_fn, system_params['dihedrals'], system_params['dihedral_params'])
    e_imp_fn = system.bonded.make_dihedral_energy_fn(displacement_fn, system_params['impropers'], system_params['improper_params'])
    e_dih_val = e_dih_fn(jax_positions)
    e_imp_val = e_imp_fn(jax_positions)
    print(f"Torsion Energy (Proper): {e_dih_val:.4f}")
    print(f"Torsion Energy (Improper): {e_imp_val:.4f}")
    print(f"Torsion Energy (Total): {e_dih_val + e_imp_val:.4f}")

    
    # NonBonded
    # We can't easily isolate NB without reconstructing.
    # But Total - Bonded = NB
    e_bonded = e_bond_fn(jax_positions) + e_angle_fn(jax_positions) + e_dih_fn(jax_positions) + e_imp_fn(jax_positions)
    print(f"NonBonded+GBSA: {e_total_initial - e_bonded:.4f}")
    
    energy_fn = system.make_energy_fn(
        displacement_fn,
        system_params,
        implicit_solvent=True,
        dielectric_constant=1.0, # Solute dielectric
        solvent_dielectric=78.5, # Solvent dielectric (OBC2 default)
        surface_tension=0.005 # GBSA surface tension (kcal/mol/A^2) - check unit!
        # OpenMM OBC2 default surface tension is usually 28.39 J/mol/A^2 ~ 0.0067 kcal/mol/A^2?
        # Actually OpenMM `app.OBC2` uses `GBSAOBCForce`.
        # Default surface tension in OpenMM is 2.25936 kJ/mol/nm^2 = 0.54 kcal/mol/nm^2 = 0.0054 kcal/mol/A^2.
        # Let's verify this value.
    )
    
    # Compute Energy & Forces
    jax_positions = jnp.array(omm_positions)
    
    # DEBUG: Check for clashes
    dr = space.map_product(displacement_fn)(jax_positions, jax_positions)
    dist = space.distance(dr)
    # Mask self distance
    dist = dist + jnp.eye(dist.shape[0]) * 100.0
    # DEBUG: Check top 10 closest pairs
    # Flatten dist and get indices
    n = dist.shape[0]
    flat_dist = dist.flatten()
    sorted_indices = jnp.argsort(flat_dist)
    
    print("\n[DEBUG] Top 10 Closest Pairs:")
    for k in range(10):
        idx = sorted_indices[k]
        i = idx // n
        j = idx % n
        d = flat_dist[idx]
        
        # Check exclusion in CURRENT system_params (if available)
        # Note: system_params might not be fully populated yet (Stage 1 happens later)
        # But we want to know if they are bonded in OpenMM (which we haven't extracted yet at this point in the script)
        # Wait, this block is BEFORE Stage 1. We haven't extracted omm_bonds yet.
        # So we can't check omm_bonds here.
        # But we can print names.
        print(f"  {k+1}. {d:.4f} A: {i} ({atom_names[i]}) - {j} ({atom_names[j]})")

    
    jax_energy_total = energy_fn(jax_positions)
    jax_forces = -jax.grad(energy_fn)(jax_positions)
    
    print(f"JAX MD Total Energy: {jax_energy_total:.4f} kcal/mol")
    
    # -------------------------------------------------------------------------
    # 3. Stage 1: Parameter Validation
    # -------------------------------------------------------------------------
    print(colored("\n[Stage 1] Parameter Validation", "magenta"))
    
    # Extract OpenMM Parameters
    # We need to iterate through forces to find NonBondedForce
    omm_charges = []
    omm_sigmas = []
    omm_epsilons = []
    
    for force in omm_system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            for i in range(force.getNumParticles()):
                c, s, e = force.getParticleParameters(i)
                omm_charges.append(c.value_in_unit(unit.elementary_charge))
                omm_sigmas.append(s.value_in_unit(unit.angstrom))
                omm_epsilons.append(e.value_in_unit(unit.kilocalories_per_mole))
                
    omm_charges = np.array(omm_charges)
    omm_sigmas = np.array(omm_sigmas)
    omm_epsilons = np.array(omm_epsilons)
    
    jax_charges = np.array(system_params['charges'])
    jax_sigmas = np.array(system_params['sigmas'])
    jax_epsilons = np.array(system_params['epsilons'])
    
    # Compare
    diff_q = np.abs(omm_charges - jax_charges)
    diff_sig = np.abs(omm_sigmas - jax_sigmas)
    diff_eps = np.abs(omm_epsilons - jax_epsilons)
    
    max_diff_q = np.max(diff_q)
    max_diff_sig = np.max(diff_sig)
    max_diff_eps = np.max(diff_eps)
    
    print(f"Max Charge Diff: {max_diff_q:.6f} (Tol: 1e-4)")
    print(f"Max Sigma Diff:  {max_diff_sig:.6f} (Tol: 1e-4)")
    print(f"Max Epsilon Diff:{max_diff_eps:.6f} (Tol: 1e-4)")
    
    if max_diff_q < 1e-4 and max_diff_sig < 1e-4 and max_diff_eps < 1e-4:
        print(colored("PASS: Parameters match.", "green"))
    else:
        print(colored("FAIL: Parameters mismatch.", "red"))
        # Identify mismatch
        idx = np.argmax(diff_q)
        print(f"Max Charge Diff at atom {idx} ({atom_names[idx]}): {omm_charges[idx]} vs {jax_charges[idx]}")
        
    # -------------------------------------------------------------------------
    # 3b. GBSA Parameter Validation
    # -------------------------------------------------------------------------
    print(colored("\n[Stage 1b] GBSA Parameter Validation", "magenta"))
    
    omm_gb_radii = []
    omm_gb_scales = []
    
    # Find GBSA force
    gbsa_force = None
    for force in omm_system.getForces():
        if isinstance(force, openmm.GBSAOBCForce):
            gbsa_force = force
            break
        elif isinstance(force, openmm.CustomGBForce):
            gbsa_force = force
            # print(f"CustomGBForce Formula:\n{gbsa_force.getEnergyFunction()}")
            # print(dir(gbsa_force))
            break
            
    if gbsa_force:
        print(f"Found GBSA Force: {type(gbsa_force).__name__}")
        if isinstance(gbsa_force, openmm.GBSAOBCForce):
            for i in range(gbsa_force.getNumParticles()):
                q, r, s = gbsa_force.getParticleParameters(i)
                omm_gb_radii.append(r.value_in_unit(unit.angstrom))
                omm_gb_scales.append(s.value_in_unit(unit.dimensionless))
        elif isinstance(gbsa_force, openmm.CustomGBForce):
            # CustomGBForce parameters depend on definition
            # Usually [charge, radius, scale] or similar
            # We need to check parameter names
            print("CustomGBForce Parameters:")
            for i in range(gbsa_force.getNumPerParticleParameters()):
                print(f"  {i}: {gbsa_force.getPerParticleParameterName(i)}")
            
            # Print first 5 atoms params raw
            print("First 5 atoms raw params:")
            for i in range(5):
                 print(f"  Atom {i} ({atom_names[i]}): {gbsa_force.getParticleParameters(i)}")
            
            # Assuming 'radius' and 'scale' exist
            r_idx = -1
            s_idx = -1
            for i in range(gbsa_force.getNumPerParticleParameters()):
                name = gbsa_force.getPerParticleParameterName(i)
                if name == 'radius': r_idx = i
                elif name == 'scale': s_idx = i
                elif name == 'or': r_idx = i # Offset Radius
                elif name == 'sr': s_idx = i # Scaled Radius
            
            if r_idx >= 0 and s_idx >= 0:
                for i in range(gbsa_force.getNumParticles()):
                    params = gbsa_force.getParticleParameters(i)
                    # 'or' is usually radius - 0.09. We want radius.
                    # 'sr' is usually radius * scale. We want scale.
                    
                    val_r = params[r_idx] # nm
                    val_s = params[s_idx] # nm (scaled radius)
                    
                    # Convert to Angstroms
                    val_r_A = val_r * 10.0
                    val_s_A = val_s * 10.0
                    
                    # If name was 'or', it is (radius - offset). Offset is 0.09 A.
                    # So radius = val_r_A + 0.09
                    if gbsa_force.getPerParticleParameterName(r_idx) == 'or':
                        val_r_A += 0.09
                        
                    omm_gb_radii.append(val_r_A)
                    omm_gb_scales.append(val_s_A)
            else:
                print("Could not identify radius/scale in CustomGBForce")

        omm_gb_radii = np.array(omm_gb_radii)
        omm_gb_scales = np.array(omm_gb_scales)
        
        jax_gb_radii = np.array(system_params['gb_radii'])
        # JAX scaled_radii is r * scale.
        jax_scaled_radii = np.array(system_params['scaled_radii'])
        
        # Compare 'or' vs 'radius' (if we added 0.09 back)
        # Compare 'sr' vs 'scaled_radii'
        
        # If we extracted 'sr', it corresponds to jax_scaled_radii directly.
        # If we extracted 'scale', we need to multiply by radius.
        
        # Let's assume we extracted 'sr' (scaled radius)
        # So we compare omm_gb_scales (which is sr) with jax_scaled_radii
        
        if len(omm_gb_radii) > 0:
            diff_r = np.abs(omm_gb_radii - jax_gb_radii)
            # Check if we need to compare against scaled_radii directly
            if gbsa_force.getPerParticleParameterName(s_idx) == 'sr':
                 diff_s = np.abs(omm_gb_scales - jax_scaled_radii)
                 print("Comparing Scaled Radii (sr vs scaled_radii)")
            else:
                 # Compare scales
                 jax_scales = jax_scaled_radii / (jax_gb_radii + 1e-8)
                 diff_s = np.abs(omm_gb_scales - jax_scales)
                 print("Comparing Scale Factors")
            
            print(f"Max GBSA Radius Diff: {np.max(diff_r):.6f}")
            print(f"Max GBSA Scaled Radius Diff:  {np.max(diff_s):.6f}")
            
            if np.max(diff_r) > 1e-4:
                print("\nTop 10 Radius Mismatches:")
                indices = np.argsort(-diff_r)[:10]
                for idx in indices:
                    if diff_r[idx] > 1e-4:
                        print(f"  Atom {idx} ({atom_names[idx]} in {residues[atom_to_res[idx][0]]}): OpenMM={omm_gb_radii[idx]:.4f}, JAX={jax_gb_radii[idx]:.4f}, Diff={diff_r[idx]:.4f}")
                
            if np.max(diff_s) > 1e-4:
                print("\nTop 10 Scaled Radius Mismatches:")
                indices = np.argsort(-diff_s)[:10]
                for idx in indices:
                    if diff_s[idx] > 1e-4:
                        # If sr, print sr values
                        if gbsa_force.getPerParticleParameterName(s_idx) == 'sr':
                             print(f"  Atom {idx} ({atom_names[idx]}): OpenMM={omm_gb_scales[idx]:.4f}, JAX={jax_scaled_radii[idx]:.4f}, Diff={diff_s[idx]:.4f}")
                        else:
                             print(f"  Atom {idx} ({atom_names[idx]}): OpenMM={omm_gb_scales[idx]:.4f}, JAX={jax_scales[idx]:.4f}, Diff={diff_s[idx]:.4f}")
        else:
            print("No GBSA parameters extracted from OpenMM.")
    else:
        print("No GBSA Force found in OpenMM System.")
    # -------------------------------------------------------------------------
    # 4. Stage 2: Energy Component Validation
    # -------------------------------------------------------------------------
    print(colored("\n[Stage 2] Energy Component Validation", "magenta"))
    
    # Get OpenMM Components
    # We need to group forces by type
    omm_components = {
        'Bond': 0.0, 'Angle': 0.0, 'Torsion': 0.0, 'NonBonded': 0.0, 'GBSA': 0.0, 'CMMotion': 0.0
    }
    
    # Note: OpenMM puts VDW + Direct Coulomb in NonBondedForce.
    # GBSAOBCForce contains Solvation (Polar + NonPolar).
    
    for force in omm_system.getForces():
        group = force.getForceGroup()
        # We need to compute energy for each group separately to get components
        # Or just use the force type if unique
        
        # Actually, to get components, we should set force groups and re-evaluate
        pass

    # Re-evaluate by group
    # Assign groups:
    # 0: HarmonicBondForce
    # 1: HarmonicAngleForce
    # 2: PeriodicTorsionForce (Proper + Improper)
    # 3: NonbondedForce (VDW + Elec)
    # 4: GBSAOBCForce
    # 5: CMAPTorsionForce
    
    group_map = {}
    for i, force in enumerate(omm_system.getForces()):
        force.setForceGroup(i)
        name = force.__class__.__name__
        group_map[i] = name
        
    # Update context
    simulation.context.reinitialize(preserveState=True)
    
    print(f"{'Component':<20} | {'OpenMM (kcal/mol)':<20} | {'JAX (kcal/mol)':<20} | {'Diff':<10} | {'Status'}")
    print("-" * 85)
    
    total_omm = 0.0
    total_jax = 0.0
    
    # Compute JAX Components individually
    # We need to access the individual energy functions inside `make_energy_fn` or reconstruct them
    # For this script, we can just call the sub-functions if we can access them.
    # But `make_energy_fn` returns a closure.
    # We can reconstruct them here for validation.
    
    # JAX Bond
    e_bond_fn = system.bonded.make_bond_energy_fn(displacement_fn, system_params['bonds'], system_params['bond_params'])
    e_bond_jax = e_bond_fn(jax_positions)
    
    # JAX Angle
    e_angle_fn = system.bonded.make_angle_energy_fn(displacement_fn, system_params['angles'], system_params['angle_params'])
    e_angle_jax = e_angle_fn(jax_positions)
    
    # JAX Torsion (Dihedral + Improper)
    e_dih_fn = system.bonded.make_dihedral_energy_fn(displacement_fn, system_params['dihedrals'], system_params['dihedral_params'])
    e_imp_fn = system.bonded.make_dihedral_energy_fn(displacement_fn, system_params['impropers'], system_params['improper_params'])
    e_dih_val = e_dih_fn(jax_positions)
    e_imp_val = e_imp_fn(jax_positions)
    e_torsion_jax = e_dih_val + e_imp_val


    
    # JAX CMAP
    e_cmap_jax = 0.0
    if 'cmap_torsions' in system_params and len(system_params['cmap_torsions']) > 0:
        # We need to manually call the internal logic or expose it.
        # Let's rely on total energy diff for now or try to replicate.
        # We can import cmap module.
        from prxteinmpnn.physics import cmap
        # Re-implement the logic briefly
        cmap_torsions = system_params["cmap_torsions"]
        cmap_indices = system_params["cmap_indices"]
        cmap_grids = system_params["cmap_energy_grids"]
        phi_indices = cmap_torsions[:, 0:4]
        psi_indices = cmap_torsions[:, 1:5]
        phi = system.compute_dihedral_angles(jax_positions, phi_indices, displacement_fn)
        psi = system.compute_dihedral_angles(jax_positions, psi_indices, displacement_fn)
        # Recalculate JAX energy with injected params
    # Note: We need to use the NEW compute_cmap_energy signature which takes coeffs
    cmap_indices = system_params["cmap_indices"]
    cmap_coeffs = system_params["cmap_coeffs"]
    
    # Use standard JAX MD displacement for free space
    displacement_fn, shift_fn = space.free()
    
    e_bond_jax = bonded.make_bond_energy_fn(displacement_fn, system_params["bonds"], system_params["bond_params"])(jax_positions)
    e_angle_jax = bonded.make_angle_energy_fn(displacement_fn, system_params["angles"], system_params["angle_params"])(jax_positions)
    e_torsion_jax = bonded.make_dihedral_energy_fn(displacement_fn, system_params["dihedrals"], system_params["dihedral_params"])(jax_positions)
    # Improper?
    e_improper_jax = bonded.make_dihedral_energy_fn(displacement_fn, system_params["impropers"], system_params["improper_params"])(jax_positions)
    
    # CMAP
    # We need to compute phi/psi for CMAP terms
    # The system.py implementation does this.
    # But here we want component-wise.
    # Let's use the helper from system.py if possible, or replicate.
    # system.py: compute_cmap_term(r)
    # We can't easily import the internal function.
    # But we can replicate the logic:
    cmap_torsions = system_params["cmap_torsions"]
    phi_indices = cmap_torsions[:, 0:4]
    psi_indices = cmap_torsions[:, 1:5]
    
    def compute_dihedrals(r, idx):
        # ... (need displacement fn)
        # Let's just use the energy function from system.py?
        # No, we want components.
        # We can use bonded.make_dihedral_energy_fn logic but just for angles?
        # Or just use features.protein_backbone_torsions?
        # No, that gives residues.
        # We need specific torsions.
        pass
        
    # Actually, let's just use the total energy function to get components if possible?
    # No, JAX MD energy fn returns total.
    
    # Let's copy compute_dihedral_angles from system.py or import it?
    from prxteinmpnn.physics.system import compute_dihedral_angles
    
    phi = compute_dihedral_angles(jax_positions, phi_indices, displacement_fn)
    psi = compute_dihedral_angles(jax_positions, psi_indices, displacement_fn)
    
    e_cmap_jax_normal = cmap.compute_cmap_energy(phi, psi, cmap_indices, cmap_coeffs)
    e_cmap_jax_swap = cmap.compute_cmap_energy(psi, phi, cmap_indices, cmap_coeffs)
    
    print(f"DEBUG: CMAP (phi, psi): {e_cmap_jax_normal:.4f}")
    print(f"DEBUG: CMAP (psi, phi): {e_cmap_jax_swap:.4f}")
    print(f"DEBUG: CMAP (phi, psi) * 0.5: {e_cmap_jax_normal * 0.5:.4f}")
    print(f"DEBUG: CMAP (psi, phi) * 0.5: {e_cmap_jax_swap * 0.5:.4f}")
    
    # Use the best match for the table (assuming 0.5 is needed and maybe swap)
    # For now, let's stick to normal * 0.5 as baseline, or whatever matches best.
    # We will update e_cmap_jax based on findings.
    e_cmap_jax = e_cmap_jax_swap # Use (psi, phi) ordering which matches OpenMM

    # JAX NonBonded (VDW + Elec + GBSA)
    # This is harder to split exactly as OpenMM splits it.
    # OpenMM NonbondedForce = VDW + Coulomb (Direct)
    # OpenMM GBSAOBCForce = GBSA (Polar) + SASA (NonPolar)
    
    # Let's compute JAX terms
    # We need to replicate the logic from system.py
    # ... (simplified replication)
    
    # Helper to get OpenMM energy by class
    def get_omm_energy(cls_name):
        e = 0.0
        for i, name in group_map.items():
            if name == cls_name:
                e += simulation.context.getState(getEnergy=True, groups={i}).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        return e
        
    omm_bond = get_omm_energy('HarmonicBondForce')
    omm_angle = get_omm_energy('HarmonicAngleForce')
    omm_torsion = get_omm_energy('PeriodicTorsionForce')
    omm_cmap = get_omm_energy('CMAPTorsionForce')
    omm_nonbonded = get_omm_energy('NonbondedForce') # VDW + Direct Elec
    omm_gbsa = get_omm_energy('GBSAOBCForce') + get_omm_energy('CustomGBForce') # Polar + NonPolar
    
    # Compare Bond
    print_row("Bond", omm_bond, e_bond_jax)
    
    # Compare Angle
    print_row("Angle", omm_angle, e_angle_jax)
    
    # Compare Torsion
    print_row("Torsion", omm_torsion, e_torsion_jax)
    
    # Compare CMAP
    print_row("CMAP", omm_cmap, e_cmap_jax)
    
    # NonBonded is tricky.
    # JAX `total_energy` includes everything.
    # Let's subtract bonded terms from total to get "Rest"
    jax_rest = jax_energy_total - (e_bond_jax + e_angle_jax + e_torsion_jax + e_cmap_jax)
    omm_rest = omm_nonbonded + omm_gbsa
    
    # Explicitly calculate LJ and Coulomb
    from prxteinmpnn.physics import electrostatics, vdw
    
    # LJ
    sigma_i = jnp.array(system_params['sigmas'])
    epsilon_i = jnp.array(system_params['epsilons'])
    
    # Compute pairwise
    dr = space.map_product(displacement_fn)(jax_positions, jax_positions)
    dist = space.distance(dr)
    
    # LJ Energy
    sigma_ij = 0.5 * (sigma_i[:, None] + sigma_i[None, :])
    epsilon_ij = jnp.sqrt(epsilon_i[:, None] * epsilon_i[None, :])
    e_lj_mat = energy.lennard_jones(dist, sigma_ij, epsilon_ij)
    
    # Apply scaling
    if "scale_matrix_vdw" in system_params:
        e_lj_mat = e_lj_mat * system_params["scale_matrix_vdw"]
    
    e_lj_total = 0.5 * jnp.sum(e_lj_mat)
    print(f"DEBUG: Explicit JAX LJ Energy: {e_lj_total:.4f} kcal/mol")
    
    # Coulomb Energy
    charges = jnp.array(system_params['charges'])
    q_ij = charges[:, None] * charges[None, :]
    dist_safe = dist + 1e-6
    e_coul_mat = 332.0637 * q_ij / dist_safe
    
    # Apply scaling
    if "scale_matrix_elec" in system_params:
        e_coul_mat = e_coul_mat * system_params["scale_matrix_elec"]
        
    e_coul_total = 0.5 * jnp.sum(e_coul_mat)
    print(f"DEBUG: Explicit JAX Coulomb Energy: {e_coul_total:.4f} kcal/mol")
    
    print_row("NonBonded+GBSA", omm_rest, jax_rest)
    
    print("-" * 85)
    print(f"DEBUG: OpenMM NonBonded: {omm_nonbonded:.4f}")
    print(f"DEBUG: OpenMM GBSA: {omm_gbsa:.4f}")
    print("-" * 85)
    print_row("TOTAL", omm_energy_total, jax_energy_total)
    
    # -------------------------------------------------------------------------
    # 5. Stage 3: Force Validation
    # -------------------------------------------------------------------------
    print(colored("\n[Stage 3] Force Validation", "magenta"))
    
    forces_omm = np.array(omm_forces)
    forces_jax = np.array(jax_forces)
    
    # Relative RMSE
    # Rel RMSE = sqrt( mean( (F_omm - F_jax)^2 ) ) / sqrt( mean( F_omm^2 ) )
    
    diff_sq = np.sum((forces_omm - forces_jax)**2, axis=1)
    omm_sq = np.sum(forces_omm**2, axis=1)
    
    rmse = np.sqrt(np.mean(diff_sq))
    norm_omm = np.sqrt(np.mean(omm_sq))
    rel_rmse = rmse / (norm_omm + 1e-8)
    
    print(f"Force RMSE: {rmse:.6f} kcal/mol/A")
    print(f"Force Rel-RMSE: {rel_rmse:.6f}")
    
    # Max Error
    max_err_idx = np.argmax(np.sqrt(diff_sq))
    max_err = np.sqrt(diff_sq[max_err_idx])
    print(f"Max Force Error: {max_err:.6f} at atom {max_err_idx} ({atom_names[max_err_idx]})")
    
    if rel_rmse < 1e-3: # Strict tolerance
         print(colored("PASS: Forces match.", "green"))
    else:
         print(colored("FAIL: Forces mismatch.", "red"))

    # -------------------------------------------------------------------------
    # 6. Detailed Analysis (Torsion & Self-Energy)
    # -------------------------------------------------------------------------
    print(colored("\n[Analysis] Detailed Discrepancy Analysis", "cyan"))
    
    # Check for NaN in components
    print(colored("\n[DEBUG] Checking Force Components for NaN:", "yellow"))
    
    def check_grad(name, fn):
        g = jax.grad(fn)(jax_positions)
        if jnp.any(jnp.isnan(g)):
            print(f"  {name}: NaN detected!")
        else:
            print(f"  {name}: OK")
            
    check_grad("Bond", e_bond_fn)
    check_grad("Angle", e_angle_fn)
    check_grad("Torsion", e_dih_fn)
    check_grad("Improper", e_imp_fn)
    
    def cmap_fn(r):
        return system.compute_cmap_term(r) # Uses system.py logic
    # We need to make sure system.py is imported and available
    # It is imported as 'system'
    # But compute_cmap_term is inside make_energy_fn closure in system.py?
    # No, I saw it as a helper function inside make_energy_fn.
    # Calculate JAX NonBonded explicitly for comparison
    # We need to use the same functions as system.py
    # Re-create them here for debug
    
    # LJ
    # We need to handle 1-4 scaling correctly.
    # system.py uses scale_matrix_vdw
    
    # Let's just use the values we printed in DEBUG if possible?
    # No, we need to sum them for the table.
    
    # We can use the component functions we defined in debug_forces.py style?
    # Or just use the fact that we have e_total and other components.
    # e_nb_gbsa_jax = e_total - e_bond - e_angle - e_torsion - e_cmap
    # This is safer.
    
    jax_nb_gbsa = jax_energy_total - e_bond_jax - e_angle_jax - e_torsion_jax - e_cmap_jax
    
    print(f"DEBUG: Inferred JAX NB+GBSA: {jax_nb_gbsa:.4f}")
    
    # NonBonded+GBSA
    omm_nb_gbsa = omm_nonbonded + omm_gbsa
    print(f"{'NonBonded+GBSA':<20} | {omm_nb_gbsa:20.4f} | {jax_nb_gbsa:20.4f} | {jax_nb_gbsa - omm_nb_gbsa:10.4f} | {'PASS' if abs(jax_nb_gbsa - omm_nb_gbsa) < 100.0 else 'FAIL'}")
    
    # A. Self-Energy Analysis
    from prxteinmpnn.physics import generalized_born, constants
    
    # Compute Born Radii
    # We need to use the same parameters
    radii = system_params['gb_radii']
    charges = system_params['charges']
    
    # Note: compute_born_radii expects (N,) arrays
    born_radii = generalized_born.compute_born_radii(
        jax_positions, radii, dielectric_offset=0.09, scaled_radii=system_params['scaled_radii']
    )
    
    # Compute Self Energy
    # E_self = -0.5 * 332 * (1/ein - 1/eout) * sum(q^2/B)
    tau = (1.0/1.0) - (1.0/78.5)
    prefactor = -0.5 * constants.COULOMB_CONSTANT * tau
    self_energy_term = prefactor * jnp.sum((charges**2) / born_radii)
    
    print(f"Calculated JAX Self-Energy: {self_energy_term:.4f} kcal/mol")
    print(f"JAX GBSA (Total): {jax_rest:.4f} kcal/mol")
    print(f"JAX GBSA (Corrected for Self): {jax_rest - self_energy_term:.4f} kcal/mol")
    print(f"OpenMM GBSA + NonBonded: {omm_rest:.4f} kcal/mol")
    print(f"Diff (Corrected): {omm_rest - (jax_rest - self_energy_term):.4f} kcal/mol")
    
    # B. Torsion Analysis
    # We want to compute energy per torsion
    # JAX Torsions
    dihedrals = system_params['dihedrals']
    dihedral_params = system_params['dihedral_params']
    
    def single_torsion_energy(r, indices, params):
        # indices: (4,)
        # params: (3,)
        r_i = r[indices[0]]
        r_j = r[indices[1]]
        r_k = r[indices[2]]
        r_l = r[indices[3]]
        
        b0 = displacement_fn(r_i, r_j)
        b1 = displacement_fn(r_k, r_j)
        b2 = displacement_fn(r_l, r_k)
        
        b1_norm = jnp.linalg.norm(b1) + 1e-8
        b1_unit = b1 / b1_norm
        
        v = b0 - jnp.dot(b0, b1_unit) * b1_unit
        w = b2 - jnp.dot(b2, b1_unit) * b1_unit
        
        x = jnp.dot(v, w)
        y = jnp.dot(jnp.cross(b1_unit, v), w)
        
        phi = jnp.arctan2(y, x)
        
        per, phase, k = params
        return k * (1.0 + jnp.cos(per * phi - phase)), phi
        
    # Vectorize
    calc_torsions = jax.vmap(single_torsion_energy, in_axes=(None, 0, 0))
    jax_torsion_energies, jax_phis = calc_torsions(jax_positions, dihedrals, dihedral_params)
    
    # OpenMM Torsions
    # We need to extract them one by one? Or assume order is preserved.
    # Order IS preserved because we loaded them sequentially.
    
    print("\nTop 10 Torsion Mismatches:")
    print(f"{'Idx':<5} | {'Atoms':<20} | {'Params (n, phi0, k)':<25} | {'JAX E':<10} | {'JAX Phi':<10}")
    
    # We can't easily get OpenMM per-torsion energy without creating separate forces or system.
    # But we can calculate what OpenMM SHOULD give using the formula and JAX phi.
    # If JAX phi matches OpenMM phi, then energy should match.
    # If JAX E differs from OpenMM Total Torsion, then either:
    # 1. Summation is wrong
    # 2. Some terms are missing/extra
    # 3. Phi is different
    
    # Let's print the sum of JAX torsions
    print(f"Sum of JAX Torsion Energies: {jnp.sum(jax_torsion_energies):.4f}")
    print(f"OpenMM Total Torsion: {omm_torsion:.4f}")
    
    # Let's check for outliers in Energy magnitude
    # Maybe some torsions are huge?
    # Build atom to residue map
    atom_to_res = {}
    current_atom = 0
    for i, res_name in enumerate(residues):
        count = atom_counts[i]
        for j in range(count):
            atom_to_res[current_atom + j] = (i, res_name)
        current_atom += count

    # Build OpenMM Torsion Map
    omm_torsion_map = {}
    for force in omm_system.getForces():
        if isinstance(force, openmm.PeriodicTorsionForce):
            for i in range(force.getNumTorsions()):
                p1, p2, p3, p4, per, phase, k = force.getTorsionParameters(i)
                # Store by sorted tuple or just forward/backward
                # Torsions are directional (1-2-3-4), but 4-3-2-1 is equivalent?
                # Usually defined 1-2-3-4.
                key = (p1, p2, p3, p4)
                # OpenMM might have multiple terms for same atoms (multiplicity).
                # We should store list of terms.
                if key not in omm_torsion_map:
                    omm_torsion_map[key] = []
                omm_torsion_map[key].append((per, phase.value_in_unit(unit.radian), k.value_in_unit(unit.kilocalories_per_mole)))
                
                # Also store reverse?
                key_rev = (p4, p3, p2, p1)
                if key_rev not in omm_torsion_map:
                    omm_torsion_map[key_rev] = []
                omm_torsion_map[key_rev].append((per, phase.value_in_unit(unit.radian), k.value_in_unit(unit.kilocalories_per_mole)))

    # Group JAX Torsions by Atoms
    jax_torsion_grouped = {}
    for i in range(len(dihedrals)):
        d = dihedrals[i]
        key = tuple(sorted((int(d[0]), int(d[1]), int(d[2]), int(d[3])))) # Sort to match OpenMM map key strategy if needed? 
        # Actually OpenMM map used (p1, p2, p3, p4) and (p4, p3, p2, p1).
        # Let's use canonical key: min(forward, backward)
        forward = (int(d[0]), int(d[1]), int(d[2]), int(d[3]))
        backward = (int(d[3]), int(d[2]), int(d[1]), int(d[0]))
        key = min(forward, backward)
        
        if key not in jax_torsion_grouped:
            jax_torsion_grouped[key] = {'energy': 0.0, 'params': [], 'phi': jax_phis[i]}
        
        jax_torsion_grouped[key]['energy'] += jax_torsion_energies[i]
        jax_torsion_grouped[key]['params'].append(dihedral_params[i])

    print(f"\n{'Atoms':<20} | {'JAX E (Sum)':<12} | {'OpenMM Params':<40} | {'Residue'}")
    
    # Sort grouped by energy
    grouped_items = sorted(jax_torsion_grouped.items(), key=lambda x: -abs(x[1]['energy']))
    
    for k in range(15):
        if k >= len(grouped_items): break
        atoms_tuple, data = grouped_items[k]
        e = data['energy']
        
        # Check OpenMM params
        omm_params_str = "MISSING"
        # Try forward
        if atoms_tuple in omm_torsion_map:
             terms = omm_torsion_map[atoms_tuple]
             omm_params_str = " | ".join([f"{t[0]},{t[1]:.2f},{t[2]:.2f}" for t in terms])
        
        res_info = f"{atom_to_res[atoms_tuple[0]][1]}{atom_to_res[atoms_tuple[0]][0]+1}"
        atoms_str = f"{atoms_tuple[0]}-{atoms_tuple[1]}-{atoms_tuple[2]}-{atoms_tuple[3]}"
        
        print(f"{atoms_str:<20} | {e:<12.4f} | {omm_params_str:<40} | {res_info}")

    # Debug Angles for NaN
    print(colored("\n[DEBUG] Checking Angles for NaN/Singularities:", "yellow"))
    angles = system_params['angles']
    angle_params = system_params['angle_params']
    
    # Compute all angles
    def compute_angle_val(r, idx):
        r_i = r[idx[0]]
        r_j = r[idx[1]]
        r_k = r[idx[2]]
        
        v_ji = r_i - r_j
        v_jk = r_k - r_j
        
        # Norms
        n_ji = jnp.linalg.norm(v_ji)
        n_jk = jnp.linalg.norm(v_jk)
        
        # Cosine
        dot = jnp.dot(v_ji, v_jk)
        cos_theta = dot / (n_ji * n_jk + 1e-8)
        # Clip? JAX MD usually clips in angle function?
        # bonded.py uses space.angle which uses clip.
        return jnp.arccos(jnp.clip(cos_theta, -0.999999, 0.999999))

    calc_angles = jax.vmap(compute_angle_val, in_axes=(None, 0))
    angle_vals = calc_angles(jax_positions, angles)
    
    # Check for NaN or extreme values
    nan_indices = jnp.where(jnp.isnan(angle_vals))[0]
    if len(nan_indices) > 0:
        print(f"Found {len(nan_indices)} NaN angles!")
        for idx in nan_indices[:5]:
             print(f"  Idx {idx}: {angles[idx]} -> NaN")
    else:
        print("No NaN angles found directly.")
        
    # Check for angles close to 0 or pi
    extreme_indices = jnp.where((angle_vals < 0.01) | (angle_vals > 3.13))[0]
    if len(extreme_indices) > 0:
        print(f"Found {len(extreme_indices)} extreme angles (near 0 or pi):")
        for idx in extreme_indices[:10]:
            deg = angle_vals[idx] * 180.0 / jnp.pi
            print(f"  Idx {idx}: {angles[idx]} -> {deg:.4f} deg ({atom_names[angles[idx][0]]}-{atom_names[angles[idx][1]]}-{atom_names[angles[idx][2]]})")
            
    # Check Force NaN specifically
    # We already know Angle force has NaN.
    # Let's find which one contributes NaN to force.
    # We can use jax.grad per angle? Slow but effective.
    
    def single_angle_energy(r, idx, param):
        return system.bonded.make_angle_energy_fn(displacement_fn, idx[None, :], param[None, :])(r)
        
    # Scan for NaN grad
    print("Scanning angles for NaN gradient...")
    # This might be slow for 2000 angles.
    # Let's check only extreme ones first.
    
    for idx in extreme_indices:
        g = jax.grad(single_angle_energy)(jax_positions, angles[idx], angle_params[idx])
        if jnp.any(jnp.isnan(g)):
            print(colored(f"  NaN Gradient at Angle {idx}: {angles[idx]}", "red"))

    # C. SASA Debugging
    print(colored("\n[Analysis] SASA Debugging", "cyan"))
    from prxteinmpnn.physics import sasa
    
    radii = system_params['gb_radii']
    gamma = 0.005
    
    sasa_energy = sasa.compute_sasa_energy_approx(jax_positions, radii, gamma=gamma, offset=0.0)
    print(f"Explicit SASA Energy Check: {sasa_energy:.4f} kcal/mol")
    
    if sasa_energy == 0.0:
        print("FAIL: SASA Energy is 0.0. Checking inputs...")
        print(f"Radii Stats: Min={jnp.min(radii)}, Max={jnp.max(radii)}")
        print(f"Positions Stats: Min={jnp.min(jax_positions)}, Max={jnp.max(jax_positions)}")


def print_row(name, val_omm, val_jax, tol=0.05):
    diff = val_omm - val_jax
    status = "PASS" if abs(diff) < tol else "FAIL"
    color = "green" if status == "PASS" else "red"
    print(colored(f"{name:<20} | {val_omm:20.4f} | {val_jax:20.4f} | {diff:10.4f} | {status}", color))

if __name__ == "__main__":
    run_validation()
