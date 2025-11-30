import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
from termcolor import colored
import pandas as pd

# OpenMM Imports
try:
    import openmm
    import openmm.app as app
    import openmm.unit as unit
    from pdbfixer import PDBFixer
except ImportError:
    print(colored("Error: OpenMM or PDBFixer not found. Please install them.", "red"))
    sys.exit(1)

# PrxteinMPNN Imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from prxteinmpnn.physics import force_fields, jax_md_bridge, system
from prxteinmpnn.utils import residue_constants

# Configuration
FF_EQX_PATH = "src/prxteinmpnn/physics/force_fields/eqx/protein19SB.eqx"
OPENMM_XMLS = ['amber14-all.xml', 'implicit/obc2.xml'] # Fallback to 14SB if 19SB not available in OpenMM

def create_tri_alanine_pdb(filename="tri_alanine.pdb"):
    """Creates a simple Tri-Alanine PDB file for testing."""
    # We can use PDBFixer to create it from sequence
    # We can use PDBFixer to create it from sequence
    # fixer = PDBFixer(filename=None)
    # PDBFixer doesn't support creating from sequence directly easily without a PDB file?
    # Actually it does via `fixer.source` but easier to just write a minimal PDB or use a known one.
    # Let's write a minimal PDB string for ALA-ALA-ALA (extended)
    # Or better, use Biotite or just rely on PDBFixer to build it if we can.
    # Let's use a simple hardcoded PDB string for 3 ALAs.
    
    pdb_content = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.362   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.408   2.434   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       2.009  -0.763   1.216  1.00  0.00           C
ATOM      6  N   ALA A   2       3.336   1.362   0.000  1.00  0.00           N
ATOM      7  CA  ALA A   2       4.032   2.649   0.000  1.00  0.00           C
ATOM      8  C   ALA A   2       5.542   2.649   0.000  1.00  0.00           C
ATOM      9  O   ALA A   2       6.143   3.721   0.000  1.00  0.00           O
ATOM     10  CB  ALA A   2       3.481   3.412   1.216  1.00  0.00           C
ATOM     11  N   ALA A   3       6.143   1.577   0.000  1.00  0.00           N
ATOM     12  CA  ALA A   3       7.601   1.577   0.000  1.00  0.00           C
ATOM     13  C   ALA A   3       8.152   2.939   0.000  1.00  0.00           C
ATOM     14  O   ALA A   3       9.366   3.134   0.000  1.00  0.00           O
ATOM     15  CB  ALA A   3       8.152   0.814   1.216  1.00  0.00           C
ATOM     16  OXT ALA A   3       7.300   3.900   0.000  1.00  0.00           O
TER
"""
    with open(filename, "w") as f:
        f.write(pdb_content)
    return filename

def run_validation():
    print(colored("===========================================================", "cyan"))
    print(colored("   Terminal Residue Validation: Tri-Alanine + ff19SB", "cyan"))
    print(colored("===========================================================", "cyan"))

    # 1. Setup Test System
    pdb_file = create_tri_alanine_pdb()
    
    # 2. OpenMM Setup (Ground Truth)
    print(colored("\n[1] Setting up OpenMM System...", "yellow"))
    fixer = PDBFixer(filename=pdb_file)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    
    omm_ff = app.ForceField(*OPENMM_XMLS)
    omm_system = omm_ff.createSystem(
        fixer.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None
    )
    
    # Extract OpenMM Parameters
    omm_params = {}
    for force in omm_system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            charges = []
            sigmas = []
            epsilons = []
            for i in range(force.getNumParticles()):
                c, s, e = force.getParticleParameters(i)
                charges.append(c.value_in_unit(unit.elementary_charge))
                sigmas.append(s.value_in_unit(unit.angstrom))
                epsilons.append(e.value_in_unit(unit.kilocalories_per_mole))
            omm_params['charges'] = np.array(charges)
            omm_params['sigmas'] = np.array(sigmas)
            omm_params['epsilons'] = np.array(epsilons)

    # 3. JAX MD Setup
    print(colored("\n[2] Setting up JAX MD System...", "yellow"))
    ff = force_fields.load_force_field(FF_EQX_PATH)
    
    # Extract topology info
    residues = []
    atom_names = []
    atom_counts = []
    
    for chain in fixer.topology.chains():
        for res in chain.residues():
            residues.append(res.name)
            count = 0
            for atom in res.atoms():
                name = atom.name
                # Fix N-term H -> H1 for Amber match
                if len(residues) == 1 and name == "H": # Only one residue so far means it's the first
                     name = "H1"
                atom_names.append(name)
                count += 1
            atom_counts.append(count)
            
    print(f"Residues: {residues}")
    print(f"N-term Atoms: {atom_names[:atom_counts[0]]}")
    print(f"C-term Atoms: {atom_names[-atom_counts[-1]:]}")

    jax_params = jax_md_bridge.parameterize_system(
        ff, residues, atom_names, atom_counts
    )
    
    # 4. Compare Parameters
    print(colored("\n[3] Comparing Parameters...", "magenta"))
    
    # A. Charges
    diff_q = np.abs(omm_params['charges'] - jax_params['charges'])
    max_diff_q = np.max(diff_q)
    print(f"Max Charge Diff: {max_diff_q:.6f}")
    
    if max_diff_q > 1e-4:
        print(colored("FAIL: Charge Mismatch", "red"))
        for i in range(len(atom_names)):
            if diff_q[i] > 1e-4:
                print(f"  Atom {i} ({atom_names[i]}): OpenMM={omm_params['charges'][i]:.4f}, JAX={jax_params['charges'][i]:.4f}")
    else:
        print(colored("PASS: Charges Match", "green"))
        
    # B. VDW
    diff_sig = np.abs(omm_params['sigmas'] - jax_params['sigmas'])
    diff_eps = np.abs(omm_params['epsilons'] - jax_params['epsilons'])
    
    if np.max(diff_sig) > 1e-4 or np.max(diff_eps) > 1e-4:
        print(colored("FAIL: VDW Mismatch", "red"))
    else:
        print(colored("PASS: VDW Match", "green"))

    # Cleanup
    if os.path.exists(pdb_file):
        os.remove(pdb_file)

if __name__ == "__main__":
    run_validation()
