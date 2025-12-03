import os
import sys
import numpy as np
import openmm
import openmm.app as app
import openmm.unit as unit
from termcolor import colored

# Configuration
PDB_PATH = "data/pdb/1UBQ.pdb"

def debug_gbsa_scaling():
    print(colored("===========================================================", "cyan"))
    print(colored("   Debugging OpenMM GBSA Scaling (1-2, 1-3, 1-4)", "cyan"))
    print(colored("===========================================================", "cyan"))

    # 1. Load Structure
    pdb = app.PDBFile(PDB_PATH)
    topology = pdb.topology
    positions = pdb.positions

    # 2. Create System with GBSAOBCForce
    # Use standard Amber force field
    ff = app.ForceField('amber14-all.xml', 'implicit/obc2.xml')
    
    # Create system
    system = ff.createSystem(topology, nonbondedMethod=app.NoCutoff)
    
    # Find GBSAOBCForce
    gb_force = None
    for force in system.getForces():
        if isinstance(force, openmm.GBSAOBCForce):
            gb_force = force
            break
            
    if gb_force is None:
        print("Error: GBSAOBCForce not found!")
        return

    print(f"GBSAOBCForce found. SoluteDielectric={gb_force.getSoluteDielectric()}, SolventDielectric={gb_force.getSolventDielectric()}")

    # 3. Calculate Energy
    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    context = openmm.Context(system, integrator)
    context.setPositions(positions)
    
    state = context.getState(getEnergy=True)
    e_total = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    print(f"Total Energy (All Pairs): {e_total:.4f} kcal/mol")
    
    # 4. Test Exclusions
    # We want to know if 1-4 pairs are included in GBSA.
    # We can't easily toggle them in GBSAOBCForce directly via API (it doesn't have setException).
    # But GBSAOBCForce says "The GBSA interaction is calculated between all pairs of particles".
    
    # Let's try to manually calculate GBSA for a subset of atoms or pairs?
    # Or create a system with NO bonds/angles/torsions (so no exclusions) and compare?
    
    print("\nCreating System with NO exclusions (Topology with no bonds)...")
    # Hack: Create a new topology with same atoms but no bonds
    new_top = app.Topology()
    new_chain = new_top.addChain()
    new_res = new_top.addResidue("UNK", new_chain)
    for atom in topology.atoms():
        new_top.addAtom(atom.name, atom.element, new_res)
        
    # We can't use ForceField.createSystem with this empty topology because it won't match templates.
    # Instead, let's use the existing system and REMOVE all exceptions from NonBondedForce.
    # But GBSAOBCForce doesn't use NonBondedForce exceptions?
    
    # Let's verify if GBSAOBCForce ignores NonBondedForce exceptions.
    # NonBondedForce has exceptions for 1-2, 1-3, 1-4.
    
    nb_force = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)][0]
    num_exceptions = nb_force.getNumExceptions()
    print(f"NonBondedForce has {num_exceptions} exceptions (1-2, 1-3, 1-4).")
    
    # If GBSA uses these exceptions, then removing them should change the energy.
    # If GBSA ignores them (includes all), then removing them should NOT change GBSA energy.
    # (It will change Coulomb/LJ energy).
    
    # We need to isolate GBSA energy.
    # We can use force groups.
    gb_force.setForceGroup(1)
    nb_force.setForceGroup(2)
    # Set others to 0
    for f in system.getForces():
        if f != gb_force and f != nb_force:
            f.setForceGroup(0)
            
    context = openmm.Context(system, integrator)
    context.setPositions(positions)
    
    e_gb_orig = context.getState(getEnergy=True, groups={1}).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    print(f"GBSA Energy (Original): {e_gb_orig:.4f} kcal/mol")
    
    # Now remove all exceptions from NonBondedForce
    # Note: modifying system requires re-creating context
    for i in range(num_exceptions):
        # setException(index, p1, p2, chargeProd, sigma, epsilon)
        # We can't remove, but we can turn them into normal interactions?
        # Or we can just clear them? No clear method.
        # But we can create a new system with createSystem and remove exceptions?
        pass
        
    # Actually, GBSAOBCForce doesn't seem to have a method to set exclusions.
    # It calculates Born Radii based on geometry.
    # And Energy sum over pairs.
    # Documentation says "between all pairs".
    
    # Let's trust the documentation and the fact that GBSA is a continuum model.
    # Usually 1-2/1-3/1-4 are included.
    
    # But let's double check if JAX MD's ~1800 kcal/mol difference matches the 1-4 contribution.
    # We can calculate the 1-4 GBSA contribution in JAX MD.
    
    print("\nTo verify, we will modify JAX MD to include 1-4s and see if it matches this value.")

if __name__ == "__main__":
    debug_gbsa_scaling()
