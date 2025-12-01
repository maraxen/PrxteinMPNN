
import pytest
import numpy as np
import jax
import jax.numpy as jnp
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import tempfile
import os
import biotite.structure.io.pdb as pdb
import biotite.structure.io as struc_io
import biotite.database.rcsb as rcsb

from prxteinmpnn.physics import force_fields, jax_md_bridge, system
from prxteinmpnn.io.parsing import biotite as parsing_biotite
from prxteinmpnn.utils import residue_constants

def test_debug_gbsa_offset():
    print("Debugging GBSA Offset...")
    
    # Enable x64
    jax.config.update("jax_enable_x64", True)

    # Fetch 1L2Y
    pdb_path = rcsb.fetch("1L2Y", "pdb", tempfile.gettempdir())
    atom_array = struc_io.load_structure(pdb_path, model=1)
    
    if atom_array.chain_id is not None:
         atom_array = atom_array[atom_array.chain_id == atom_array.chain_id[0]]
         
    # Save to temp PDB
    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w+") as tmp:
        pdb_file = pdb.PDBFile()
        pdb_file.set_structure(atom_array)
        pdb_file.write(tmp)
        tmp.flush()
        
        # Load with Biotite (native IO)
        atom_array = parsing_biotite.load_structure_with_hydride(tmp.name, model=1, add_hydrogens=True)
        
        # Parameterize JAX MD
        ff_path = "src/prxteinmpnn/physics/force_fields/eqx/protein19SB.eqx"
        if not os.path.exists(ff_path):
            ff_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src/prxteinmpnn/physics/force_fields/eqx/protein19SB.eqx"))
            
        ff = force_fields.load_force_field(ff_path)
        params, coords = parsing_biotite.biotite_to_jax_md_system(atom_array, ff)
        
        # Compute JAX Energy
        from jax_md import space
        displacement_fn, _ = space.free()
        
        # Create Energy Fn
        energy_fn = system.make_energy_fn(
            displacement_fn=displacement_fn,
            system_params=params,
            implicit_solvent=True,
            solvent_dielectric=78.5,
            solute_dielectric=1.0,
            dielectric_offset=0.009, 
            surface_tension=0.0 
        )
        
        jax_energy = float(energy_fn(coords))
        print(f"JAX MD Total Energy: {jax_energy:.4f} kcal/mol")
        
        # Compute OpenMM Energy
        with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w+") as tmp_omm:
            pdb_file = pdb.PDBFile()
            pdb_file.set_structure(atom_array)
            pdb_file.write(tmp_omm)
            tmp_omm.flush()
            tmp_omm.seek(0)
            
            pdb_file_omm = app.PDBFile(tmp_omm.name)
            topology = pdb_file_omm.topology
            positions = pdb_file_omm.positions
            
            forcefield = app.ForceField('amber14-all.xml', 'implicit/obc2.xml')
            
            try:
                system_omm = forcefield.createSystem(
                    topology,
                    nonbondedMethod=app.CutoffNonPeriodic,
                    nonbondedCutoff=2.0*unit.nanometer,
                    constraints=None,
                )
            except Exception as e:
                print(f"OpenMM createSystem failed: {e}. Retrying with addHydrogens...")
                modeller = app.Modeller(topology, positions)
                modeller.addHydrogens(forcefield)
                system_omm = forcefield.createSystem(
                    modeller.topology,
                    nonbondedMethod=app.CutoffNonPeriodic,
                    nonbondedCutoff=2.0*unit.nanometer,
                    constraints=None,
                )
                positions = modeller.positions
            
            # Assign Force Groups
            for i, force in enumerate(system_omm.getForces()):
                force.setForceGroup(i)
            
            integrator = mm.VerletIntegrator(1.0*unit.femtosecond)
            simulation = app.Simulation(topology if 'modeller' not in locals() else modeller.topology, system_omm, integrator)
            simulation.context.setPositions(positions)
            
            state = simulation.context.getState(getEnergy=True)
            omm_energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
            print(f"OpenMM Total Energy: {omm_energy:.4f} kcal/mol")
            
            diff = jax_energy - omm_energy
            print(f"Difference: {diff:.4f} kcal/mol")
            
            # Breakdown OpenMM Forces
            print("\nOpenMM Breakdown:")
            for i, force in enumerate(system_omm.getForces()):
                group = i
                state = simulation.context.getState(getEnergy=True, groups={group})
                e = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
                print(f"  {force.__class__.__name__}: {e:.4f} kcal/mol")

if __name__ == "__main__":
    test_debug_gbsa_offset()
