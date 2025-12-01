import pytest
import numpy as np
import jax
import jax.numpy as jnp
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import tempfile
import os
from biotite.structure.io import pdb

from prxteinmpnn.physics import force_fields, jax_md_bridge, system
from prxteinmpnn.io.parsing import biotite as parsing_biotite
from prxteinmpnn.utils import residue_constants

# Simple ALA-ALA dipeptide (heavy atoms)
PDB_ALA_ALA = """ATOM      1  N   ALA A   1      -0.525   1.364   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       1.526   0.000   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       2.153  -1.062   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1      -0.541  -0.759  -1.212  1.00  0.00           C
ATOM      6  N   ALA A   2       2.103   1.192   0.000  1.00  0.00           N
ATOM      7  CA  ALA A   2       3.562   1.192   0.000  1.00  0.00           C
ATOM      8  C   ALA A   2       4.088   2.616   0.000  1.00  0.00           C
ATOM      9  O   ALA A   2       5.289   2.846   0.000  1.00  0.00           O
ATOM     10  CB  ALA A   2       4.103   0.433   1.212  1.00  0.00           C
"""

@pytest.mark.slow
def test_jax_openmm_energy_equivalency():
    """Test that JAX MD and OpenMM energies match for a simple system loaded via native IO."""
    
    # 1. Download 1UAO and truncate to ensure valid geometry
    import biotite.database.rcsb as rcsb
    import biotite.structure.io as struc_io
    
    # Enable x64
    jax.config.update("jax_enable_x64", True)

    # Fetch to temp dir
    pdb_path = rcsb.fetch("1L2Y", "pdb", tempfile.gettempdir())
    atom_array = struc_io.load_structure(pdb_path, model=1)
    
    # Use full structure (small enough)
    if atom_array.chain_id is not None:
         atom_array = atom_array[atom_array.chain_id == atom_array.chain_id[0]]
         
    # Save to temp PDB
    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w+") as tmp:
        pdb_file = pdb.PDBFile()
        pdb_file.set_structure(atom_array)
        pdb_file.write(tmp)
        tmp.flush()
        
        # 2. Load with Biotite (native IO) - adds hydrogens using Hydride
        atom_array = parsing_biotite.load_structure_with_hydride(tmp.name, model=1, add_hydrogens=True)
        
        # 3. Parameterize JAX MD
        # Assuming FF is available at this path relative to repo root
        ff_path = "src/prxteinmpnn/physics/force_fields/eqx/protein19SB.eqx"
        if not os.path.exists(ff_path):
            ff_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src/prxteinmpnn/physics/force_fields/eqx/protein19SB.eqx"))
            
        ff = force_fields.load_force_field(ff_path)
        params, coords = parsing_biotite.biotite_to_jax_md_system(atom_array, ff)
        
        # 4. Compute JAX Energy
        from jax_md import space
        displacement_fn, _ = space.free()
        
        energy_fn = system.make_energy_fn(
            displacement_fn=displacement_fn,
            system_params=params,
            implicit_solvent=True,
            solvent_dielectric=78.5,
            solute_dielectric=1.0,
            dielectric_offset=0.009, # Match OpenMM default/standard
            surface_tension=0.0 # Match OpenMM
        )
        
        jax_energy = float(energy_fn(coords))
        
            # 5. Compute OpenMM Energy
        # Convert atom_array to OpenMM Topology/Positions
        with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w+") as tmp_omm:
            pdb_file = pdb.PDBFile()
            pdb_file.set_structure(atom_array)
            pdb_file.write(tmp_omm)
            tmp_omm.flush()
            tmp_omm.seek(0)
            
            pdb_file_omm = app.PDBFile(tmp_omm.name)
            topology = pdb_file_omm.topology
            positions = pdb_file_omm.positions
            
            # Use local ff19SB XML to match JAX MD's protein19SB.eqx
            ff19sb_xml = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../openmmforcefields/openmmforcefields/ffxml/amber/protein.ff19SB.xml"))
            if not os.path.exists(ff19sb_xml):
                 # Fallback for CI environment if path differs, or error out
                 raise FileNotFoundError(f"Could not find protein.ff19SB.xml at {ff19sb_xml}")

            forcefield = app.ForceField(ff19sb_xml, 'implicit/obc2.xml')
            
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
                # Update positions if addHydrogens changed them (added atoms)
                positions = modeller.positions
            
            # Context
            integrator = mm.VerletIntegrator(1.0*unit.femtosecond)
            simulation = app.Simulation(topology if 'modeller' not in locals() else modeller.topology, system_omm, integrator)
            simulation.context.setPositions(positions)
            
            state = simulation.context.getState(getEnergy=True)
            omm_energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)

            # 6. Compare
            diff = abs(jax_energy - omm_energy)
            
            print(f"\nJAX Energy: {jax_energy:.4f} kcal/mol")
            print(f"OpenMM Energy: {omm_energy:.4f} kcal/mol")
            print(f"Difference: {diff:.4f} kcal/mol")
            
            # Strict check: With identical Force Fields (ff19SB) and GBSA model (OBC2),
            # energies should match very closely (< 1.0 kcal/mol).
            assert diff < 1.0, f"Energy difference too large: {diff} kcal/mol (Expected < 1.0 with identical FF)"
            assert np.isfinite(jax_energy)
            assert np.isfinite(omm_energy)
