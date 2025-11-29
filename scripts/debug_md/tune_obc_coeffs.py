
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import jax
import jax.numpy as jnp
import numpy as np
from prxteinmpnn.physics import generalized_born

def get_openmm_born_radii(r_distance, radius1=1.5, radius2=1.5):
    system = mm.System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    
    force = mm.GBSAOBCForce()
    force.addParticle(0.0, radius1, 1.0) # charge, radius, scale
    force.addParticle(0.0, radius2, 1.0)
    system.addForce(force)
    
    integrator = mm.VerletIntegrator(0.001)
    context = mm.Context(system, integrator)
    
    positions = [
        mm.Vec3(0, 0, 0),
        mm.Vec3(r_distance, 0, 0)
    ]
    context.setPositions(positions)
    
    # Extract Born Radii?
    # OpenMM doesn't expose Born Radii directly in Python easily?
    # We can get Energy.
    # E = -0.5 * (1/1 - 1/80) * (q1*q2/f + q1^2/2B1 + q2^2/2B2).
    # If q1=1, q2=0. E = -0.5 * tau * (1/2B1).
    # So B1 = -0.5 * tau / (2 * E).
    
    # Let's set q1=1, q2=0.
    force.setParticleParameters(0, 1.0, radius1, 1.0)
    force.setParticleParameters(1, 0.0, radius2, 1.0)
    force.updateParametersInContext(context)
    
    state = context.getState(getEnergy=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    
    # Constants
    # COULOMB = 332.063711
    # tau = (1/1 - 1/78.5) (OpenMM default water?)
    # Let's check OpenMM defaults.
    # soluteDielectric=1.0, solventDielectric=78.5.
    
    tau = (1.0/1.0 - 1.0/78.5)
    prefactor = -0.5 * 332.063711 * tau # kcal/mol * A
    
    # E = prefactor * (1/B1). (Self energy of particle 0).
    # B1 = prefactor / E.
    
    # Note: OpenMM GBSA includes non-polar term?
    # GBSAOBCForce includes surface area term?
    # "The GBSAOBCForce class implements the implicit solvation model... It also includes a nonpolar term..."
    # surfaceAreaEnergy = surfaceTension * SASA.
    # We need to disable nonpolar term.
    # setSurfaceTension(0.0).
    
    return energy

def run_tuning():
    # Setup OpenMM system to get pure GBSA energy (no SA)
    system = mm.System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    force = mm.GBSAOBCForce()
    # force.setSurfaceTension(0.0) # Disable SA
    # print(dir(force))
    # exit()
    # Try to find it dynamically
    if hasattr(force, 'setSurfaceAreaEnergy'):
        force.setSurfaceAreaEnergy(0.0)
    elif hasattr(force, 'setNonPolarPrefactor'):
        force.setNonPolarPrefactor(0.0)
    else:
        print("WARNING: Could not disable SA term. Energy will include SASA.")
    force.addParticle(1.0, 0.15, 1.0) # q=1, r=1.5 A (0.15 nm), scale=1.0
    force.addParticle(0.0, 0.15, 1.0) # q=0
    system.addForce(force)
    
    integrator = mm.VerletIntegrator(0.001)
    context = mm.Context(system, integrator)
    
    distances = np.linspace(0.3, 2.0, 20) # nm. 3A to 20A.
    
    print("Distance (A) | OpenMM E | OpenMM B1 | JAX B1 (Current)")
    
    for d in distances:
        d_angstrom = d * 10.0
        positions = [mm.Vec3(0,0,0), mm.Vec3(d,0,0)]
        context.setPositions(positions)
        state = context.getState(getEnergy=True)
        e_omm = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        
        # Calculate B1 from E
        # E = -0.5 * 332.06 * (1-1/78.5) * (1/B1)
        prefactor = -0.5 * 332.063711 * (1.0 - 1.0/78.5)
        b1_omm = prefactor / e_omm if e_omm != 0 else 0
        
        # JAX Calculation
        # We need to call compute_born_radii with current coefficients
        # We can mock the function or just use the one in generalized_born.py
        # But we need to pass arrays.
        
        pos_jax = jnp.array([[0.0, 0.0, 0.0], [d_angstrom, 0.0, 0.0]])
        radii_jax = jnp.array([1.5, 1.5])
        
        # We need to expose compute_born_radii or copy it here.
        # Let's import it.
        b_jax = generalized_born.compute_born_radii(pos_jax, radii_jax, dielectric_offset=0.009 * 10) # 0.09 A
        b1_jax = b_jax[0]
        
        print(f"{d_angstrom:10.4f} | {e_omm:8.4f} | {b1_omm:8.4f} | {b1_jax:8.4f}")

if __name__ == "__main__":
    run_tuning()
