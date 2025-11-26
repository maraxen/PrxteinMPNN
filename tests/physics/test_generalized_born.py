"""Tests for Generalized Born implicit solvent model."""

import jax
import jax.numpy as jnp
import pytest
from jax import test_util

from prxteinmpnn.physics import constants, generalized_born

class TestGeneralizedBorn:
    def test_born_ion_analytical(self):
        """Test GB energy against analytical Born Ion solvation energy.
        
        Delta G = -0.5 * (1/eps_in - 1/eps_out) * q^2 / R
        """
        # Single ion
        q = 1.0
        r_vdw = 2.0
        
        # In our implementation, we pass radii.
        # The Born radius of a single ion in vacuum is its intrinsic radius.
        # compute_born_radii should return r_vdw - offset?
        # No, for a single ion, I_i = 0 (no neighbors).
        # So B^-1 = 1/rho_i - 0.
        # So B = rho_i = r_vdw - offset.
        
        offset = 0.09
        expected_born_radius = r_vdw - offset
        
        positions = jnp.array([[0.0, 0.0, 0.0]])
        charges = jnp.array([q])
        radii = jnp.array([r_vdw])
        
        eps_in = 1.0
        eps_out = 78.5
        
        # Analytical Energy
        # E = -0.5 * C * (1/eps_in - 1/eps_out) * q^2 / born_radius
        tau = (1.0 / eps_in) - (1.0 / eps_out)
        expected_energy = -0.5 * constants.COULOMB_CONSTANT * tau * (q**2) / expected_born_radius
        
        # Computed Energy
        computed_energy = generalized_born.compute_gb_energy(
            positions, charges, radii,
            solvent_dielectric=eps_out,
            solute_dielectric=eps_in,
            dielectric_offset=offset
        )
        
        assert jnp.allclose(computed_energy, expected_energy, rtol=1e-5)

    def test_finite_difference_gradients(self):
        """Check that the energy function is differentiable and gradients match finite difference."""
        
        # 2-particle system
        # Avoid r = r_i + r_j (kink in max function)
        # r_i = 1.5, r_j = 1.5. Sum = 3.0.
        # Use distance 3.5
        positions = jnp.array([
            [0.0, 0.0, 0.0],
            [3.5, 0.0, 0.0]
        ])
        charges = jnp.array([1.0, -1.0])
        radii = jnp.array([1.5, 1.5])
        
        def energy_fn(pos):
            return generalized_born.compute_gb_energy(
                pos, charges, radii
            )
        
        # Analytical Gradient
        grad_fn = jax.grad(energy_fn)
        grads = grad_fn(positions)
        
        # Numerical Gradient
        eps = 1e-4
        num_grads = jnp.zeros_like(grads)
        
        for i in range(positions.shape[0]):
            for j in range(3):
                # Perturb +
                pos_p = positions.at[i, j].add(eps)
                e_p = energy_fn(pos_p)
                
                # Perturb -
                pos_m = positions.at[i, j].add(-eps)
                e_m = energy_fn(pos_m)
                
                num_grads = num_grads.at[i, j].set((e_p - e_m) / (2 * eps))
        
        # Check match
        # We use a slightly loose tolerance for FD due to float32 precision
        # Relative error was ~0.3%, so 1% tolerance is safe.
        assert jnp.allclose(grads, num_grads, rtol=1e-2, atol=1e-2)

    def test_neighbor_list_consistency(self):
        """Check that neighbor list implementation matches dense implementation."""
        
        # 3-particle system
        positions = jnp.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [5.0, 0.0, 0.0]
        ])
        charges = jnp.array([0.5, -0.5, 0.5])
        radii = jnp.array([1.5, 1.5, 1.5])
        
        # Dense Energy
        e_dense = generalized_born.compute_gb_energy(positions, charges, radii)
        
        # Neighbor List Energy
        # Construct a fake neighbor list
        # 0 sees 1, 1 sees 0, 1 sees 2, 2 sees 1.
        # 0 and 2 are far (5.0), let's say cutoff is 4.0.
        
        # Neighbor indices (padded to N=3)
        # 0: [1, 3, 3]
        # 1: [0, 2, 3]
        # 2: [1, 3, 3]
        # 3 is padding (N=3)
        
        neighbor_idx = jnp.array([
            [1, 3, 3],
            [0, 2, 3],
            [1, 3, 3]
        ])
        
        e_neighbor = generalized_born.compute_gb_energy_neighbor_list(
            positions, charges, radii, neighbor_idx
        )
        
        # They should be close, but not identical because 0-2 interaction is missing in neighbor list.
        # But GB is long range.
        # However, for this test, let's include all neighbors to verify implementation correctness.
        
        neighbor_idx_full = jnp.array([
            [1, 2, 3],
            [0, 2, 3], # 1 sees 0 and 2
            [0, 1, 3]  # 2 sees 0 and 1
        ])
        
        e_neighbor_full = generalized_born.compute_gb_energy_neighbor_list(
            positions, charges, radii, neighbor_idx_full
        )
        
        assert jnp.allclose(e_dense, e_neighbor_full, rtol=1e-5)

    def test_energy_conservation_nve(self):
        """Check energy conservation in NVE simulation."""
        from jax_md import simulate, space
        
        # 3-particle system
        positions = jnp.array([
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 4.0, 0.0]
        ])
        charges = jnp.array([0.5, -0.5, 0.5])
        radii = jnp.array([1.5, 1.5, 1.5])
        masses = jnp.array([10.0, 10.0, 10.0])
        
        displacement_fn, shift_fn = space.free()
        
        def energy_fn(pos):
            return generalized_born.compute_gb_energy(
                pos, charges, radii
            )
        
        dt = 1e-3
        steps = 100
        
        # init_fn, apply_fn = simulate.velocity_verlet(energy_fn, shift_fn, dt)
        # jax_md.simulate.velocity_verlet returns (init_fn, apply_fn)
        # But apply_fn takes (state, **kwargs).
        
        # Wait, jax_md.simulate.velocity_verlet signature:
        # velocity_verlet(force_fn, shift_fn, dt, state=None) -> (init_fn, apply_fn)
        # Actually, it usually takes energy_or_force_fn.
        
        # Let's check how it's used in simulate.py or standard usage.
        # simulate.nvt_langevin was used in simulate.py.
        
        # Standard usage:
        # init_fn, apply_fn = simulate.velocity_verlet(energy_fn, shift_fn, dt)
        # state = init_fn(key, R) or similar.
        
        # The error says "missing 1 required positional argument: 'state'".
        # This implies I might be calling it wrong or the version requires state to be passed during creation?
        # Or maybe I am calling apply_fn wrong?
        
        # Ah, looking at the error: "TypeError: velocity_verlet() missing 1 required positional argument: 'state'"
        # This suggests I am calling `simulate.velocity_verlet(...)` and it expects `state`?
        # No, `velocity_verlet` is a generator.
        
        # Maybe I imported wrong?
        # from jax_md import simulate
        
        # Let's try passing the initial state to the generator if required, or check docs.
        # But usually it's (energy_fn, shift_fn, dt).
        
        # Wait, maybe I need to pass `mass` to init_fn?
        
        # Let's look at the error again.
        # "FAILED ... - TypeError: velocity_verlet() missing 1 required positional argument: 'state'"
        # This error comes from the line `init_fn, apply_fn = simulate.velocity_verlet(energy_fn, shift_fn, dt)`?
        # Or from `apply_fn(s)`?
        
        # If it's from the generator call, then `velocity_verlet` might not be the generator but the step function itself?
        # No, `simulate.velocity_verlet` is usually the setup.
        
        # Let's assume I need to use `simulate.nve` which might be the high level wrapper?
        # Or `simulate.velocity_verlet` is correct.
        
        # Let's try to fix by checking if I need to provide more args.
        # In JAX MD, `velocity_verlet` often requires `mass` if it's not 1.0?
        
        # Let's try a simpler integrator `simulate.nve` if available, or just fix `velocity_verlet`.
        # Actually, let's look at `simulate.py` to see how `nvt_langevin` is called.
        # `init_fn, apply_fn = simulate.nvt_langevin(energy_fn, shift_fn, dt, kT, gamma)`
        
        # So `velocity_verlet` should be similar.
        # Maybe I am missing `mass` in the setup?
        
        # Let's try:
        # init_fn, apply_fn = simulate.velocity_verlet(energy_fn, shift_fn, dt)
        
        # If that fails, maybe I should check if `velocity_verlet` is actually `nvt_nose_hoover` or similar?
        # No, `velocity_verlet` is standard.
        
        # Let's try to debug by printing or just using a standard NVE loop manually if needed.
        # But `jax_md` should have it.
        
        # Maybe the error is in `apply_fn(s)`?
        # "lambda i, s: apply_fn(s)"
        
        # Wait, the error "missing 1 required positional argument: 'state'" usually happens when you call a method that expects self/state but don't provide it?
        
        # Let's try to use `simulate.nve` if it exists.
        # Or just implement Verlet manually since it's simple.
        
        # Manual Velocity Verlet:
        # R(t+dt) = R(t) + v(t)*dt + 0.5*a(t)*dt^2
        # v(t+0.5dt) = v(t) + 0.5*a(t)*dt
        # a(t+dt) = F(R(t+dt)) / m
        # v(t+dt) = v(t+0.5dt) + 0.5*a(t+dt)*dt
        
        # Let's do manual NVE to avoid library issues and ensure control.
        
        def manual_step(i, state):
            R, V, F = state
            M = masses[:, None]
            A = F / M
            
            # Half step velocity
            V_half = V + 0.5 * A * dt
            
            # Full step position
            R_new = shift_fn(R, V_half * dt)
            
            # New Force
            E_new = energy_fn(R_new) # We need grad
            F_new = -jax.grad(energy_fn)(R_new)
            
            A_new = F_new / M
            
            # Full step velocity
            V_new = V_half + 0.5 * A_new * dt
            
            return (R_new, V_new, F_new)

        # Initial force
        F_init = -jax.grad(energy_fn)(positions)
        
        # Initialize with some velocity
        key = jax.random.PRNGKey(0)
        v_init = jax.random.normal(key, positions.shape) * 0.1
        
        state = (positions, v_init, F_init)
        
        # Run
        final_state = jax.lax.fori_loop(0, steps, manual_step, state)
        
        R_final, V_final, _ = final_state
        
        E_initial = energy_fn(positions) + 0.5 * jnp.sum(masses[:, None] * v_init**2)
        E_final = energy_fn(R_final) + 0.5 * jnp.sum(masses[:, None] * V_final**2)
        
        drift = jnp.abs(E_final - E_initial) / jnp.abs(E_initial)
        print(f"\nEnergy Drift: {drift}")
        assert drift < 1e-4
