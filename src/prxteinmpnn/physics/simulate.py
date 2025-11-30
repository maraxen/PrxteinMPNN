"""Simulation runner for minimization and thermalization."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax import random, jit
from jax_md import minimize, simulate, space, util, quantity
import dataclasses
from functools import partial

from prxteinmpnn.physics import system
from prxteinmpnn.physics.constants import BOLTZMANN_KCAL
from prxteinmpnn.physics.jax_md_bridge import SystemParams

Array = util.Array


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class NVTLangevinState:
  position: Array
  momentum: Array
  force: Array
  mass: Array
  rng: Array

  def set(self, **kwargs):
      return dataclasses.replace(self, **kwargs)

  def tree_flatten(self):
    return ((self.position, self.momentum, self.force, self.mass, self.rng), None)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(*children)

def rattle_langevin(
    energy_or_force_fn: Callable[..., Array],
    shift_fn: Callable[..., Array],
    dt: float,
    kT: float,
    gamma: float = 0.1,
    mass: float | Array = 1.0,
    constraints: tuple[Array, Array] | None = None
):
  """Langevin dynamics with RATTLE constraints."""
  force_fn = quantity.canonicalize_force(energy_or_force_fn)
  dt_2 = dt / 2

  def init_fn(key, R, mass=mass, **kwargs):
    _kT = kwargs.pop('kT', kT)
    key, split = random.split(key)
    force = force_fn(R, **kwargs)
    
    # Handle mass
    mass = jnp.array(mass, dtype=R.dtype)
    if mass.ndim == 0:
        mass = jnp.ones_like(R[..., 0]) * mass
    elif mass.ndim == 1 and mass.shape[0] == R.shape[0]:
        mass = mass[:, None] # (N, 1) for broadcasting
        
    state = NVTLangevinState(R, None, force, mass, key)
    return simulate.initialize_momenta(state, split, _kT)

  def apply_fn(state, **kwargs):
    _dt = kwargs.pop('dt', dt)
    _kT = kwargs.pop('kT', kT)
    dt_2 = _dt / 2
    
    # 1. B (Velocity Update 1)
    # v = v + 0.5 * dt * F/m
    momentum = state.momentum + 0.5 * _dt * state.force
    
    # 2. A (Position Update 1)
    # r = r + 0.5 * dt * v/m
    velocity = momentum / state.mass
    position = shift_fn(state.position, 0.5 * _dt * velocity)
    
    # 3. O (Stochastic Update)
    # v = c1 * v + c2 * noise
    # c1 = exp(-gamma * dt)
    # c2 = sqrt(1 - c1^2) * sqrt(m * kT)
    c1 = jnp.exp(-gamma * _dt)
    c2 = jnp.sqrt(1 - c1**2)
    
    key, split = random.split(state.rng)
    noise = random.normal(split, state.momentum.shape)
    momentum = c1 * momentum + c2 * jnp.sqrt(state.mass * _kT) * noise
    
    # 4. A (Position Update 2)
    # r = r + 0.5 * dt * v/m
    velocity = momentum / state.mass
    position = shift_fn(position, 0.5 * _dt * velocity)
    
    # --- SHAKE (Position Constraint) ---
    if constraints is not None:
        pairs, lengths = constraints
        # Project position to satisfy bond lengths
        # Iterative SHAKE
        # We need to update position AND momentum (to be consistent r(t+dt) - r(t))?
        # Standard RATTLE updates r(t+dt) and v(t+dt/2).
        # Here we are at step 4 (end of position update).
        # We should project position.
        # And correct momentum?
        # In BAOAB with RATTLE, usually:
        # R_new = R_old + ...
        # Project R_new.
        # V_new = (R_new - R_old) / dt? No, that's Verlet.
        # For Langevin, we just project R.
        # And usually we project V at the end (RATTLE).
        
        position = project_positions(position, pairs, lengths, state.mass, shift_fn)
    
    # 5. Force Update
    force = force_fn(position, **kwargs)
    
    # 6. B (Velocity Update 2)
    # v = v + 0.5 * dt * F/m
    momentum = momentum + 0.5 * _dt * force
    
    # --- RATTLE (Velocity Constraint) ---
    if constraints is not None:
        pairs, lengths = constraints
        # Project momentum to be orthogonal to bonds
        momentum = project_momenta(momentum, position, pairs, state.mass, shift_fn)

    return NVTLangevinState(position, momentum, force, state.mass, key)

  return init_fn, apply_fn

def project_positions(R, pairs, lengths, mass, shift_fn, tol=1e-5, max_iter=10):
    """Iterative SHAKE projection for positions."""
    # pairs: (M, 2) indices
    # lengths: (M,) target lengths
    # mass: (N, 1)
    
    # Pre-compute inverse mass for pairs
    # inv_mass = 1.0 / mass
    # w1 = inv_mass[pairs[:, 0]]
    # w2 = inv_mass[pairs[:, 1]]
    # w_sum = w1 + w2
    
    def body_fn(i, R_curr):
        # Compute deviations
        r1 = R_curr[pairs[:, 0]]
        r2 = R_curr[pairs[:, 1]]
        d_vec = space.map_product(shift_fn)(r1, r2) # r1 - r2
        # Actually shift_fn(r1, r2) -> r1 - r2? Check docs.
        # Usually displacement(R1, R2). shift_fn(R, dR) -> R+dR.
        # We need displacement_fn.
        # But we only have shift_fn passed to integrator.
        # Usually integrator gets displacement_fn? No, shift_fn.
        # Wait, force_fn usually needs displacement.
        # But here we need it for constraints.
        # If we don't have displacement_fn, we can't compute distances in PBC.
        # But for 'free' space, d = r1 - r2.
        # We'll assume free space for now (as in run_simulation).
        # Or we can pass displacement_fn to rattle_langevin?
        # Standard nvt_langevin only takes shift_fn.
        # But we need displacement for constraints.
        # We should pass displacement_fn.
        # For now, assume simple difference (no PBC).
        
        d_vec = r1 - r2
        d2 = jnp.sum(d_vec**2, axis=-1)
        diff = d2 - lengths**2
        
        # Correction scalar
        # g = diff / (2 * d_vec . d_vec_old * (1/m1 + 1/m2)) ?
        # Approximation: d_vec . d_vec_old ~ d^2 ~ lengths^2
        # g = diff / (2 * lengths^2 * (1/m1 + 1/m2))?
        # Better: g = (lengths^2 - d2) / (2 * (r1-r2) . (r1-r2) * ...)
        # Standard SHAKE: delta = (d^2 - L^2) / (4 * d . r_ij) ?
        # Let's use: correction = diff / (2 * (1/m1 + 1/m2) * d . d) ?
        # Actually, d . d is d2.
        
        inv_m1 = 1.0 / mass[pairs[:, 0], 0]
        inv_m2 = 1.0 / mass[pairs[:, 1], 0]
        w_sum = inv_m1 + inv_m2
        
        # g = diff / (2 * w_sum * d_vec . d_vec) ??
        # No, linearization: |r + dr|^2 = L^2
        # |r|^2 + 2 r.dr = L^2
        # 2 r.dr = L^2 - |r|^2 = -diff
        # dr = g * r
        # 2 r . (g * r) = 2 g |r|^2 = -diff
        # g = -diff / (2 |r|^2)
        # But we have masses.
        # dr1 = g * w1 * r12
        # dr2 = -g * w2 * r12
        # dr12 = g * (w1+w2) * r12
        # |r12 + dr12|^2 = L^2
        # |r12|^2 + 2 r12 . dr12 = L^2
        # d2 + 2 r12 . (g * w_sum * r12) = L^2
        # d2 + 2 g w_sum d2 = L^2
        # 2 g w_sum d2 = L^2 - d2 = -diff
        # g = -diff / (2 * w_sum * d2)
        
        g = -diff / (2.0 * w_sum * d2 + 1e-8)
        
        # Apply
        delta = d_vec * g[:, None] # (M, 3)
        
        # Update R
        # We need to scatter add.
        # R[p1] += w1 * delta
        # R[p2] -= w2 * delta
        
        # Use index_add
        d1 = delta * inv_m1[:, None]
        d2 = -delta * inv_m2[:, None]
        
        R_curr = R_curr.at[pairs[:, 0]].add(d1)
        R_curr = R_curr.at[pairs[:, 1]].add(d2)
        
        return R_curr

        return R_curr
    
    # Use more iterations for stability
    return jax.lax.fori_loop(0, 100, lambda i, r: body_fn(i, r), R)

def project_momenta(P, R, pairs, mass, shift_fn, tol=1e-6, max_iter=100):
    """Iterative RATTLE projection for momenta."""
    # Project P such that v . r_ij = 0
    # v1 = P1/m1, v2 = P2/m2
    # (v1 - v2) . r12 = 0
    
    inv_m1 = 1.0 / mass[pairs[:, 0], 0]
    inv_m2 = 1.0 / mass[pairs[:, 1], 0]
    w_sum = inv_m1 + inv_m2
    
    r1 = R[pairs[:, 0]]
    r2 = R[pairs[:, 1]]
    r12 = r1 - r2 # Assume free space
    
    def body_fn(i, P_curr):
        v1 = P_curr[pairs[:, 0]] * inv_m1[:, None]
        v2 = P_curr[pairs[:, 1]] * inv_m2[:, None]
        v12 = v1 - v2
        
        # dot = v12 . r12
        dot = jnp.sum(v12 * r12, axis=-1)
        
        # We want (v12 + dv12) . r12 = 0
        # v12.r12 + dv12.r12 = 0
        # dv1 = k * r12 / m1
        # dv2 = -k * r12 / m2
        # dv12 = k * (1/m1 + 1/m2) * r12
        # dv12 . r12 = k * w_sum * r12.r12
        # k * w_sum * d2 = -dot
        # k = -dot / (w_sum * d2)
        
        d2 = jnp.sum(r12**2, axis=-1)
        k = -dot / (w_sum * d2 + 1e-8)
        
        impulse = r12 * k[:, None] # (M, 3)
        
        # Update P (Momentum)
        # P1 += impulse
        # P2 -= impulse
        
        P_curr = P_curr.at[pairs[:, 0]].add(impulse)
        P_curr = P_curr.at[pairs[:, 1]].add(-impulse)
        
        return P_curr

        return P_curr

    return jax.lax.fori_loop(0, 100, lambda i, p: body_fn(i, p), P)


def run_minimization(
  energy_fn: Callable[[Array], Array],
  r_init: Array,
  steps: int = 500,
  dt_start: float = 2e-3,  # 2 fs - typical for MD
  dt_max: float = 4e-3,     # 4 fs max
) -> Array:
  """Run energy minimization using FIRE descent.

  Args:
      energy_fn: Energy function E(R).
      r_init: Initial positions (N, 3).
      steps: Number of minimization steps.
      dt_start: Initial time step.
      dt_max: Maximum time step.

  Returns:
      Minimized positions.

  """
  init_fn, apply_fn = minimize.fire_descent(energy_fn, shift_fn=space.free()[1], dt_start=dt_start, dt_max=dt_max)
  state = init_fn(r_init)

  # JIT the loop body for speed
  @jax.jit
  def step_fn(i, state):  # noqa: ARG001
    return apply_fn(state)

  state = jax.lax.fori_loop(0, steps, step_fn, state)
  return state.position


def run_thermalization(
  energy_fn: Callable[[Array], Array],
  r_init: Array,
  temperature: float = 300.0,
  steps: int = 1000,
  dt: float = 2e-3,  # 2 fs - standard with SHAKE
  gamma: float = 1.0, # 1.0/ps - standard coupling
  mass: Array | float = 1.0,
  use_shake: bool = False,
  constraints: tuple[Array, Array] | None = None,
  key: Array | None = None,
) -> Array:
  """Run NVT thermalization using Langevin dynamics.

  Args:
      energy_fn: Energy function E(R).
      r_init: Initial positions (N, 3).
      temperature: Temperature in Kelvin.
      steps: Number of simulation steps.
      dt: Time step (ps).
      gamma: Friction coefficient (1/ps).
      mass: Particle mass(es).
      use_shake: Whether to use RATTLE constraints.
      constraints: (pairs, lengths) for RATTLE.
      key: PRNG key.

  Returns:
      Final positions.

  """
  if key is None:
    key = jax.random.PRNGKey(0)

  kT = BOLTZMANN_KCAL * temperature
  
  if use_shake and constraints is not None:
      init_fn, apply_fn = rattle_langevin(
        energy_fn,
        shift_fn=space.free()[1],
        dt=dt,
        kT=kT,
        gamma=gamma,
        mass=mass,
        constraints=constraints
      )
  else:
      init_fn, apply_fn = simulate.nvt_langevin(
        energy_fn,
        shift_fn=space.free()[1],
        dt=dt,
        kT=kT,
        gamma=gamma,
        mass=mass
      )
  
  state = init_fn(key, r_init)

  @jax.jit
  def step_fn(i, state):  # noqa: ARG001
    return apply_fn(state)

  state = jax.lax.fori_loop(0, steps, step_fn, state)
  return state.position


def run_simulation(
  system_params: SystemParams,
  r_init: Array,
  temperature: float = 300.0,
  min_steps: int = 500,
  therm_steps: int = 1000,
  dielectric_constant: float = 1.0,
  implicit_solvent: bool = True,
  solvent_dielectric: float = 78.5,
  solute_dielectric: float = 1.0,
  key: Array | None = None,
) -> Array:
  """Run full simulation: Minimization -> Thermalization.

  Args:
      system_params: System parameters.
      r_init: Initial positions.
      temperature: Temperature in Kelvin.
      min_steps: Minimization steps.
      therm_steps: Thermalization steps.
      dielectric_constant: Dielectric constant for screened Coulomb (if implicit_solvent=False).
      implicit_solvent: Whether to use Generalized Born implicit solvent.
      solvent_dielectric: Solvent dielectric for GB.
      solute_dielectric: Solute dielectric for GB.
      key: PRNG key.

  Returns:
      Final positions.

  """
  displacement_fn, _ = space.free()
  energy_fn = system.make_energy_fn(
      displacement_fn, 
      system_params, 
      dielectric_constant=dielectric_constant,
      implicit_solvent=implicit_solvent,
      solvent_dielectric=solvent_dielectric,
      solute_dielectric=solute_dielectric
  )

  # 1. Minimize
  r_min = run_minimization(energy_fn, r_init, steps=min_steps)

  # Extract masses if available, else default to 1.0
  # Convert amu to internal units (kcal/mol/A^2 * ps^2)^-1 ?
  # We want a = F/m in A/ps^2.
  # F in kcal/mol/A. m in amu.
  # Conversion factor is 418.4.
  # m_internal = m_amu / 418.4
  masses = system_params.get("masses", 1.0)
  if not isinstance(masses, float):
      masses = masses / 418.4
  elif masses != 1.0:
      masses = masses / 418.4
  # Extract constraints
  constrained_bonds = system_params.get("constrained_bonds")
  constrained_lengths = system_params.get("constrained_bond_lengths")
  
  use_shake = False
  constraints = None
  if constrained_bonds is not None and constrained_lengths is not None:
      if len(constrained_bonds) > 0:
          use_shake = True
          constraints = (constrained_bonds, constrained_lengths)

  # 2. Thermalize
  r_final = run_thermalization(
    energy_fn,
    r_min,
    temperature=temperature,
    steps=therm_steps,
    mass=masses,
    use_shake=use_shake,
    constraints=constraints,
    key=key
  )

  return r_final
