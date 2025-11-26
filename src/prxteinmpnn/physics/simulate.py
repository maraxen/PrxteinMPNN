"""Simulation runner for minimization and thermalization."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax_md import minimize, simulate, space, util

from prxteinmpnn.physics import system
from prxteinmpnn.physics.constants import BOLTZMANN_KCAL
from prxteinmpnn.physics.jax_md_bridge import SystemParams

Array = util.Array


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
  dt: float = 2e-3,  # 2 fs - typical for MD
  gamma: float = 0.1,
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
      key: PRNG key.

  Returns:
      Final positions.

  """
  if key is None:
    key = jax.random.PRNGKey(0)

  kT = BOLTZMANN_KCAL * temperature
  
  init_fn, apply_fn = simulate.nvt_langevin(
    energy_fn,
    shift_fn=space.free()[1],
    dt=dt,
    kT=kT,
    gamma=gamma
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
  key: Array | None = None,
) -> Array:
  """Run full simulation: Minimization -> Thermalization.

  Args:
      system_params: System parameters.
      r_init: Initial positions.
      temperature: Temperature in Kelvin.
      min_steps: Minimization steps.
      therm_steps: Thermalization steps.
      key: PRNG key.

  Returns:
      Final positions.

  """
  displacement_fn, _ = space.free()
  displacement_fn, _ = space.free()
  energy_fn = system.make_energy_fn(displacement_fn, system_params, dielectric_constant=dielectric_constant)

  # 1. Minimize
  r_min = run_minimization(energy_fn, r_init, steps=min_steps)

  # 2. Thermalize
  r_final = run_thermalization(
    energy_fn,
    r_min,
    temperature=temperature,
    steps=therm_steps,
    key=key
  )

  return r_final
