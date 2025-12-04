"""Shared fixtures for physics tests."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest


@pytest.fixture
def simple_charges() -> jax.Array:
    """Simple charge distribution for testing."""
    return jnp.array([1.0, -1.0, 0.5, -0.5])


@pytest.fixture
def simple_positions() -> jax.Array:
    """Simple atomic positions for testing."""
    return jnp.array([
        [0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [5.0, 5.0, 0.0],
    ])


@pytest.fixture
def backbone_positions_single_residue() -> jax.Array:
    """Backbone positions for a single idealized residue [N, CA, C, O, CB]."""
    return jnp.array([[
        [0.0, 0.0, 0.0],      # N
        [1.5, 0.0, 0.0],      # CA
        [2.5, 1.0, 0.0],      # C
        [2.5, 2.0, 0.0],      # O
        [1.5, 0.0, 1.5],      # CB (perpendicular to backbone plane)
    ]])

@pytest.fixture
def backbone_charges_single_residue() -> jax.Array:
    """Backbone charges for a single residue [N, CA, C, O, CB]."""
    return jnp.array([[-0.3, 0.1, 0.5, -0.5, 0.2]])


@pytest.fixture
def backbone_positions_multi_residue() -> jax.Array:
    """Backbone positions for multiple residues."""
    # Create 3 residues in a simple extended conformation
    residue1 = jnp.array([
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [2.5, 1.0, 0.0],
        [2.5, 2.0, 0.0],
        [1.5, 0.0, 1.5],
    ])

    residue2 = jnp.array([
        [3.8, 1.5, 0.0],
        [5.3, 1.5, 0.0],
        [6.3, 2.5, 0.0],
        [6.3, 3.5, 0.0],
        [5.3, 1.5, 1.5],
    ])

    residue3 = jnp.array([
        [7.6, 3.0, 0.0],
        [9.1, 3.0, 0.0],
        [10.1, 4.0, 0.0],
        [10.1, 5.0, 0.0],
        [9.1, 3.0, 1.5],
    ])

    return jnp.stack([residue1, residue2, residue3])

@pytest.fixture
def backbone_charges_multi_residue() -> jax.Array:
    """Backbone charges for multiple residues."""
    # Charges for N, CA, C, O, CB
    residue_charges = jnp.array([-0.3, 0.1, 0.5, -0.5, 0.2])
    return jnp.tile(residue_charges, (3, 1))  # 3 residues


@pytest.fixture
def lj_parameters() -> dict[str, jax.Array]:
    """Simple LJ parameters for testing."""
    return {
        "sigma": jnp.array([3.5, 3.0, 2.5, 3.2]),
        "epsilon": jnp.array([0.1, 0.15, 0.08, 0.12]),
    }


@pytest.fixture
def temp_ff_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for force field files."""
    ff_dir = tmp_path / "force_fields"
    ff_dir.mkdir()
    return ff_dir
