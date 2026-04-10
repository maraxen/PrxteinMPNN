"""Fast parity tests based on deterministic golden fixtures."""

from __future__ import annotations

import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.parity.reference_utils import project_root


@pytest.mark.parity_fast
def test_linear_layer_golden_fixture() -> None:
  """Validate fixed linear-layer outputs match checked-in golden fixture."""
  fixture_dir = project_root() / "tests/golden/parity"
  fixture_path = fixture_dir / "linear_layer_fixture.npz"
  metadata_path = fixture_dir / "metadata.json"
  if not fixture_path.exists() or not metadata_path.exists():
    pytest.skip("Golden parity fixtures missing. Run scripts/generate_parity_golden_fixtures.py.")

  with np.load(fixture_path) as data:
    inputs = data["inputs"]
    weight = data["weight"]
    bias = data["bias"]
    expected = data["expected"]

  metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
  atol = float(metadata["tolerances"]["atol"])
  rtol = float(metadata["tolerances"]["rtol"])

  layer = eqx.nn.Linear(
    inputs.shape[-1],
    expected.shape[-1],
    key=jax.random.PRNGKey(0),
  )
  layer = eqx.tree_at(lambda model: model.weight, layer, jnp.array(weight))
  layer = eqx.tree_at(lambda model: model.bias, layer, jnp.array(bias))

  observed = np.asarray(jax.vmap(layer)(jnp.array(inputs)))
  np.testing.assert_allclose(observed, expected, rtol=rtol, atol=atol)
