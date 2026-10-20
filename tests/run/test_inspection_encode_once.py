"""A6: runner.inspect encodes once per structure for the encoder-sharing features (spec 261020).

decoded_node_features, encoded_node_features and edge_features see the same encoder inputs
(same structure, backbone_noise=0.0, same subkey), so inspect must run the encoder once per
structure for all three. The counter counts Python calls of the encode function that inspect
receives from make_encoding_conditional_logits_split_fn. The structure loader is stubbed with
a one-structure batch and a tiny random-weight model.

Needs the RunSpec.precision_config rename (fix/runspec-precision-config-261020), which keeps
InspectionSpecification constructible under xtrax >= 0.4.1b1.
"""

from __future__ import annotations

import types
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aminx.host.runner as runner
import aminx.sampling.conditional_logits as conditional_logits
from aminx.model import Aminx
from aminx.run.specs import InspectionSpecification

L = 12
_FEATURES = ("decoded_node_features", "encoded_node_features", "edge_features")


def _model() -> Aminx:
  return Aminx(
    dropout_rate=0.0,
    node_features=32,
    edge_features=32,
    hidden_features=32,
    num_encoder_layers=1,
    num_decoder_layers=2,
    k_neighbors=8,
    key=jax.random.PRNGKey(3),
  )


def _batch() -> Any:
  rng = np.random.default_rng(1)
  return types.SimpleNamespace(
    coordinates=jnp.asarray(rng.normal(size=(1, L, 4, 3)).astype(np.float32) * 5.0),
    mask=jnp.ones((1, L), dtype=jnp.float32),
    residue_index=jnp.arange(L, dtype=jnp.int32)[None],
    chain_index=jnp.zeros((1, L), dtype=jnp.int32),
    aatype=jnp.zeros((1, L), dtype=jnp.int32),
    atom_mask=None,
    full_atom_mask=None,
  )


def test_a6_inspect_encodes_once_per_structure(monkeypatch: pytest.MonkeyPatch) -> None:
  """A6: one encode call per structure; each feature equals its single-feature run."""
  model = _model()
  calls = [0]
  real_factory = conditional_logits.make_encoding_conditional_logits_split_fn

  def counting_factory(m: Aminx) -> tuple[Any, Any]:
    encode_fn, decode_fn = real_factory(m)

    def counting_encode(*args: Any, **kwargs: Any) -> Any:
      calls[0] += 1
      return encode_fn(*args, **kwargs)

    return counting_encode, decode_fn

  monkeypatch.setattr(
    conditional_logits, "make_encoding_conditional_logits_split_fn", counting_factory,
  )
  monkeypatch.setattr(runner, "prep_protein_stream_and_model", lambda spec: (iter([_batch()]), model))

  def run(features: tuple[str, ...]) -> dict[str, Any]:
    spec = InspectionSpecification(inputs=["x.pdb"], inspection_features=list(features))
    return runner.inspect(spec)

  combined = run(_FEATURES)
  assert calls[0] == 1, "inspect must encode once per structure for the three features"

  for feature in _FEATURES:
    calls[0] = 0
    single = run((feature,))
    assert calls[0] == 1
    np.testing.assert_allclose(
      np.asarray(combined[feature]), np.asarray(single[feature]), rtol=1e-6, atol=0,
    )
