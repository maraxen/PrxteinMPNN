"""Regression test: unconditional ``Decoder.__call__`` honors the ``inference`` flag.

Locks in the behavior investigated in spec ``260709`` §4.1. The unconditional
decode path had a dead ``inference = True`` assignment (F841) that was never
threaded into the layer call; dropout was nonetheless correctly disabled at
inference via two independent guards (``DecoderLayer`` re-derives ``inference``
on ``key=None``; ``Dropout`` returns ``x`` on ``key=None``). An ``inference``
parameter was then added for symmetry with ``call_conditional`` so the
energy/score readout path can request deterministic decoding while still
supplying a key. These tests pin all three behaviors:

- ``key=None``                          -> deterministic (dropout disabled)
- ``inference=True`` with a real key     -> deterministic across keys (dropout off)
- ``inference=False`` with different keys -> outputs differ (dropout active)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from aminx.model.decoder import Decoder

_DIM = 16
_HIDDEN = 32
_LAYERS = 2
_L = 8
_K = 5


def _make_decoder() -> Decoder:
    # High dropout rate so its effect is unmistakable when active.
    return Decoder(
        node_features=_DIM,
        edge_features=_DIM,
        hidden_features=_HIDDEN,
        num_layers=_LAYERS,
        dropout_rate=0.5,
        key=jax.random.PRNGKey(123),
    )


def _inputs(seed: int = 0):
    kn, ke, ki = jax.random.split(jax.random.PRNGKey(seed), 3)
    node = jax.random.normal(kn, (_L, _DIM))
    edge = jax.random.normal(ke, (_L, _K, _DIM))
    neighbors = jax.random.randint(ki, (_L, _K), 0, _L)
    mask = jnp.ones((_L,))
    return node, edge, neighbors, mask


def test_key_none_is_deterministic() -> None:
    """key=None must disable dropout -> identical outputs across calls."""
    decoder = _make_decoder()
    node, edge, nei, mask = _inputs()
    out_a = decoder(node, edge, nei, mask, key=None)
    out_b = decoder(node, edge, nei, mask, key=None)
    assert jnp.allclose(out_a, out_b)


def test_inference_true_disables_dropout_even_with_key() -> None:
    """inference=True must disable dropout even when a real key is supplied."""
    decoder = _make_decoder()
    node, edge, nei, mask = _inputs()
    out1 = decoder(node, edge, nei, mask, inference=True, key=jax.random.PRNGKey(1))
    out2 = decoder(node, edge, nei, mask, inference=True, key=jax.random.PRNGKey(2))
    assert jnp.allclose(out1, out2)


def test_training_mode_dropout_is_stochastic() -> None:
    """inference=False with distinct keys must produce distinct outputs (dropout on)."""
    decoder = _make_decoder()
    node, edge, nei, mask = _inputs()
    out1 = decoder(node, edge, nei, mask, inference=False, key=jax.random.PRNGKey(1))
    out2 = decoder(node, edge, nei, mask, inference=False, key=jax.random.PRNGKey(2))
    assert not jnp.allclose(out1, out2)
