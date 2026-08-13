"""The AR path must honour multi_state_strategy, state_weights and temperature.

Regression tests for the 2026-07-28 finding that `make_stage_set` wired
`ar_logit_transform=ARLogitFuse()` -- constructed with no strategy and no weights --
so the autoregressive decode fused states with a hardcoded unweighted
`jnp.mean(logits, axis=0) + bias`. Callers asking for `"product"` fusion with
non-uniform `state_weights` silently got an unweighted arithmetic mean, and the
manifest recorded the strategy they asked for. Weighted product-of-experts fusion
did not exist on the AR path.

WHY THESE TESTS, AND NOT THE EXISTING KNOB HARNESS.
`tests/host/test_knob_differential.py` asserts *reachability*: does knob X arrive
at consumer Y. Both `multi_state_strategy` and `state_weights` DID arrive -- at
`make_stage_set`, which then used them for one of its two output slots. A
reachability harness cannot see a knob that reaches its consumer and is dropped on
one of two code paths. These tests are *path-aware*: they assert the AR slot
responds, and that the two slots agree.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from aminx.inference.logits import ARLogitFuse, make_stage_set

S, V = 3, 21


def _per_state() -> jnp.ndarray:
  rng = np.random.default_rng(0)
  return jnp.asarray(rng.normal(size=(S, V)).astype("float32"))


def _bias() -> jnp.ndarray:
  return jnp.zeros((V,), dtype=jnp.float32)


UNIFORM = jnp.asarray([1 / 3, 1 / 3, 1 / 3], dtype=jnp.float32)
SKEWED = jnp.asarray([0.90, 0.05, 0.05], dtype=jnp.float32)
FIRST_ONLY = jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float32)


@pytest.mark.parametrize("strategy", ["arithmetic_mean", "geometric_mean", "product"])
def test_ar_transform_responds_to_state_weights(strategy: str) -> None:
  """THE REGRESSION. Changing state_weights must change AR-fused logits.

  Fails on the pre-fix wiring for every strategy: ARLogitFuse ignored weights
  entirely, so uniform and skewed produced bit-identical output.
  """
  per_state, bias = _per_state(), _bias()
  uni = make_stage_set(strategy=strategy, state_weights=UNIFORM).ar_logit_transform(per_state, bias)
  skew = make_stage_set(strategy=strategy, state_weights=SKEWED).ar_logit_transform(per_state, bias)
  assert not np.allclose(np.asarray(uni), np.asarray(skew), atol=1e-6), (
    f"AR fusion ignored state_weights for strategy={strategy!r} -- weighted "
    "product-of-experts is not reaching the autoregressive decode."
  )


def test_ar_transform_responds_to_strategy() -> None:
  """Changing multi_state_strategy must change AR-fused logits."""
  per_state, bias = _per_state(), _bias()
  outs = {
    s: np.asarray(make_stage_set(strategy=s, state_weights=UNIFORM).ar_logit_transform(per_state, bias))
    for s in ("arithmetic_mean", "geometric_mean", "product")
  }
  assert not np.allclose(outs["product"], outs["arithmetic_mean"], atol=1e-6), (
    "AR fusion ignored multi_state_strategy: 'product' matched 'arithmetic_mean'."
  )


def test_ar_transform_respects_strategy_temperature() -> None:
  """strategy_temperature must reach the AR path (geometric_mean consumes it).

  Also pins the DIRECTION: smaller temperature sharpens. Guards against a
  temperature-inversion regression, where a plausible-looking change flips the
  convention and silently uniformises every distribution.
  """
  bias = _bias()
  sharp = jnp.asarray(np.tile((np.eye(1, V, 0) * 10.0), (S, 1)).astype("float32"))

  def peak(temp: float) -> float:
    out = make_stage_set(
      strategy="geometric_mean", strategy_temperature=temp, state_weights=UNIFORM,
    ).ar_logit_transform(sharp, bias)
    shifted = out - jnp.max(out)
    probs = jnp.exp(shifted) / jnp.sum(jnp.exp(shifted))
    return float(jnp.max(probs))

  hot, cold = peak(10.0), peak(0.1)
  assert cold > hot, f"temperature had no effect on AR path (cold={cold}, hot={hot})"
  assert cold > 0.9, f"small temperature must SHARPEN; got peak prob {cold} (convention inverted?)"
  assert hot < 0.5, f"large temperature must FLATTEN; got peak prob {hot} (convention inverted?)"


@pytest.mark.parametrize("strategy", ["arithmetic_mean", "geometric_mean", "product"])
def test_zero_weight_drops_its_state_on_ar_path(strategy: str) -> None:
  """A zero-weighted state must not contribute to AR fusion.

  tev_design's k=9 weight variants rely on this: profiles like
  ``crystal_only=[1,1,0,0,0,0,0,0,0]`` express "use only these states" as zero
  weights. Under the pre-fix unweighted mean a zero-weighted state contributed
  exactly as much as every other one, so those profiles were inert.
  """
  per_state, bias = _per_state(), _bias()
  fused = make_stage_set(strategy=strategy, state_weights=FIRST_ONLY).ar_logit_transform(
    per_state, bias,
  )
  np.testing.assert_allclose(
    np.asarray(fused), np.asarray(per_state[0]), atol=1e-4,
    err_msg=f"strategy={strategy!r}: zero-weighted states still contributed to AR fusion",
  )


@pytest.mark.parametrize("strategy", ["arithmetic_mean", "geometric_mean", "product"])
def test_ar_and_non_ar_paths_agree(strategy: str) -> None:
  """The two fusion slots must not diverge.

  The original defect was precisely a divergence: `logit_transform` honoured the
  configuration while `ar_logit_transform` did not. Asserting behavioural equality
  keeps them pinned together even if the implementation stops sharing one instance.
  """
  per_state, bias = _per_state(), _bias()
  ss = make_stage_set(strategy=strategy, state_weights=SKEWED, strategy_temperature=1.0)
  np.testing.assert_allclose(
    np.asarray(ss.ar_logit_transform(per_state, bias)),
    np.asarray(ss.logit_transform(per_state, bias)),
    atol=1e-6,
    err_msg=f"strategy={strategy!r}: AR and non-AR fusion disagree",
  )


def test_arlogitfuse_still_available_as_explicit_unweighted_mean() -> None:
  """ARLogitFuse remains valid when an unweighted mean is what you actually want.

  The fix removes it as the silent DEFAULT, not as an option.
  """
  per_state, bias = _per_state(), _bias()
  np.testing.assert_allclose(
    np.asarray(ARLogitFuse()(per_state, bias)),
    np.asarray(jnp.mean(per_state, axis=0) + bias),
    atol=1e-6,
  )
