"""`multi_state_sharpness` must reach fusion, and `product` must stay a product.

Companion to `test_ar_fusion_honors_strategy.py`. That file covers the 2026-07-28
finding that the AR path ignored `multi_state_strategy` / `state_weights`. This
file covers the *semantic* half of the same problem: even once the weights arrive,
a weight vector constrained to `sum(w) = 1` makes `ProductOfProbabilities` compute
a weighted geometric mean -- a logarithmic opinion pool -- not a product of
experts. The strategy is still named "product" and no error is raised.

WHY THIS IS NOT A REACHABILITY TEST.
`state_weights` reached its consumer and was used. The defect was that one knob was
carrying two independent quantities: which states matter (mixing proportions) and
how concentrated the result is (sharpness). Normalising the first silently pinned
the second to "no sharpening at all". Splitting them into `weights` and `sharpness`
is what these tests lock in -- so the operator can be checked against its own name.

THE DECISIVE INVARIANT, which no reachability harness can express: for S identical
experts a product must return `p^S`, while a `sum(w)=1` pool returns `p` unchanged.
`test_identical_experts_discriminates_product_from_pool` is that check.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from aminx.inference.logits import ProductOfProbabilities, make_stage_set
from aminx.run.spec import build_run_spec
from aminx.run.specs import SamplingSpecification

S, V = 4, 21


def _per_state() -> jnp.ndarray:
  rng = np.random.default_rng(11)
  return jnp.asarray(rng.normal(size=(S, V)).astype("float32"))


def _bias() -> jnp.ndarray:
  return jnp.zeros((V,), dtype=jnp.float32)


SIMPLEX = jnp.asarray([0.25, 0.25, 0.25, 0.25], dtype=jnp.float32)


# ---------------------------------------------------------------------------
# The semantic invariant
# ---------------------------------------------------------------------------


def test_identical_experts_discriminates_product_from_pool() -> None:
  """S identical experts: a product sharpens to p^S, an opinion pool does not.

  This is the check that distinguishes the two operators. If both branches ever
  agree, `product` has silently become an average again.
  """
  single = jnp.asarray(np.random.default_rng(3).normal(size=(V,)).astype("float32"))
  per_state = jnp.broadcast_to(single, (S, V))

  pool = ProductOfProbabilities(weights=SIMPLEX, sharpness=1.0)(per_state)
  product = ProductOfProbabilities(weights=SIMPLEX, sharpness=None)(per_state)

  np.testing.assert_allclose(np.asarray(pool), np.asarray(single), atol=1e-4)
  np.testing.assert_allclose(np.asarray(product), S * np.asarray(single), atol=1e-4)


def test_simplex_weights_with_auto_sharpness_equal_plain_product() -> None:
  """sum(w)=1 + sharpness=None reproduces the unweighted product sum(L_i) exactly.

  This is the recipe that lets weights stay a genuine simplex -- which a mixture
  design requires -- without the operator degrading to an average.
  """
  per_state = _per_state()
  weighted = ProductOfProbabilities(weights=SIMPLEX, sharpness=None)(per_state)
  plain = jnp.sum(per_state, axis=0)
  np.testing.assert_allclose(np.asarray(weighted), np.asarray(plain), atol=1e-4)


def test_sharpness_is_independent_of_mixing_proportions() -> None:
  """Changing sharpness at fixed weights changes only the scale, not the direction.

  Mixing and concentration are orthogonal; this pins that they stay so.
  """
  per_state = _per_state()
  base = np.asarray(ProductOfProbabilities(weights=SIMPLEX, sharpness=1.0)(per_state))
  scaled = np.asarray(ProductOfProbabilities(weights=SIMPLEX, sharpness=2.5)(per_state))
  np.testing.assert_allclose(scaled, 2.5 * base, atol=1e-4)


# ---------------------------------------------------------------------------
# Path-awareness: both fusion slots must see the same sharpness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sharpness", [1.0, 2.0, None])
def test_ar_and_non_ar_slots_agree_on_sharpness(sharpness: float | None) -> None:
  """AR and non-AR fusion must produce identical output for any sharpness.

  a23 made these the same instance; this asserts the new knob cannot reintroduce
  a divergence between the two paths.
  """
  per_state, bias = _per_state(), _bias()
  stage_set = make_stage_set(
    strategy="product",
    state_weights=SIMPLEX,
    sharpness=sharpness,
  )
  ar = np.asarray(stage_set.ar_logit_transform(per_state, bias))
  non_ar = np.asarray(stage_set.logit_transform(per_state, bias))
  np.testing.assert_allclose(ar, non_ar, atol=1e-6)


def test_ar_transform_responds_to_sharpness() -> None:
  """Changing sharpness must change AR-fused logits.

  The analogue of the a23 regression: a knob that is accepted but discarded on
  the AR path would leave these bit-identical.
  """
  per_state, bias = _per_state(), _bias()
  flat = make_stage_set(
    strategy="product", state_weights=SIMPLEX, sharpness=1.0,
  ).ar_logit_transform(per_state, bias)
  sharp = make_stage_set(
    strategy="product", state_weights=SIMPLEX, sharpness=None,
  ).ar_logit_transform(per_state, bias)
  assert not np.allclose(np.asarray(flat), np.asarray(sharp), atol=1e-6), (
    "AR fusion ignored `sharpness` -- product semantics are not reaching the "
    "autoregressive decode."
  )


# ---------------------------------------------------------------------------
# Spec plumbing: SamplingSpecification -> SamplingConfig
# ---------------------------------------------------------------------------


def test_spec_carries_sharpness_to_config() -> None:
  """A sharpness set on the user-facing spec survives into the internal config."""
  spec = SamplingSpecification(
    inputs=[], multi_state_strategy="product", multi_state_sharpness=None,
  )
  assert build_run_spec(spec).sampling.multi_state_sharpness is None


def test_spec_default_sharpness_is_identity() -> None:
  """Omitting sharpness preserves pre-existing unscaled behaviour."""
  spec = SamplingSpecification(inputs=[], multi_state_strategy="product")
  assert build_run_spec(spec).sampling.multi_state_sharpness == 1.0


def test_none_sharpness_is_not_coalesced_to_default() -> None:
  """`None` is a value ("use S"), not a missing field.

  An `or`-style fallback in the spec->config conversion would rewrite it to 1.0
  and silently restore opinion-pool semantics -- exactly the class of silent
  downgrade this whole change exists to remove.
  """
  spec = SamplingSpecification(
    inputs=[], multi_state_strategy="product", multi_state_sharpness=None,
  )
  config = build_run_spec(spec).sampling
  assert config.multi_state_sharpness is None, (
    "sharpness=None was coalesced to a default; 'use S' semantics were lost."
  )
