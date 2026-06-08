"""Plain callable / lambda support in StageSet slots.

eqx.filter_jit automatically marks non-array objects (plain Python functions,
lambdas) as static, so any callable works in a StageSet slot without requiring
an eqx.Module subclass. This is the preferred pattern for stateless strategies.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from aminx.types.stages import StageSet


# ---------------------------------------------------------------------------
# Plain function in logit_transform
# ---------------------------------------------------------------------------

def test_plain_fn_in_logit_transform_survives_filter_jit():
    """A plain function in logit_transform is accepted by eqx.filter_jit."""
    def mean_fuse(per_state, bias=None):
        result = jnp.mean(per_state, axis=0)
        return result if bias is None else result + bias

    ss = StageSet(logit_transform=mean_fuse)

    @eqx.filter_jit
    def run(stage_set, x):
        return stage_set.logit_transform(x)

    x = jnp.ones((3, 5, 21))
    out = run(ss, x)
    assert out.shape == (5, 21)


def test_lambda_in_logit_transform_survives_filter_jit():
    """A lambda in logit_transform is accepted by eqx.filter_jit."""
    ss = StageSet(logit_transform=lambda per_state, bias=None: jnp.sum(per_state, axis=0))

    @eqx.filter_jit
    def run(stage_set, x):
        return stage_set.logit_transform(x)

    out = run(ss, jnp.ones((2, 4, 21)))
    assert out.shape == (4, 21)


# ---------------------------------------------------------------------------
# Plain function in ar_logit_transform
# ---------------------------------------------------------------------------

def test_plain_fn_in_ar_logit_transform_survives_filter_jit():
    """A plain function in ar_logit_transform is accepted by eqx.filter_jit."""
    def ar_fuse(logits, bias):
        return jnp.mean(logits, axis=0) + bias

    ss = StageSet(ar_logit_transform=ar_fuse)

    @eqx.filter_jit
    def run(stage_set, logits, bias):
        return stage_set.ar_logit_transform(logits, bias)

    logits = jnp.ones((3, 21))   # (S, V)
    bias = jnp.zeros((21,))
    out = run(ss, logits, bias)
    assert out.shape == (21,)


# ---------------------------------------------------------------------------
# Plain function in tie_group_fuse
# ---------------------------------------------------------------------------

def test_plain_fn_in_tie_group_fuse_survives_filter_jit():
    """A plain function in tie_group_fuse is accepted by eqx.filter_jit."""
    def logsumexp_fuse(logits, mask):
        group = jnp.where(mask[:, None], logits, -jnp.inf)
        n = jnp.sum(mask)
        return jax.scipy.special.logsumexp(group, axis=0) - jnp.log(jnp.maximum(n, 1))

    ss = StageSet(tie_group_fuse=logsumexp_fuse)

    @eqx.filter_jit
    def run(stage_set, logits, mask):
        return stage_set.tie_group_fuse(logits, mask)

    logits = jnp.ones((5, 21))   # (L, V)
    mask = jnp.array([True, True, False, False, False])
    out = run(ss, logits, mask)
    assert out.shape == (21,)


# ---------------------------------------------------------------------------
# eqx.Module with array leaves still works (regression guard)
# ---------------------------------------------------------------------------

def test_eqx_module_with_weights_still_works():
    """eqx.Module with traced weight leaves works alongside plain-callable slots."""
    class WeightedMean(eqx.Module):
        weights: jax.Array

        def __call__(self, per_state, bias=None):
            w = self.weights / self.weights.sum()
            return jnp.einsum("s,s...v->...v", w, per_state)

    weights = jnp.array([1.0, 0.5])
    ss = StageSet(logit_transform=WeightedMean(weights=weights))

    @eqx.filter_jit
    def run(stage_set, x):
        return stage_set.logit_transform(x)

    out = run(ss, jnp.ones((2, 4, 21)))
    assert out.shape == (4, 21)


# ---------------------------------------------------------------------------
# Different plain fns trigger recompilation (static identity check)
# ---------------------------------------------------------------------------

def test_different_plain_fns_retrace():
    """Swapping to a different fn object causes filter_jit to retrace (static)."""
    call_count = [0]

    @eqx.filter_jit
    def run(stage_set, x):
        call_count[0] += 1
        return stage_set.logit_transform(x)

    fn_a = lambda x, bias=None: jnp.mean(x, axis=0)  # noqa: E731
    fn_b = lambda x, bias=None: jnp.sum(x, axis=0)   # noqa: E731

    x = jnp.ones((2, 4, 21))
    run(StageSet(logit_transform=fn_a), x)
    run(StageSet(logit_transform=fn_a), x)  # same fn → no retrace
    run(StageSet(logit_transform=fn_b), x)  # different fn → retrace

    # At least 2 distinct traces happened (fn_a and fn_b)
    assert call_count[0] >= 2
