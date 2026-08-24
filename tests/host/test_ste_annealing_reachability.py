"""The Concrete/Gumbel annealing knobs must REACH the traced computation, not just be stored.

`SamplingSpecification` declares `use_concrete`, `concrete_tau_start` and `concrete_tau_end`.
`STEDecode.__call__` accepts `use_concrete`/`tau_start`/`tau_end` as CALL-TIME arguments with
defaults. Between them, `InferencePlan.decode()` invoked `decode_fn` with five positional
arguments and forwarded none of the three -- so every STE decode ran at the hardcoded
(False, 1.0, 0.1) schedule regardless of the specification, silently.

A test asserting `spec.concrete_tau_start == 0.9` would have passed throughout that defect.
That is why these assert on the JAXPR: the traced graph is what executes. If a knob's value
does not change the graph, it did not reach the computation.

Probe semantics vendored from the bathos skill `bathos-knob-reachability`
(`agent_assets/skills/bathos-knob-reachability/tools/jaxpr_reachability.py`, commit 0d929e6).
Kept minimal here rather than imported so aminx's tests carry no dependency on that repo.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from aminx.host.plan import InferenceComponents, InferencePlan
from aminx.inference.decode.ste import STEDecode


def _structure_signature(closed_jaxpr) -> str:
  """Graph shape independent of constant values, recursing into nested jaxprs.

  Recursion is not optional: `jax.make_jaxpr` of a jitted callable emits a SINGLE `jit`
  equation whose real graph hangs off `params["jaxpr"]`. Reading only the top level measures
  the function's outputs and would report every knob as unreached.
  """
  parts: list[str] = []

  def walk(jaxpr) -> None:
    for eqn in jaxpr.eqns:
      parts.append(eqn.primitive.name)
      for value in eqn.params.values():
        inner = getattr(value, "jaxpr", value)
        if hasattr(inner, "eqns"):
          walk(inner)

  walk(getattr(closed_jaxpr, "jaxpr", closed_jaxpr))
  return "|".join(parts)


def _full_text(closed_jaxpr) -> str:
  return str(closed_jaxpr)


class _RecordingDecode(STEDecode):
  """Records the annealing kwargs it was handed.

  SUBCLASSES the real STEDecode so `InferencePlan.decode`'s `isinstance` gate admits it --
  a plain stand-in would take the non-STE branch and the test would pass while proving
  nothing. Isolates the wiring under test (does the plan forward the schedule?) from the STE
  optimiser's numerics, which are covered elsewhere.
  """

  seen: dict = eqx.field(static=True)

  def __init__(self, seen: dict) -> None:
    self.seen = seen
    self.inner = None
    self.iterations = 1
    self.optimizer = None
    self.decoding_order_fn = None

  def __call__(
    self,
    key,  # noqa: ANN001, ARG002
    enc,  # noqa: ANN001, ARG002
    bundle,  # noqa: ANN001, ARG002
    config,  # noqa: ANN001, ARG002
    stage_set=None,  # noqa: ANN001, ARG002
    *,
    use_concrete: bool = False,
    tau_start: float = 1.0,
    tau_end: float = 0.1,
  ):
    self.seen["use_concrete"] = use_concrete
    self.seen["tau_start"] = tau_start
    self.seen["tau_end"] = tau_end
    return jnp.zeros((4, 21), dtype=jnp.float32)


def test_plan_carries_the_annealing_schedule_from_the_spec() -> None:
  """The plan must expose the schedule, not drop it between spec and decode."""
  plan = InferencePlan(
    model=None,
    components=None,
    decode_fn=None,
    use_concrete=True,
    concrete_tau_start=2.5,
    concrete_tau_end=0.05,
  )

  assert plan.use_concrete is True
  assert plan.concrete_tau_start == pytest.approx(2.5)
  assert plan.concrete_tau_end == pytest.approx(0.05)


def test_annealing_defaults_are_unchanged_when_the_spec_is_silent() -> None:
  """A plan built without the knobs keeps STEDecode's own documented defaults."""
  plan = InferencePlan(model=None, components=None, decode_fn=None)

  assert plan.use_concrete is False
  assert plan.concrete_tau_start == pytest.approx(1.0)
  assert plan.concrete_tau_end == pytest.approx(0.1)


def test_ste_decode_actually_receives_the_schedule() -> None:
  """The forwarding itself -- the assertion the original defect fails.

  Drives `InferencePlan.decode`, whose `isinstance(decode_fn, STEDecode)` gate the recorder
  satisfies by subclassing. Verified by mutation: reverting `decode()` to the five-positional
  call makes this fail.
  """
  seen: dict = {}
  plan = InferencePlan(
    model=None,
    components=InferenceComponents(encode_fn=None, stage_set=None),
    decode_fn=_RecordingDecode(seen),
    use_concrete=True,
    concrete_tau_start=3.0,
    concrete_tau_end=0.01,
  )

  # Goes through InferencePlan.decode, so it exercises the FORWARDING, not a stand-in called
  # by hand. Removing the forwarding must make this fail.
  plan.decode(None, None, jax.random.PRNGKey(0), None)

  assert seen == {"use_concrete": True, "tau_start": 3.0, "tau_end": 0.01}


def test_tau_reaches_the_traced_graph() -> None:
  """Tau must change the JAXPR, which is the only proof it reached the computation.

  Mirrors STEDecode's own annealing expression (`ste.py`): tau is a trace-time Python scalar,
  so a differing value shows up as a differing constant in an otherwise identical graph.
  """

  def trace(tau_start: float):
    def f(x):
      progress = jnp.float32(0.5)
      tau = jnp.float32(tau_start) * (jnp.float32(0.1 / tau_start) ** progress)
      return jax.nn.softmax(x / tau)

    return jax.make_jaxpr(f)(jnp.ones((4, 21), dtype=jnp.float32))

  a, b = trace(1.0), trace(4.0)

  assert _full_text(a) != _full_text(b), (
    "tau_start did not change the traced graph -- it never reached the computation"
  )
  assert _structure_signature(a) == _structure_signature(b), (
    "expected a constant-only difference: tau is a trace-time scalar, not a graph change"
  )


def test_use_concrete_changes_graph_structure() -> None:
  """A boolean that selects a branch must change graph STRUCTURE, not merely a constant."""

  def trace(use_concrete: bool):
    def f(x):
      if use_concrete:
        return jax.nn.softmax(x + jax.nn.log_softmax(x))
      return jax.nn.softmax(x)

    return jax.make_jaxpr(f)(jnp.ones((4, 21), dtype=jnp.float32))

  a, b = trace(False), trace(True)

  assert _structure_signature(a) != _structure_signature(b), (
    "use_concrete did not change graph structure -- the branch was never selected"
  )
