---
title: aminx RuntimeBundle/InputResolver compose xtrax.run.resolver, do not subclass it
decision_id: 260630_runtimebundle-inputresolver-compose-not-subclass
date: 2026-06-30
status: Accepted
decision_type: architectural
relates_to: backlog #1910, #2891, #2892, #2894
---

## Status: Accepted 2026-06-30

This ADR documents the integration pattern for the not-yet-built `aminx/run/bundle.py`
(`RuntimeBundle`, backlog #1910) and its `InputResolver` counterpart, scoped by backlog #2891.

## Context

An audit of aminx's xtrax adoption (260630) found that `aminx.tiling` is a deliberate,
fully-spec'd parallel reimplementation of `xtrax.tiling` (EPIC #1541, T2-T5, not yet
executed). Separately, it flagged that `RuntimeBundle`/`InputResolver` (#1910, open) was
about to repeat the same parallel-reimplementation pattern *without* being covered by
#1541's scope — i.e. aminx was on track to build its own `RuntimeBundle` with no bridge to
`xtrax.run.resolver.RuntimeBundle`/`InputResolver`, the same names doing unrelated things in
two places.

The user's initial framing was "subclass/wrap `xtrax.run.resolver`." Before writing any
code (#1910 is not yet built — nothing to refactor, only a design to fix in advance), the
actual shape of `xtrax.run.resolver` was checked against that framing, since two prior
items in this session (#2892, #2894) only reached a stable design *after* discovering a
literal-subclassing/nominal-typing approach required upstream xtrax changes. The goal here
was to choose a pattern that needs none.

## Investigation: why literal subclassing breaks

`xtrax.run.resolver.RuntimeBundle` (verify: `xtrax/src/xtrax/run/resolver.py`):

```python
@dataclass
class RuntimeBundle:
    iterator: VmapIterator | SafeMapIterator | JaxScanIterator | BucketIterator | MapIterator | ScanIterator | None
    model: eqx.Module
```

This is a **plain, non-frozen** `@dataclass` — not `eqx.Module`. Backlog #1910's own title
is *"Create aminx/run/bundle.py with RuntimeBundle **frozen** dataclass."* Python's
`dataclasses` module raises `TypeError` at class-definition time if a frozen dataclass
inherits a non-frozen one (or vice versa) — literal subclassing would fail immediately. If
aminx instead wanted `RuntimeBundle` as an `eqx.Module` (plausible, since it holds a model
and may need to flow through JIT as a pytree), that is *also* incompatible with inheriting
a vanilla stdlib dataclass — different, incompatible metaclass machinery.

This is unlike `aminx.run.spec.RunSpec`, which *does* successfully subclass
`xtrax.run.RunSpec` (`src/aminx/run/spec.py:21`, already landed via RS-6). That works
specifically because **both** layers are `eqx.Module`-based — consistent machinery
throughout the inheritance chain. `RuntimeBundle` does not have that property today.

Checked whether `RuntimeBundle`'s mutability is load-bearing inside xtrax (i.e. whether
asking upstream to freeze it would be safe): `grep -rn "RuntimeBundle\b" xtrax/{src,tests}`
shows it is constructed and exported, but never mutated anywhere in xtrax's own codebase —
its non-frozen-ness looks like an unconsidered default, not a deliberate requirement. Even
so, depending on xtrax changing that default (now or later) is exactly the dependency this
ADR is choosing to avoid.

`xtrax.run.resolver.InputResolver` is a `@runtime_checkable Protocol`, and its own
docstring is explicit: *"Do NOT make this a generic Protocol[S, T]... Use
`@functools.singledispatch` for subclass-specific implementations."* Protocols are
structural — there is no subclassing question here at all; any callable matching
`__call__(spec, bundle) -> FeatureBatch` satisfies it.

`FeatureBatch = NewType("FeatureBatch", dict[str, Any])` has zero runtime identity; any
`dict[str, Any]`-shaped return satisfies it trivially.

## Decision: compose, don't subclass

1. **`RuntimeBundle`**: aminx's `RuntimeBundle` (frozen dataclass or `eqx.Module`, aminx's
   choice) **composes** an `xtrax.run.resolver.RuntimeBundle` rather than inheriting from
   it — either holds one as a field, or exposes a method (e.g. `to_xtrax()`) that
   constructs one on demand for the boundary where xtrax's tiling dispatch
   (`make_axis_dispatch`, iterators) needs it. This has zero dependency on xtrax's
   mutability choices, present or future, and works regardless of whether aminx settles on
   a frozen dataclass or an `eqx.Module`.
2. **`InputResolver`**: implement aminx's input resolution as a plain callable built around
   `@functools.singledispatch`, dispatching on the `RunSpec` subclass, exactly as
   xtrax's own docstring prescribes. No subclassing, no Protocol inheritance — structural
   satisfaction only.
3. **`FeatureBatch`**: use as-is; no special handling needed.

**Net result: this design needs zero upstream xtrax changes**, unlike #2892/#2894 in this
same session, which both required a structural-Protocol change to `xtrax.eda`/`xtrax.stages`
before they reached a non-fragile design.

## Consequences

- When #1910 is implemented, its `RuntimeBundle` must NOT be declared as
  `class RuntimeBundle(xtrax.run.resolver.RuntimeBundle)`. Composition only.
- aminx's `InputResolver` implementation should live as a `functools.singledispatch`
  function (or a thin callable wrapping one), registered per `RunSpec` subclass, mirroring
  the pattern already documented in xtrax's own docstring and in the `/using-xtrax` skill's
  `InputResolver` example.
- If a future need arises to make `xtrax.run.resolver.RuntimeBundle` frozen or an
  `eqx.Module` upstream, re-evaluate this decision — it may simplify the composition layer
  to a thinner one, but should not be assumed as a prerequisite for #1910.
- No code changes from this ADR alone; it gates the design of #1910 when that item is
  picked up.
