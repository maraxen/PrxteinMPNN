# Spec: port `MappedBy`/`resolve_mapped_by` upstream to xtrax as a generic primitive

**Task:** 260826_chain-selection-vendor-superset-audit · **Status:** DRAFT, pending review ·
**Scope:** xtrax (external dependency, consumed by aminx via PyPI, pinned `xtrax[io]==0.4.0a5`
— no local editable checkout in this workspace as of this writing; `pyproject.toml` documents
the `uv.toml`/`[sources]` override for live co-development against a sibling `../xtrax`
checkout if/when one exists). This spec is written to be actionable by whoever picks it up, in
xtrax's own repo, not implemented here. Tracked as praxia debt #1465 (filed with a caveat about
workspace-binding uncertainty on that filing — verify it landed against the right workspace
before trusting it as the sole record of this work).

## Why this exists

G1 (`.praxia/docs/specs/260826_chain-selection-gap-closure.md`) implemented
`MappedBy`/`resolve_mapped_by` — "a per-item batch value supplied as `{key: value}`, keyed by a
declared input field" — as a minimal, aminx-local primitive
(`src/aminx/run/batch_mapping.py`), deliberately scoped to `by="path"` only and to
`SamplingSpecification.fixed_mask`/`fixed_tokens`. The mechanism itself is generic: nothing
about "resolve a per-batch-item value keyed by that item's identity, broadcast a plain value to
every item, raise if any item is missing an entry" is specific to aminx, to sampling, or to
`fixed_mask`. It belongs in xtrax (the batching/run-spec framework aminx and other downstream
projects build on) so future xtrax-based projects get it for free instead of reimplementing the
same primitive per project — which is exactly how aminx ended up needing to build it from
scratch for this one field.

## What NOT to do

Do not use this port as a chance to expand scope beyond what G1 actually needed and validated.
Specifically:

- Do not implement `by` values beyond what a real caller has asked for. G1 shipped `by="path"`
  only, with every other value raising `NotImplementedError` naming itself unsupported — carry
  that same discipline upstream. A generic primitive is not an invitation to speculatively
  support `by="chain_id"`, `by="index"`, etc. "while we're in there."
- Do not couple this to `SamplingSpecification`, `fixed_mask`, or anything aminx-specific. The
  upstream version's job is: given a per-item identity list for the current batch and a
  possibly-`MappedBy` value, resolve or broadcast. What "item identity" means, and which spec
  fields accept a `MappedBy`, stays a downstream (aminx) decision.

## Proposed home and shape

`xtrax/run/batch_mapping.py` (mirrors aminx's own module name; sits alongside
`xtrax/run/spec.py`'s `RunSpec` in the existing `xtrax.run` execution-config layer — see that
module's public `__init__.py` for the existing `RunSpec`/`SinkSpec`/`InputResolver` export
pattern this would join), exporting `MappedBy` and `resolve_mapped_by` from `xtrax.run`
alongside `RunSpec`.

Proposed generic shape, adapted from `src/aminx/run/batch_mapping.py`'s working implementation
(full current source there — this is the reference to port, not to redesign from scratch):

```python
@dataclass(frozen=True, slots=True)
class MappedBy[T]:
  """A per-item value supplied as {key: value}, keyed by a declared input field.

  `by` names which field of the caller's own item-identity scheme the mapping is keyed by
  (e.g. "path" for a filesystem-path-derived id) -- xtrax does not interpret `by` itself; the
  caller's resolve_mapped_by invocation supplies the actual per-batch identity list, and
  `by` is carried only for the NotImplementedError/error-message and for a caller-side
  `_SUPPORTED_MAPPED_BY`-style allowlist (see "Composition with a caller's own `by` allowlist"
  below -- xtrax does NOT own that allowlist).
  """
  by: str
  mapping: dict[str, T]


def resolve_mapped_by(value: T | MappedBy[T] | None, *, item_ids: Sequence[str],
                       field_name: str) -> list[T | None]:
  """Resolve a possibly-MappedBy value into one value per item, id-order matched.

  Non-MappedBy value broadcasts to every item. MappedBy resolves per item id; an item id
  absent from the mapping raises ValueError naming the missing ids -- no implicit fallback.
  """
```

### Composition with a caller's own `by` allowlist

G1's aminx-local `MappedBy.__post_init__` validates `by` against a hardcoded
`_SUPPORTED_MAPPED_BY = frozenset({"path"})` and raises `NotImplementedError` for anything
else — this validation is aminx's own policy (which `by` values `fixed_mask` accepts), not a
generic xtrax concern. The ported version should NOT hardcode any allowlist in
`xtrax.run.batch_mapping.MappedBy.__post_init__` — either drop the `by`-validation
responsibility entirely (let any downstream caller reject unsupported `by` values in its own
field-resolution wrapper, the way aminx's `_resolve_mapped_by_field` already validates
`batch_structure_ids is not None` as a separate concern from `MappedBy` construction), or accept
an optional `supported_by: frozenset[str] | None = None` constructor-time parameter that a
caller passes explicitly (xtrax provides the check-and-raise mechanism, the caller supplies the
allowed set). Prefer the former (no allowlist in xtrax at all) unless the port author finds a
concrete case where sharing the check-and-raise codepath across multiple xtrax-based projects
is worth the extra parameter — decide this once actually porting, not speculatively here.

## Migration plan for aminx once the upstream version lands

1. Once `xtrax.run.batch_mapping.MappedBy`/`resolve_mapped_by` ship in a released xtrax version,
   bump aminx's `xtrax[io]==...` pin in `pyproject.toml` to that version.
2. Replace `src/aminx/run/batch_mapping.py`'s contents with a thin re-export:
   `from xtrax.run.batch_mapping import MappedBy, resolve_mapped_by` (or
   `from xtrax.run import MappedBy, resolve_mapped_by` if xtrax re-exports from its top-level
   `run` `__init__.py`, matching the `RunSpec`/`SinkSpec` pattern) — keep the aminx-local
   `_SUPPORTED_MAPPED_BY = frozenset({"path"})` gate wherever `SamplingSpecification` actually
   constructs/validates a `MappedBy` (this is aminx's field-level policy, not xtrax's, per the
   allowlist discussion above), so aminx's own "only `by='path'` is implemented" contract for
   `fixed_mask`/`fixed_tokens` is unchanged for existing callers.
3. Do not remove `src/aminx/run/batch_mapping.py` outright — keep it as the re-export module so
   `from aminx.run.batch_mapping import MappedBy` (used by `specs.py`, `_sampling_helper.py`,
   `runner.py`, `multistate_poe.py`, and their tests) keeps working without touching every call
   site. Update its module docstring to point at the upstream xtrax module as the source of
   truth.
4. Re-run `tests/host/test_mapped_by_fixed_mask.py`,
   `tests/host/test_fixed_mask_score_jacobian_guard.py`'s MappedBy cases, and
   `tests/sampling/test_multistate_poe.py`'s MappedBy cases unchanged against the new import —
   they exercise aminx's field-resolution wiring, not `MappedBy`'s internals, so they should
   pass without modification if the re-export is behaviorally identical. If any fail, that is a
   signal the ported version's semantics drifted from G1's — reconcile before merging the
   pin bump, don't silently adjust aminx's tests to match.

## Acceptance criteria

- xtrax ships `MappedBy`/`resolve_mapped_by` (or equivalent naming, xtrax maintainers' call)
  behaviorally matching G1's contract: non-`MappedBy` broadcasts, `MappedBy` resolves in
  caller-supplied id order, missing id raises with the missing ids named.
- aminx's own `by="path"`-only restriction and canonical-structure-id keying convention are
  preserved as aminx-level policy, not baked into the upstream primitive.
- The migration in aminx (step 2 above) is a re-export, not a call-site rewrite — if it turns
  out not to be, the upstream shape probably diverged from what G1 actually needed and should
  be reconciled before adopting it.
