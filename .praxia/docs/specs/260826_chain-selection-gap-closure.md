# Spec: closing the chain-selection vendor-superset gaps

**Task:** 260826_chain-selection-vendor-superset-audit · **Status:** DRAFT, pending review · **Scope:** aminx only; G1's `MappedBy`/`resolve_mapped_by` is a minimal aminx-local implementation, with the generic xtrax port tracked as deferred debt (see bottom section), not built here.

Source: Axis B round 1 (`vendor_superset_coverage.json`, `vendor_superset_summary.md`) of the
260826_chain-selection-vendor-superset-audit sprint. Four gaps identified; this spec proposes
an implementation for each, ranked by what's actually shovel-ready vs. blocked on a prior
question.

## G1 — generic batch input mapping ("json-mapped-by-X")

**Gap**: `fixed_residues_multi`/`redesigned_residues_multi` (vendor) let a caller supply a
per-structure value via a JSON object keyed by PDB path: `{"/path/to/pdb": "A12 A13 A14 B2
B25"}`. aminx's `fixed_mask` is a single per-call value — no batch-keyed variant exists.

**Directive from discussion**: json-mapped-by-pdb-path is one strategy among several
plausible ones (keyed by structure index, by a caller-assigned structure ID, by chain_id,
etc.). Build the mechanism generically — `json-mapped-by-X` where `X` names a *declared input
field* — but implement only `X = "path"` now. Do not build the other `X` values speculatively.

### Design

New generic wrapper type, `src/aminx/run/batch_mapping.py`:

```python
_SUPPORTED_MAPPED_BY: frozenset[str] = frozenset({"path"})

@dataclass(frozen=True, slots=True)
class MappedBy(Generic[T]):
    """A per-structure value supplied as {key: value}, keyed by a declared input field.

    Only by="path" is implemented today. `by` is a real field (not a Literal["path"]) so the
    shape survives when a second X is added -- the resolver below is where support is
    actually gated, not the type.
    """
    by: str
    mapping: dict[str, T]

    def __post_init__(self) -> None:
        if self.by not in _SUPPORTED_MAPPED_BY:
            msg = (
                f"MappedBy(by={self.by!r}) is not supported yet -- only "
                f"{sorted(_SUPPORTED_MAPPED_BY)} is implemented. Construct your batch mapping "
                f"keyed by input path, or file the new `by` value as a feature request rather "
                f"than working around this check."
            )
            raise NotImplementedError(msg)


def resolve_mapped_by(
    value: T | MappedBy[T] | None, *, structure_paths: Sequence[str], field_name: str,
) -> list[T | None]:
    """Resolve a possibly-MappedBy field into one value per structure, in `structure_paths` order.

    A non-MappedBy value broadcasts to every structure (existing single-value behavior,
    unchanged). A MappedBy value is resolved per path; a structure path absent from the
    mapping raises -- silently defaulting an omitted structure to None/unset is exactly the
    kind of quiet wrong-answer this audit sprint exists to catch (see FA2/FA3, and the
    fixed_tokens Alanine-collapse guard in _sampling_helper.py).
    """
    if value is None or not isinstance(value, MappedBy):
        return [value] * len(structure_paths)
    missing = [p for p in structure_paths if p not in value.mapping]
    if missing:
        msg = (
            f"{field_name}: MappedBy(by='path') is missing an entry for "
            f"{len(missing)} of {len(structure_paths)} structures in this batch: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}. Every structure path in "
            f"spec.inputs must have a mapping entry -- there is no implicit fallback."
        )
        raise ValueError(msg)
    return [value.mapping[p] for p in structure_paths]
```

**Field-level change**: `RunSpecification.fixed_mask` type widens to
`ArrayLike | MappedBy[ArrayLike] | None = None`. Same for `fixed_tokens` (they're a pair — see
`_prepare_fixed_controls`'s existing guard; a `MappedBy` `fixed_mask` without a matching
`MappedBy` `fixed_tokens`, or vice versa, must raise the same way the existing scalar guard
does, not silently zip mismatched shapes).

**Resolution site**: `host/_sampling_helper.py`'s existing `fixed_mask`/`fixed_tokens`
handling (around the `_prepare_fixed_controls` guard, lines ~655-698 per Axis A's F_A1
investigation) is the only place that currently reads these fields for the sample path. Call
`resolve_mapped_by` there, per-batch, before the existing broadcast/validation logic — the
per-structure value it produces should look identical to what a caller passing a plain array
already produces, so the existing downstream logic doesn't need to know `MappedBy` exists at
all.

**Axis A implication**: `score()`/`jacobian()` currently raise NotImplementedError on any
nonzero `fixed_mask` at all (FA2/FA3 fix, this sprint). A `MappedBy`-typed `fixed_mask` must
hit that same guard — `bool(jnp.any(...))` on a `MappedBy` instance will raise on
`isinstance` mismatch, so the guard needs `isinstance(spec.fixed_mask, MappedBy) or
bool(jnp.any(jnp.asarray(spec.fixed_mask)))` before this lands, or `MappedBy` support silently
bypasses FA2/FA3's guard for exactly the surfaces that most need it.

**Test plan**: golden test with 2 structures, distinct `fixed_mask`/`fixed_tokens` per
structure via `MappedBy(by="path", mapping={...})`; assert `sample()` forces the correct
distinct identity per structure (same technique as `F_A1_fixed_mask_sample_probe.py`, made
permanent). Negative tests: missing-path raises `ValueError`; `by="chain_id"` raises
`NotImplementedError`; `MappedBy` on `score()`/`jacobian()` still hits FA2/FA3's guard.

## G2 — `chains_to_design` (chain-letter design/fix selector)

**Gap**: vendor's `--chains_to_design "A,B"` lets a caller name chains by letter; everything
else is fixed. aminx has no equivalent — the orphaned `chain_mask_fixed` (#1881) is
residue-index-only.

### Design

New field on `SamplingSpecification` (not the `RunSpecification` base — see open question
below): `chains_to_design: Sequence[str] | None = None`.

Resolution requires the parsed structure's `chain_index` (per-residue chain assignment), which
is not available until prep-time — cannot be resolved at spec-construction. Resolve in
`host/prep.py` (alongside the existing `chain_id` consumption at line 96) or
`_sampling_helper.py`, deriving a `fixed_mask` where every residue whose chain letter is NOT in
`chains_to_design` is marked fixed.

**Open question (owner decision, not resolved by this spec)**: precedence when both
`chains_to_design` and an explicit `fixed_mask` are set. Options: (a) mutually exclusive, raise
if both set; (b) `chains_to_design` computes a base mask, explicit `fixed_mask` further
restricts it (union of fixed positions); (c) explicit `fixed_mask` wins outright, ignoring
`chains_to_design`. Recommend (a) — matches this sprint's "loud over guessed" precedent, and
vendor's own CLI doesn't document a combined-flags interaction either — but the sprint's
methodology is deliberately not the place to adjudicate this; needs a decision before
implementation starts, not during code review.

**Scope note**: per Axis A's own precedent, if `chains_to_design` lands on `sample()` only
(matching where `fixed_mask` is actually differential-probe-confirmed to work today), it
should get the SAME FA2/FA3-style loud guard on `score()`/`jacobian()`/`inspect()` rather than
silently accepting and ignoring the field there too.

## G3/G4 — `symmetry_weights` and `homo_oligomer`

**BLOCKED, not spec'd here.** Both map conceptually to aminx's `tie_group_map`/
`tied_positions`/`pass_mode` family (per `vendor_superset_coverage.json`'s "present" mapping
for `symmetry_residues`), but that family's reachability across `sample`/`score`/`jacobian`/
`inspect` was explicitly flagged UNTRACED in `seed_findings.md` and never differential-probed
the way `fixed_mask` was (FA1-FA3). Designing new fields (`symmetry_weights`, a
`homo_oligomer` convenience preset) on top of an unverified foundation risks building on a
field that turns out to be silently inert somewhere, the same shape this whole sprint exists
to catch.

**Prerequisite before this can be spec'd for real**: run the FA1-style differential probes
against `tie_group_map`/`tied_positions` across all 4 runner surfaces (same technique, new
target). That is Axis A follow-up work, tracked in `seed_findings.md`, not part of this spec.

## Tech debt (deferred, out of scope for this spec): generic mapped-json in xtrax

`MappedBy`/`resolve_mapped_by` (G1) is a genuinely generic primitive — "a per-item batch value
supplied as JSON keyed by a declared input field" is not aminx-specific, and belongs upstream
in xtrax (the batching/run-spec framework aminx is built on) rather than reimplemented per
downstream project. G1's aminx-local implementation above is intentionally the *minimal*
version needed now; do not block G1 on the xtrax port, and do not let the xtrax port silently
expand G1's scope (e.g. implementing `by="chain_id"` "while we're in there").

Filed as debt (see debt entry logged alongside this spec) rather than implemented here.
