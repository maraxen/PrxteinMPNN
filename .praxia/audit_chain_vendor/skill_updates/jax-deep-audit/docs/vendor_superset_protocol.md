# U5 protocol — vendor/reference feature-surface superset

## The question this checker answers

For a target codebase and an external reference implementation ("vendor") that overlaps in
domain, does the target's declared configuration surface contain every concept the vendor
exposes? This is a **structural coverage** question, answered by field/flag presence — not a
reachability question (that's U1, scoped entirely within the target codebase).

Do not conflate the two. A field can be U5-PASS (the concept exists in the target) and
U1-FAIL (it's declared but never read by any consumer) simultaneously. Treat them as
independent axes in any coverage matrix that reports both.

## Why AST for the vendor side, reflection for the target side

Vendor CLIs are typically `argparse`-based scripts, not reflective dataclasses — there is no
`fields()` to call, so `extract_argparse_flags` walks the AST for
`<argparser-like-object>.add_argument(...)` calls. The target side, by contrast, is assumed to
be safely importable Python with real dataclasses, so `check_u5` uses `dataclasses.fields()`
reflection rather than re-deriving `AnnAssign` structure via AST — reflection is more robust
whenever it's available; AST is the fallback for code that has no reflective structure at all
(a CLI's flag surface never does).

If your vendor doesn't use argparse (a config-dataclass-based CLI, a YAML schema, etc.), write
a sibling extractor with the same output shape (`{flag, type, default, help, source_line}`) —
`check_u5` only needs that shape, not argparse specifically.

## The alias map is the auditor's judgment call, not the checker's

`check_u5` never guesses which vendor flag maps to which target field. The caller supplies
`alias_map: {vendor_flag_name: {"target_fields": [...], "category": str, "notes": str}}`. This
is deliberate: cross-codebase concept mapping requires domain judgment a static checker cannot
supply, and a wrong-but-confident auto-generated mapping is worse than an honest human-curated
one. An alias_map entry with an empty `target_fields` list still counts — it asserts "the
auditor looked, and there is no target equivalent" — so a real gap doesn't silently disappear
from the matrix; it appears with a `notes` field explaining why.

## Batch/multi-item ergonomics is its own category, not a binary present/absent

A real-world discovery (260826_chain-selection-vendor-superset-audit): a vendor flag and a
target field can be a legitimate 1:1 semantic match (`PASS` under U5) while still differing in
an ergonomically significant way — e.g. the vendor accepts a JSON mapping keyed by input path
for batch operation (`{"/path/to/item": "value"}`), while the target field only accepts a
single value per call. Don't force this into PASS or FAIL; add a `batch_ergonomics_gap: true`
flag (or equivalent) to the verdict dict alongside the PASS, so the coverage report doesn't
bury a real usability gap inside an otherwise-green cell. See the aminx audit's
`fixed_residues_multi`/`redesigned_residues_multi` -> `fixed_mask` finding for a worked example.

## What "vendor pin" bugs taught this checker

Before running `extract_argparse_flags` against a live vendor clone, verify which pin is
actually load-bearing. A project can have the SAME stale commit SHA appear in multiple places
with very different consequences:

- A **functional pin** (code that clones/checks out that commit at runtime) — stale here means
  the mechanism is actually broken; fix it and re-verify against live upstream.
- A **documentation/provenance string** (a comment or JSON field recording what a *frozen,
  already-committed* fixture or ported code section was derived from) — stale here is a
  cosmetic inaccuracy, not a functional break. Do NOT "fix" these by rewriting history to match
  a newer commit; that would misrepresent what was actually used to produce the frozen
  artifact. Fix the documentation to be internally consistent (e.g. point at the correct
  functional pin's value) without implying the frozen artifact was re-derived.

Check both categories separately before claiming a pin is stale; grep will find every
occurrence but won't tell you which ones are load-bearing.
