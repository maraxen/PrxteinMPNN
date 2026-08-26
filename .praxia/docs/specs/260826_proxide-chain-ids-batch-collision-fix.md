# Spec: fix proxide's batched `chain_ids` collision (unblocks G2)

**Task:** 260826_chain-selection-vendor-superset-audit · **Status:** DRAFT, pending review ·
**Scope:** proxide (external dependency, consumed by aminx via PyPI — no local editable
checkout in this workspace as of this writing; `pyproject.toml` documents the
`uv.toml`/`[sources]` override for live co-development against a sibling `../proxide` checkout
if/when one exists). This spec is written to be actionable by whoever picks it up, in
proxide's own repo, not implemented here.

## Why this exists

Finding `FG2` (`.praxia/audit_chain_vendor/findings.jsonl`, evidence in
`evidence/F_G2_chain_ids_batch_collision_probe.py`/`_report.json`) confirmed via a real
differential probe that batching two structures with the same chain COUNT but different chain
LETTERS silently corrupts `Protein.chain_ids` on the batched result: row 1's real chain `'C'`
resolves as `'A'`. No exception, no warning, valid-shaped output. This blocks G2
(`chains_to_design`, `.praxia/docs/specs/260826_chain-selection-gap-closure.md`), which needs
exactly this data (`chain_index` + `chain_ids` → per-residue chain letter) to work correctly.

## Root cause (diagnosed precisely, not just reproduced)

`proxide/ops/transforms.py::_stack_padded_proteins` (called by `pad_and_collate_proteins`,
which `create_protein_dataset` uses to batch multiple parsed structures):

```python
def _stack_padded_proteins(padded_proteins: list[Protein]) -> Protein:
  def stack_fn(*arrays: np.ndarray | None) -> np.ndarray | None:
    non_none = [a for a in arrays if a is not None]
    if not non_none:
      return None
    first = non_none[0]
    if not hasattr(first, "shape") or first.ndim == 0:
      return first                      # <-- the bug: silently keeps only `first`
    if not all(hasattr(a, "shape") and a.shape == first.shape for a in non_none):
      return None
    return np.stack(non_none, axis=0)

  return jax.tree_util.tree_map(stack_fn, *padded_proteins)
```

`Protein.chain_ids: Any | None = None` (`proxide/core/containers.py:115`) is populated as a
plain Python `list[str]` (e.g. `['A', 'B']`) — confirmed directly:
`create_protein_dataset([...]).chain_ids == ['A', 'B']` for a real 2-chain structure.

**`list` is a JAX pytree container by default.** `jax.tree_util.tree_map(stack_fn,
*padded_proteins)` therefore does NOT call `stack_fn` once per `Protein` with each one's whole
`chain_ids` list — it recurses *into* each list and calls `stack_fn` once per **matched
position**, on the individual chain-letter **strings** at that position across all the
`Protein`s being combined:

- **Same length** (2 chains + 2 chains): position 0 gets `stack_fn('A', 'C')`, position 1 gets
  `stack_fn('B', 'D')`. Strings have no `.shape` attribute, so `stack_fn`'s fallback branch
  (`if not hasattr(first, "shape")...: return first`) fires and silently returns structure 0's
  string at every position — exactly the observed corruption: `chain_ids == ['A', 'B']` for
  BOTH rows, discarding structure 1's real `['C', 'D']` with no signal.
- **Different length** (2 chains + 1 chain): the two `chain_ids` lists have different pytree
  structure (different lengths), so `tree_map` itself raises `ValueError: pytree structure
  error: different lengths of list at key path tree.chain_ids` before any stacking happens —
  a crash, not a silent wrong answer, but still blocks any real batch mixing chain counts.

This is a structural mismatch between what `chain_ids` conceptually IS (one variable-length,
per-structure list of strings — genuinely ragged data with no natural "stack" operation) and
how `_stack_padded_proteins` treats every field (uniformly, via generic `tree_map` + `np.stack`,
which assumes every leaf is either a fixed-shape array or an already-identical scalar). No
other `Any | None` field in `Protein` is a `list` of variable per-structure length in the same
way — most (`source`, `format`, etc.) are plain scalars or `None`, which `tree_map` treats as a
single leaf, not a container, so they don't hit this failure mode.

## Candidate fixes (ranked; not mutually exclusive with parallel aminx-side mitigation)

### Fix A — special-case `chain_ids` assembly in `_stack_padded_proteins` (recommended, proxide-side)

Before calling `jax.tree_util.tree_map`, pop `chain_ids` off each `Protein` in
`padded_proteins`, build the batched result as an explicit `list[list[str]]` — one entry per
row, each row's own real chain-letter vocabulary, no stacking attempted — and set it on the
final batched `Protein` after `tree_map` runs on everything else. This preserves per-row
identity exactly, requires no `Protein` schema/type change, and generalizes to any other
per-structure ragged metadata field discovered later (grep proxide's `Protein` fields for other
`Any | None` fields populated as variable-length lists — `chain_ids` may not be the only one).

Downstream consumers that currently read `batched_protein.chain_ids` assuming a single flat
list (if any exist in proxide or other consumers) would need to switch to
`batched_protein.chain_ids[row_index]`. Grep proxide's own tree for `.chain_ids` reads before
landing this to catch any other silent-collision consumer.

### Fix B — mark `chain_ids` `pytree_node=False` at the field level

`Protein` already uses this pattern for at least one field (`coulomb14scale: Any | None =
struct.field(default=None, pytree_node=False)`, `proxide/core/containers.py:148`). Marking
`chain_ids` this way would make it aux_data (structural metadata), not a mapped leaf — but
`tree_map` across multiple pytrees with *different* aux_data at the same field still needs a
defined merge policy (JAX doesn't auto-resolve differing aux_data across combined trees), so
this alone doesn't solve the problem — it would need to be paired with the same explicit
list-of-lists assembly as Fix A, just triggered by a different mechanism. Slightly more
invasive (changes `Protein`'s pytree registration semantics) for no clear benefit over Fix A;
listed for completeness, not recommended as primary.

### Fix C — aminx-side workaround (only if Fix A/B can't land promptly)

If proxide's fix timeline doesn't line up with when G2 needs to ship, aminx could re-derive
correct per-row `chain_ids` in `host/prep.py` by parsing each structure's chain letters
independently (bypassing the collated `Protein.chain_ids` entirely) BEFORE calling
`create_protein_dataset`, and threading the correct per-row `list[str]` alongside the batch by
structure-id (the same `_canonical_structure_ids_for_spec` convention G1 uses). This duplicates
work proxide's parser already does once, and only fixes aminx's own consumption of the field
(any other proxide-dependent code with the same bug stays broken) — a stopgap, not a real fix.
Only worth doing if G2 is time-critical and Fix A is stalled upstream.

## Acceptance criteria (whichever fix lands)

Re-run `evidence/F_G2_chain_ids_batch_collision_probe.py`'s "same chain COUNT, different
LETTERS" arm against the fixed proxide version — `row1_resolved_first_chain_letter` must equal
`row1_expected_first_chain_letter` (`'C'`, not `'A'`). Do not close this without that probe
passing for real; a code-read confirming the fix "looks right" is not sufficient (this whole
finding exists because a code-read of `_prepare_fixed_controls` similarly missed a real bug
until differential-probed).

## Relationship to G2

Once this lands and is re-verified, `chains_to_design`
(`.praxia/docs/specs/260826_chain-selection-gap-closure.md`, G2 section) can proceed as
originally spec'd — no other part of that design depends on this fix beyond needing correct
`chain_ids`/`chain_index` data to resolve chain letters from.
