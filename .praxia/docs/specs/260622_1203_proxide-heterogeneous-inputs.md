---
title: Heterogeneous Input Resolution via Proxide URIs (aminx run/spec CLI)
issue: "#1203"
task_id: "260618_autonomous-loop"
plan_id: "260622_1203_design"
date: "2026-06-22"
status: "SPEC_DRAFT — pending adversarial spec-review (challenger/defender)"
research_note: ".praxia/docs/research/260622_1203_proxide-heterogeneous-inputs.md"
---

# Heterogeneous Input Resolution via Proxide URIs

## 1. Background

The `aminx run` CLI (`sample` / `score` / `jacobian` / `inspect`) and the matching
`spec emit-*` commands accept `--inputs`, expanded by
`_expand_inputs()` (`src/aminx/cli.py:96`). Today expansion is **local-only**:
directories and globs are filtered to `_STRUCTURE_EXTS = {".pdb", ".cif"}`
(`src/aminx/cli.py:92`); any other entry is passed through verbatim as a concrete
local path (`src/aminx/cli.py:155-158`). The resolved `list[str]` becomes
`RunSpecification.inputs` (`src/aminx/run/specs.py:182`) and flows to the runner via
`prep_protein_stream_and_model()` -> `create_protein_dataset(_loader_inputs(spec.inputs), ...)`
(`src/aminx/host/prep.py:102-103`).

Proxide (`proxide>=0.1.0a9`, `pyproject.toml:23`) exposes structural fetchers that
download remote sources to **local files** and return the path:
- `proxide.io.fetch_rcsb(pdb_id, output_dir='.', format_type='mmcif') -> str`
- `proxide.io.fetch_afdb(uniprot_id, output_dir='.', version=4) -> str`
- `proxide.io.fetch_md_cath(md_cath_id, output_dir='.') -> str`

All structural sources (local PDB/CIF, RCSB, AlphaFold, MD-CATH, HDF5) ultimately
parse to a proxide `Protein` (Atom37) via `create_protein_dataset` — **no adapter
shim is needed** (verified: `proxide.io` exports `fetch_rcsb/fetch_afdb/fetch_md_cath`
with the signatures above; fetchers are Rust-extension backed via `_proxider`,
`.venv/.../proxide/io/fetching.py:6-44`).

`#1203`'s title mentions SMILES and FASTA. **Proxide v0.1.0a9 has neither** (no
public loader; research note A.4, D.2). They are explicitly out of scope here
(section 11) — designing them now would be vaporware.

## 2. Scope

### In scope
- A **URI-scheme front-end** for `--inputs` that recognises `pdb://`, `afdb://`,
  `mdcath://`, `file://`, and bare local paths.
- An **input resolver** that runs at CLI/submit time (replacing/wrapping
  `_expand_inputs`), fetches remote sources to a **local cache dir**, and returns
  **resolved local file paths**. `RunSpecification.inputs` always holds local paths.
- A `--input-type {auto,file,pdb,afdb,mdcath}` override flag (default `auto`).
- A `--cache-dir` flag with a documented default; cache hit/reuse; clear, actionable
  errors on unresolvable accession / offline / network failure.
- Backward-compatible local-path/dir/glob behaviour, byte-for-byte unchanged.
- Tests: URI parsing, scheme dispatch, offline-mode errors, local-path regression,
  one mocked integration smoke (no network in CI).

### Out of scope (this issue)
- SMILES -> 3D and FASTA -> predicted structure (blocked on upstream proxide; section 11).
- FoldComp database integration (already handled via `foldcomp_database` kwarg;
  no `--inputs` change needed).
- Trajectory frame selection / `iterload` chunk tuning.
- Any change to runner-side parsing (`create_protein_dataset` already accepts the
  resolved local paths).

## 3. URI scheme & precedence

Accepted `--inputs` entry forms:

| Form | Example | Source | Resolver action |
| --- | --- | --- | --- |
| Bare path | `/data/1ubq.pdb`, `./mydir`, `*.cif` | local | dir/glob/file expansion (current behaviour, **unchanged**) |
| `file://` | `file:///data/1ubq.pdb` | local | strip scheme -> local path; then current behaviour |
| `pdb://<id>` | `pdb://1A3A` | RCSB | `fetch_rcsb(id, cache_dir, format_type)` -> local path |
| `afdb://<uniprot>` | `afdb://P12345` | AlphaFold | `fetch_afdb(uniprot, cache_dir)` -> local path |
| `mdcath://<id>` | `mdcath://1abcA00` | MD-CATH | `fetch_md_cath(id, cache_dir)` -> local `.h5` path |

**Scheme grammar.** A scheme is the leading `<token>://` where `<token>` matches
`^[a-z][a-z0-9+.-]*$` (RFC-3986-ish, lowercased). Detection is on the literal
prefix only; the remainder is the accession (passed verbatim to the fetcher, never
glob-expanded). A bare Windows path like `C:\x` is **not** a scheme (`:` not followed
by `//`).

**`pdb://` format suffix.** `pdb://1A3A` -> `format_type='mmcif'` (proxide default).
`pdb://1A3A.pdb` -> `format_type='pdb'`, accession `1A3A`. `pdb://1A3A.cif` ->
`format_type='mmcif'`, accession `1A3A`. Any other `.<ext>` -> error (unknown format).

**Precedence (explicit flag vs scheme):**
1. `--input-type` is `auto` (default) -> scheme detection governs per-entry; bare/`file://`
   entries are local.
2. `--input-type {file,pdb,afdb,mdcath}` -> **applies to every entry that has no
   explicit scheme**. Entries that *do* carry a scheme keep their scheme, and a
   **conflict with a non-`auto` flag is a hard error** (`typer.BadParameter`), e.g.
   `--input-type pdb` with `afdb://P12345`. Rationale: silent override of an explicit
   scheme is a footgun; fail loud. So `--input-type` only changes how *schemeless*
   tokens are dispatched (e.g. force `1A3A` to mean an RCSB id via `--input-type pdb`).
3. Bare-path default under `auto`: a schemeless token is local (current behaviour).

**Mixed inputs** (`--inputs pdb://1A3A /tmp/x.pdb afdb://P1`) are allowed; each entry
resolves independently and the resolved local paths are concatenated, dedup
preserving first-seen order (current `_expand_inputs` semantics).

## 4. Resolution architecture

```
CLI --inputs [+ --input-type, --cache-dir]
  |
  v
resolve_inputs(entries, input_type, cache_dir, *, fail_fast)   # src/aminx/cli.py (new), or io/input_resolver.py
  |   per entry:
  |     classify(entry, input_type) -> (scheme, accession|path)
  |     local  -> existing _expand_inputs path (dir/glob/file)
  |     remote -> io/proxide_fetch.py: fetch_<scheme>(accession, cache_dir) -> local path
  |               (cache hit short-circuits the fetch)
  v
list[str]  (LOCAL PATHS ONLY)  -> RunSpecification.inputs (src/aminx/run/specs.py:182)
  v
spec JSON serialization (str | list[str] only — src/aminx/run/spec_json.py:83-96)
  v
prep_protein_stream_and_model(spec)  (offline-safe; never fetches)
  v
create_protein_dataset(_loader_inputs(spec.inputs), ...)  (src/aminx/host/prep.py:102)
```

**Components**

| Component | Kind | Role |
| --- | --- | --- |
| `io/uri.py` (new) | new | pure scheme parser: `parse_input_uri(entry) -> ParsedInput(scheme, accession, fmt)`; no I/O |
| `io/proxide_fetch.py` (new) | new | thin wrappers: `fetch_pdb/fetch_afdb/fetch_mdcath(accession, cache_dir) -> Path`; cache-aware; error-mapping |
| `cli._expand_inputs` | changed | becomes `resolve_inputs`; keeps local dir/glob/file branch verbatim, adds scheme dispatch |
| `cli` run/spec subcommands | changed | add `--input-type`, `--cache-dir` options; pass to resolver |
| `host/prep.py` | reused | unchanged — already consumes local paths; offline-safe by construction |

**Cache dir.** Flag `--cache-dir <path>` (Path). Default resolution order:
1. `--cache-dir` if given.
2. `AMINX_CACHE_DIR` env var if set.
3. `b.cache_path / "inputs"` if `--cache-path` is set (reuse existing flag,
   `src/aminx/cli.py:283`).
4. `~/.cache/aminx/inputs/` (XDG-respecting: `$XDG_CACHE_HOME/aminx/inputs` if set).

The cache dir is created (`mkdir(parents=True, exist_ok=True)`) before any fetch.

**Cache hit / reuse.** Cache key = deterministic filename the proxide fetcher writes
for `(scheme, accession, format)`. The resolver checks for an existing non-empty file
matching the expected name **before** calling the fetcher; on hit, return the cached
path and skip the network entirely. Because proxide's fetcher itself writes into
`output_dir=cache_dir`, the resolver computes the expected path; if present and
non-empty it short-circuits, otherwise it fetches then asserts the returned path is
inside `cache_dir`. (Cache invalidation: honour the existing `--overwrite-cache`
flag, `src/aminx/cli.py` / `specs.py:218` — when set, delete the cached file first
and re-fetch.)

**Error handling (all -> clear, actionable, non-zero exit):**
- **Offline / network failure:** proxide fetcher raises (native `_proxider` exception).
  Wrap broadly (`except Exception as exc`) and re-raise as a typed
  `InputResolutionError` with: the URI, the resolved cache dir, and the hint
  "remote fetches require network access at CLI/submit time; resolve on a connected
  host, then submit with the cached local path (cluster compute nodes are offline)".
- **Unresolvable accession (404 / not found):** same path; message includes the
  accession and source ("RCSB returned no structure for 'pdb://1A3X'").
- **Malformed URI / unknown scheme:** `typer.BadParameter` at classify time, before
  any I/O.
- **`--input-type` vs scheme conflict:** `typer.BadParameter` (section 3 precedence rule 2).
- **`fail_fast` semantics:** preserve current `_expand_inputs(fail_fast=...)` contract
  (`src/aminx/cli.py:96`) — in `fail_fast=False`, a failed *remote* fetch warns and
  skips (consistent with current dir/glob skip-on-empty); in `fail_fast=True`, exit 1
  on first failure. Offline detection is reported regardless.

## 5. Serialization impact

**No schema break.** `RunSpecification.inputs` stays `Sequence[str | TextIO] | str | TextIO`
(`src/aminx/run/specs.py:182`); the resolver guarantees it holds **local path strings**.
The JSON encoder requires exactly `str | list[str]` (`_ensure_jsonable_inputs`,
`src/aminx/run/spec_json.py:83-96`) and the decoder coerces the same
(`spec_json.py:168-172`) — resolving to local paths is therefore **required**, not
merely convenient.

**Portable-JSON (RS-8) cross-check — no regression.** `run_spec_portable_to_dict`
(`src/aminx/run/run_spec_portable_json.py:174-211`) serializes only the `RunSpec`
sub-configs (`io`, `multistate`, `resource`, `precision`). **It never touches
`inputs`.** This feature adds no field to `RunSpec` and changes no sub-config, so the
v2 portable wire format and its guards are untouched. Acceptance gate G7 asserts the
existing `tests/run/test_run_spec_portable_json.py` suite stays green.

**Optional provenance (`inputs_metadata`) — DEFERRED, not in this spec.** A source
trace (`{resolved_path: {uri, source, fetched_at}}`) is *desirable* for reproducibility
but introduces a new serializable field, a JSON schema addition, and a portable-format
question. To keep this slice minimal and the portable guard untouched, provenance is
**out of scope** here; section 11 records it as the natural follow-up.
Justification: the resolved local paths are already in `inputs` and the cache dir is
deterministic, so re-resolution is reproducible without a new field for v1.

## 6. Dependency gate

- `proxide>=0.1.0a9` — already a hard dep (`pyproject.toml:23`). Provides all three
  fetchers. **No change.**
- `h5py>=3.11` — already a hard dep (`pyproject.toml:17`). Covers `mdcath://` (`.h5`)
  and HDF5 trajectory parsing. **No change.**
- `requests` — proxide's Rust extension owns the network call (`_proxider`); aminx
  does not import `requests` directly. **No new declaration.** (It is transitively
  present, but the resolver must not depend on it — fetch errors come from proxide.)
- **Net new aminx deps: none.** The feature is pure-Python orchestration over existing
  deps.

## 7. Acceptance gates

Project gates (all must pass; `uv run`): `uv run pytest`, `uv run ty check`,
`uv run ruff check .`, `uv run ruff format --check .`. **No-network rule: tests MUST
mock proxide fetchers (`monkeypatch` `fetch_rcsb/afdb/md_cath`); never hit RCSB/AFDB
in CI.**

| Gate | Assertion |
| --- | --- |
| G1 URI parse | `parse_input_uri` unit table: bare/`file://`/`pdb://`/`afdb://`/`mdcath://`, `pdb://1A3A.pdb` -> fmt=`pdb`, Windows `C:\x` -> local, unknown scheme -> error. |
| G2 dispatch | `--input-type` x scheme matrix: `auto` uses scheme; non-`auto` applies to schemeless tokens; flag-vs-scheme conflict raises `typer.BadParameter`. |
| G3 fetch wrappers | `proxide_fetch` returns a `Path` inside `cache_dir`; mocked fetcher called once on miss, **zero** times on cache hit; `--overwrite-cache` forces re-fetch. |
| G4 offline error | mocked fetcher raising -> `InputResolutionError` whose message names the URI, the cache dir, and the offline/submit-time hint; `fail_fast=True` exits 1, `fail_fast=False` warns+skips. |
| G5 local regression | `tests/data/*.pdb`/`*.cif`, dirs, globs resolve byte-identically to pre-change `_expand_inputs` (golden list); `file://` to a local fixture resolves to the same path. |
| G6 integration smoke | one end-to-end: `--inputs pdb://TEST` with `fetch_rcsb` monkeypatched to copy `tests/data/1ubq.pdb` into `cache_dir`; resulting spec's `inputs` is `[<cache_dir>/...pdb]`; spec round-trips through `run_specification_to_json`/`...from_json`. |
| G7 portable no-regress | `tests/run/test_run_spec_portable_json.py` unchanged and green. |
| G8 spec-emit parity | all 8 `_expand_inputs` call sites (`cli.py:465,559,639,715,921,1009,1083,1153`) route through the resolver; `spec emit-*` JSON for a bare-path input is unchanged vs baseline. |

## 8. Task DAG

Spine lands a vertical slice (URI parse -> `pdb://` fetch -> cache -> wire into one
call site -> tests) before breadth (`afdb`/`mdcath`, all call sites, integration).

| ID | Title | Scope (files) | depends_on | Difficulty | Category | Per-task gates |
| --- | --- | --- | --- | --- | --- | --- |
| T1 | URI parser | `src/aminx/io/uri.py` (new); `tests/io/test_uri.py` (new) | — | quick | impl | G1 |
| T2 | Proxide fetch wrappers + cache | `src/aminx/io/proxide_fetch.py` (new); `tests/io/test_proxide_fetch.py` (new) | T1 | standard | impl | G3, G4 |
| T3 | Resolver core (`resolve_inputs`, replaces `_expand_inputs`) | `src/aminx/cli.py:92-164` | T1, T2 | standard | impl | G2, G5 (local-regression golden) |
| T4 | CLI flags `--input-type`, `--cache-dir` on run + spec groups | `src/aminx/cli.py` (`_RunBase`/`_run_base` ~208-249; subcommand sigs) | T3 | standard | impl | G2 (dispatch via flag) |
| T5 | Wire resolver into all 8 call sites | `src/aminx/cli.py:465,559,639,715,921,1009,1083,1153` | T3, T4 | quick | impl | G8 |
| T6 | `afdb://` + `mdcath://` breadth (parser cases + wrappers) | `io/uri.py`, `io/proxide_fetch.py`; tests | T2, T3 | quick | impl | G1, G3 for afdb/mdcath |
| T7 | Offline/error-path hardening + `InputResolutionError` type | `io/proxide_fetch.py`, `src/aminx/cli.py` | T2, T3 | quick | impl | G4 |
| T8 | Integration smoke + portable-no-regress assertion | `tests/cli/test_inputs_integration.py` (new); `tests/run/test_run_spec_portable_json.py` (touch only to assert) | T5, T6, T7 | standard | test | G6, G7 |
| T9 | Docs: CLI help text + `using-aminx` skill `--inputs` examples; URI table | `src/aminx/cli.py` help strings; skill docs | T4, T5 | quick | docs | help text renders; examples match grammar section 3 |

**Parallelism**
- **Cluster A (serial spine):** T1 -> T2 -> T3 -> T4 -> T5.
- **Post-T3 parallel:** T6 (`afdb`/`mdcath`) and T7 (error hardening) can run
  concurrently once T3 lands (both extend the resolver without colliding: T6 touches
  parser/wrapper *cases*, T7 touches *error mapping*; coordinate one merge on
  `proxide_fetch.py`).
- **T9 (docs)** can start in parallel with T6/T7 after T4 fixes the flag surface.
- **T8** is the join — depends on T5, T6, T7.

## 9. Risk table

| Risk | Severity | Likelihood | Mitigation |
| --- | --- | --- | --- |
| Offline cluster: a `*://` URI reaches the runner and fetches on a compute node | high | low (resolved at CLI by design) | Resolver runs at CLI/submit time only; `spec.inputs` is local paths post-resolve; prep.py never fetches. G4 + a runner-side assert that inputs contain no `://` (defensive). |
| Ambiguous scheme (`C:\path`, `s3://`, double `//`) | med | med | Strict grammar section 3 (`^[a-z][a-z0-9+.-]*://`); unknown scheme -> `typer.BadParameter` (no silent local fallback). G1 covers Windows-path + unknown-scheme. |
| Accession typo -> 404 / hang | med | med | proxide raises on 404; resolver maps to `InputResolutionError` with the accession; `fail_fast` honoured. G4. (Format pre-validation, e.g. 4-char PDB id, is optional follow-up, not blocking.) |
| Cache staleness (RCSB updates a structure) | low | low | Deterministic cached filename; `--overwrite-cache` forces re-fetch (G3). Document that the cache is content-addressed by accession, not versioned. |
| Proxide API drift (signature/return change) | med | low | Wrappers isolate all proxide calls in `io/proxide_fetch.py`; signatures pinned by `proxide>=0.1.0a9`; G3 mocks the exact signature. A single file to update on drift. |
| Partial failure in multi-input batch | med | med | Per-entry resolution; `fail_fast=False` warns+skips a bad entry (matches current dir/glob behaviour); `fail_fast=True` exits on first. Resolved list dedups first-seen. G4/G5. |
| `--input-type` silently overrides an explicit scheme | low | med | Made a **hard error** (section 3 rule 2), not a silent override. G2. |

## 10. Backward compatibility

- `_expand_inputs`'s local branch (dir/glob/concrete-file, `cli.py:122-158`) is
  preserved verbatim inside the resolver; G5 pins it against a golden list from the
  current implementation.
- `file://` is additive (new accepted form); bare paths behave exactly as today.
- No spec field added/removed; JSON wire format unchanged (section 5). Existing emitted
  specs deserialize unchanged.
- All 8 call sites keep the `fail_fast=inputs_fail_fast` contract.

## 11. Out of scope / future

- **SMILES / FASTA inputs** — no proxide v0.1.0a9 loader (research A.4, D.2).
  Blocked on upstream proxide adding `smiles://` (rdkit) and `fasta://`
  (ESMFold/OmegaFold) support. Revisit when proxide >=0.2 ships them.
- **`inputs_metadata` provenance field** — deferred (section 5). Natural follow-up
  once a v3 portable wire format is on the table; would carry
  `{resolved_path -> uri/source/fetched_at}`.
- **PDB-id format pre-validation** (`--strict-accessions`) — optional UX nicety;
  proxide's 404 path already yields a clear error.
- **Trajectory frame selection** for `mdcath://`/HDF5 — separate concern.
