---
title: Heterogeneous Input Resolution via Proxide URIs (aminx run/spec CLI)
issue: "#1203"
task_id: "260618_autonomous-loop"
plan_id: "260622_1203_design"
date: "2026-06-22"
status: "SPEC_RECONCILED — adversarial review resolved (challenger a4982fb / defender a98d4df); 2 blockers + 4 major + 5 minor addressed"
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

**Grammar edge cases (m1):**
- **Empty accession** (`pdb://`, `afdb://`, `mdcath://` with nothing after `//`) ->
  `typer.BadParameter` ("scheme 'pdb://' has no accession").
- **`file://host/path`** -> the authority/host segment is **stripped**; the resolver
  resolves `/path` as a local path (chosen behaviour: strip-to-path, not error). So
  `file://localhost/data/x.pdb` and `file:///data/x.pdb` both resolve to `/data/x.pdb`.
- **Known grammar, no fetcher** (`s3://...`, `http://...`, `https://...`, or any
  scheme that matches the grammar but is not one of `file`/`pdb`/`afdb`/`mdcath`) ->
  `typer.BadParameter` ("no fetcher for scheme 's3://'"). There is **NO silent local
  fallback** — a `scheme://` that aminx cannot resolve fails loud.

**`pdb://` format suffix.** `pdb://1A3A` -> `format_type='mmcif'` (proxide default).
`pdb://1A3A.pdb` -> `format_type='pdb'`, accession `1A3A`. `pdb://1A3A.cif` ->
`format_type='mmcif'`, accession `1A3A`. Any other `.<ext>` -> error (unknown format).

**Precedence (explicit flag vs scheme):**
1. `--input-type` is `auto` (default) -> scheme detection governs per-entry; bare/`file://`
   entries are local.
2. **`--input-type file` is the "force all local" escape hatch — it SUPPRESSES scheme
   detection entirely.** Every token is treated as a literal local path (dir/glob/file
   expansion), even one that looks like `pdb://1A3A` (which becomes a literal path
   string). No fetcher is ever called under `--input-type file`. This is the
   intentional override for users whose local filenames collide with the scheme
   grammar.
3. `--input-type {pdb,afdb,mdcath}` (the non-`file`, non-`auto` values) -> **applies
   only to entries that have no explicit scheme**. Entries that *do* carry a scheme
   keep their scheme, and a **conflict with the flag is a hard error**
   (`typer.BadParameter`), e.g. `--input-type pdb` with `afdb://P12345`. Rationale:
   silent override of an explicit scheme is a footgun; fail loud. So these flags only
   change how *schemeless* tokens are dispatched (e.g. force `1A3A` to mean an RCSB id
   via `--input-type pdb`).
4. Bare-path default under `auto`: a schemeless token is local (current behaviour).

**Mixed inputs** (`--inputs pdb://1A3A /tmp/x.pdb afdb://P1`) are allowed; each entry
resolves independently and the resolved local paths are concatenated, dedup preserving
first-seen order (current `_expand_inputs` semantics).

**M2 — pre-resolution dedup on `(scheme, accession, fmt)`.** Before any fetch, the
*entry list itself* is deduplicated by the classified key `(scheme, accession, fmt)`
(first-seen order preserved). This prevents a **double fetch on a cold cache**: e.g.
`--inputs pdb://1A3A pdb://1A3A` (or `pdb://1A3A` repeated across argv) would otherwise
issue two concurrent fetches into the same cold subdir before either populates it. The
dedup is **in addition to** the existing post-resolution dedup of resolved local paths
(which catches the case where two distinct accessions happen to resolve to the same
file). Local (bare/`file://`) entries dedup by their normalised path as today.

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
| `run/specs.py::_loader_inputs` | changed | **runner-side unresolved-URI guard** (B2) — the single chokepoint all runner paths pass through |

**Runner-side unresolved-URI guard (B2 — first-class, located, gated).** Offline safety
must not be convention-only. `_loader_inputs` (`src/aminx/run/specs.py:84`) is the
**single chokepoint** that `prep.py:102-103` calls to materialise `spec.inputs` for the
runner — and it is reached by **every** path that builds a `RunSpecification`, including
the two bypass paths the challenger verified do **not** pass through the CLI resolver:

1. `run_specification_from_json` (`src/aminx/cli.py:1247`, also `:1259`) — a spec JSON
   loaded directly, never re-run through `resolve_inputs`.
2. Manifest rows in `host/campaign.py` (`~840`, `sampling_spec` payloads) — campaign
   specs reconstructed from a manifest.

The guard: in `_loader_inputs`, before returning, scan each input **string** (skip
`TextIO` handles) and raise a typed `InputResolutionError` if any matches the §3 scheme
grammar `^[a-z][a-z0-9+.-]*://`. The message names the offending token and states:
*"inputs must be resolved to local paths before the runner; re-run resolution on a
connected host."* This converts the §9 "defensive runner-side assert" into an enforced,
located, gated requirement covering **both** bypass paths in one place (gate G4b on
task **T7b**).

**Cache dir.** Flag `--cache-dir <path>` (Path). Default resolution order:
1. `--cache-dir` if given.
2. `AMINX_CACHE_DIR` env var if set.
3. `~/.cache/aminx/inputs/` (XDG-respecting: `$XDG_CACHE_HOME/aminx/inputs` if set).

**The input cache dir is deliberately decoupled from `--cache-path`.** The existing
`--cache-path` flag (`src/aminx/cli.py:283`) feeds the **JAX compilation cache**
(`jax_compilation_cache_dir`, `src/aminx/host/prep.py:129-130`); reusing
`cache_path / "inputs"` for fetched structures would collide that JAX cache tree with
downloaded artifacts. So the input cache has its own independent precedence (above),
defaulting to the XDG cache home — never derived from `--cache-path`.

The cache dir is created (`mkdir(parents=True, exist_ok=True)`) before any fetch.
**m2 (mkdir/write error mapping):** the `mkdir` and the post-fetch write are wrapped so
that `PermissionError`, a full-disk `OSError`, or any I/O failure surfaces as a typed
`InputResolutionError` naming the cache dir and the OS error ("cannot write input cache
at <dir>: <errno> — check permissions / free space"). Per-accession subdirs (B1)
isolate concurrent writes: two processes resolving different accessions touch disjoint
subdirs, so the only contended path is the shared parent, created idempotently with
`exist_ok=True`.

**Cache hit / reuse (aminx-controlled layout).** The cache key is **owned by aminx**,
not derived from proxide's internal filename. Each remote source is fetched into a
**per-accession subdirectory** under the cache dir:

```
cache_dir/<scheme>/<accession>[/<fmt>]/
```

e.g. `cache_dir/pdb/1A3A/mmcif/`, `cache_dir/afdb/P12345/`, `cache_dir/mdcath/1abcA00/`.
The `<fmt>` segment is present only where format is part of identity (`pdb://` carries
`mmcif`/`pdb`; `afdb`/`mdcath` have a single canonical format, so no `<fmt>` segment).
The accession segment is the verbatim accession; path separators in an accession are
**rejected at classify time** (`typer.BadParameter`), so the subdir name is always
filesystem-safe and never glob-expanded.

- **Cache HIT** = that per-accession subdir already exists **and contains ≥1 non-empty
  file**. On hit, the resolver returns the path to the cached artifact (the single
  non-empty file in the subdir, or the subdir itself for multi-file sources) and
  **skips the network and any fetcher call entirely** — there is **no dependency on
  proxide's internal naming** and no foreknowledge of the written filename. aminx owns
  the key; the hit signal is pure *subdir-presence + non-empty content*.
- **Cache MISS** = subdir absent or empty. The resolver creates the subdir
  (`mkdir(parents=True, exist_ok=True)`), calls the fetcher with
  `output_dir=<that subdir>`, and on success the subdir now contains the artifact —
  which becomes a hit for all future runs.

Because aminx alone decides the subdir layout and the hit signal is verifiable without
knowing what proxide names the file, the cache key is fully under aminx control. (Cache
invalidation: honour the existing `--overwrite-cache` flag, `src/aminx/cli.py` /
`specs.py:218` — when set, delete the per-accession subdir first, then re-fetch.)

**Mock boundary (M4 — mandatory).** The new `io/proxide_fetch.py` module calls each
proxide fetcher **via module attribute**, never via a `from`-import:

```python
import proxide.io  # module-level import
...
proxide.io.fetch_rcsb(accession, output_dir=str(subdir), format_type=fmt)
proxide.io.fetch_afdb(accession, output_dir=str(subdir))
proxide.io.fetch_md_cath(accession, output_dir=str(subdir))
```

Tests patch at the **`aminx.io.proxide_fetch` namespace** — the exact monkeypatch
targets are `aminx.io.proxide_fetch.proxide.io.fetch_rcsb` (and `.fetch_afdb`,
`.fetch_md_cath`). Because the wrapper resolves the symbol through `proxide.io` at call
time (not a bound name captured at import), patching `proxide.io.fetch_rcsb` is
sufficient and deterministic. Gates G3/G4/G6 name this target so CI never hits the
network.

**Partial-download cleanup (M1 — mandatory).** A fetcher may raise **after** writing a
partial artifact into the per-accession subdir (truncated download, mid-stream socket
error). On **any** fetcher exception, the resolver **deletes the partially-written
artifact / the just-created per-accession subdir for that accession BEFORE** it
warns+skips (`fail_fast=False`) or re-raises (`fail_fast=True`). This guarantees no
corrupt file is ever left to satisfy a later cache HIT (closes m3): a subsequent run
sees an absent/empty subdir → MISS → clean re-fetch. The cleanup targets only the
subdir created for the failing accession (B1 isolation), so a sibling accession's
cached artifact is never touched.

**Error handling (all -> clear, actionable, non-zero exit):**
- **Offline / network failure:** proxide fetcher raises (native `_proxider` exception).
  Wrap broadly (`except Exception as exc`), **run partial-download cleanup first**, then
  re-raise as a typed `InputResolutionError` with: the URI, the resolved cache dir, and
  the hint "remote fetches require network access at CLI/submit time; resolve on a
  connected host, then submit with the cached local path (cluster compute nodes are
  offline)".
- **Unresolvable accession (404 / not found):** same path; message includes the
  accession and source ("RCSB returned no structure for 'pdb://1A3X'").
- **Cache mkdir / write failure (m2):** `PermissionError` / full-disk `OSError` on the
  `mkdir` or post-fetch write is mapped to `InputResolutionError` naming the cache dir
  and OS error (see §4 "Cache dir").
- **Malformed URI / unknown scheme:** `typer.BadParameter` at classify time, before
  any I/O.
- **`--input-type` vs scheme conflict:** `typer.BadParameter` (section 3 precedence rule 3).
- **`fail_fast` semantics:** preserve current `_expand_inputs(fail_fast=...)` contract
  (`src/aminx/cli.py:96`) — in `fail_fast=False`, a failed *remote* fetch (after
  cleanup) warns and skips (consistent with current dir/glob skip-on-empty); in
  `fail_fast=True`, exit 1 on first failure. Offline detection is reported regardless.

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
| G1 URI parse | `parse_input_uri` unit table: bare/`file://`/`pdb://`/`afdb://`/`mdcath://`, `pdb://1A3A.pdb` -> fmt=`pdb`, Windows `C:\x` -> local, unknown scheme -> error. **Plus (m1):** empty accession `pdb://` -> `typer.BadParameter`; `file://host/path` -> **strip host, resolve `/path`** (host segment dropped); `s3://bucket/k` and `http(s)://...` -> match grammar but have **no fetcher** -> `typer.BadParameter` (NO silent local fallback). |
| G2 dispatch | `--input-type` x scheme matrix: `auto` uses scheme; `pdb`/`afdb`/`mdcath` apply to schemeless tokens; flag-vs-scheme conflict raises `typer.BadParameter`. **Plus (M3) two cells:** (a) `--input-type file` + `pdb://1A3A` -> token treated as a **literal local path**, scheme detection suppressed, **zero fetcher calls**; (b) `--input-type pdb` + `afdb://P1` -> **hard `typer.BadParameter`**. |
| G3 fetch wrappers | `proxide_fetch` fetches into `cache_dir/<scheme>/<accession>[/<fmt>]/` and returns a `Path` inside that subdir; **cache HIT = subdir present + ≥1 non-empty file** (B1, aminx-controlled — no proxide-filename foreknowledge); mocked fetcher (patched at **`aminx.io.proxide_fetch.proxide.io.fetch_rcsb`** etc., M4) called **once** on miss, **zero** times on cache hit; `--overwrite-cache` deletes the subdir and forces re-fetch. |
| G4 offline error | mocked fetcher (patched at `aminx.io.proxide_fetch.proxide.io.*`, M4) raising -> **partial artifact / per-accession subdir is deleted first (M1)**, then `InputResolutionError` whose message names the URI, the cache dir, and the offline/submit-time hint; `fail_fast=True` exits 1, `fail_fast=False` warns+skips. Assert: after a raising fetch, the per-accession subdir is absent/empty (no corrupt hit, m3). |
| G4b runner guard (T7b) | feed an **unresolved-URI spec JSON** (`inputs=["pdb://1A3A"]`) through `run_specification_from_json` (`cli.py:1247`) -> `_loader_inputs` raises `InputResolutionError` naming the token (B2); same assertion for a campaign manifest row payload (`host/campaign.py`). |
| G5 local regression | `tests/data/*.pdb`/`*.cif`, dirs, globs resolve byte-identically to a golden list **captured from `main` HEAD before T1 and committed as a fixture** (m4); `file://` to a local fixture resolves to the same path. |
| G6 integration smoke | one end-to-end: `--inputs pdb://TEST` with the fetcher monkeypatched at **`aminx.io.proxide_fetch.proxide.io.fetch_rcsb`** (M4) to copy `tests/data/1ubq.pdb` into the per-accession subdir; resulting spec's `inputs` is `[<cache_dir>/pdb/TEST/.../...pdb]`; spec round-trips through `run_specification_to_json`/`...from_json`. |
| G7 portable no-regress | `tests/run/test_run_spec_portable_json.py` unchanged and green. |
| G8 spec-emit parity | all 8 `_expand_inputs` call sites (`cli.py:465,559,639,715,921,1009,1083,1153`) route through the resolver; `spec emit-*` JSON for a bare-path input is byte-identical to a **golden captured from `main` HEAD before T1 and committed as a fixture** (m4). |

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
| T7 | Offline/error-path hardening (partial-download cleanup, mkdir/write mapping) + `InputResolutionError` type | `io/proxide_fetch.py`, `src/aminx/cli.py` | T2, T3 | quick | impl | G4 |
| T7b | Runner-side unresolved-URI guard in `_loader_inputs` | `src/aminx/run/specs.py:84`; `tests/run/test_loader_inputs_guard.py` (new) | T1 | quick | impl | G4b |
| T8 | Integration smoke + portable-no-regress assertion | `tests/cli/test_inputs_integration.py` (new); `tests/run/test_run_spec_portable_json.py` (touch only to assert) | T5, T6, T7, T7b | standard | test | G6, G7 |
| T9 | Docs: CLI help text + `using-aminx` skill `--inputs` examples; URI table | `src/aminx/cli.py` help strings; skill docs | T4, T5 | quick | docs | help text renders; **examples-match-grammar check is a doctest (n3)** on the §3 grammar table |

**Parallelism**
- **Cluster A (serial spine):** T1 -> T2 -> T3 -> T4 -> T5.
- **Post-T1 parallel:** **T7b** (runner-side guard in `_loader_inputs`) depends only on
  T1 (it reuses the §3 scheme regex) and touches `run/specs.py` — disjoint from the
  resolver/CLI files — so it can run concurrently with the entire T2->T5 spine.
- **Post-T3 parallel:** T6 (`afdb`/`mdcath`) and T7 (error hardening) can run
  concurrently once T3 lands (both extend the resolver without colliding: T6 touches
  parser/wrapper *cases*, T7 touches *error mapping*; coordinate one merge on
  `proxide_fetch.py`).
- **T9 (docs)** can start in parallel with T6/T7 after T4 fixes the flag surface.
- **T8** is the join — depends on T5, T6, T7, T7b.

## 9. Risk table

| Risk | Severity | Likelihood | Mitigation |
| --- | --- | --- | --- |
| Offline cluster: a `*://` URI reaches the runner and fetches on a compute node | high | low (resolved at CLI by design) | Resolver runs at CLI/submit time only; `spec.inputs` is local paths post-resolve; prep.py never fetches. **First-class runner-side guard in `_loader_inputs` (B2) raises `InputResolutionError` on any unresolved `://` token — covers both bypass paths (`run_specification_from_json`, campaign manifest rows).** Gated by T7b/G4b. |
| Ambiguous scheme (`C:\path`, `s3://`, double `//`) | med | med | Strict grammar section 3 (`^[a-z][a-z0-9+.-]*://`); unknown scheme -> `typer.BadParameter` (no silent local fallback). G1 covers Windows-path + unknown-scheme. |
| Accession typo -> 404 / hang | med | med | proxide raises on 404; resolver maps to `InputResolutionError` with the accession; `fail_fast` honoured. G4. (Format pre-validation, e.g. 4-char PDB id, is optional follow-up, not blocking.) |
| Cache staleness (RCSB updates a structure) | low | low | Deterministic cached filename; `--overwrite-cache` forces re-fetch (G3). Document that the cache is content-addressed by accession, not versioned. |
| Proxide API drift (signature/return change) | med | low | Wrappers isolate all proxide calls in `io/proxide_fetch.py`; signatures pinned by `proxide>=0.1.0a9`; G3 mocks the exact signature. A single file to update on drift. |
| Partial failure in multi-input batch | med | med | Per-entry resolution; `fail_fast=False` warns+skips a bad entry (matches current dir/glob behaviour); `fail_fast=True` exits on first. Resolved list dedups first-seen. G4/G5. |
| `--input-type` silently overrides an explicit scheme | low | med | Made a **hard error** (section 3 rule 3), not a silent override. G2. |

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

**n1 — flag supersession.** This spec's `--input-type {auto,file,pdb,afdb,mdcath}`
flag **supersedes** the `--input-source` flag proposed in the research note
(`.praxia/docs/research/260622_1203_proxide-heterogeneous-inputs.md`). Implementers
must **not** re-introduce `--input-source`; the single `--input-type` flag (with `file`
as the force-all-local escape hatch, §3 rule 2) is the canonical surface.

## Review reconciliation (260622)

Reconciles the adversarial spec review for `#1203` (challenger `a4982fb`, defender
`a98d4df`). **2 blockers + 4 major + 5 minor resolved:**

| ID | Resolution (one line) | Landed in |
| --- | --- | --- |
| B1 | Cache key is aminx-controlled: per-accession subdir `cache_dir/<scheme>/<accession>[/<fmt>]/`; HIT = subdir present + ≥1 non-empty file. No proxide-filename foreknowledge. | §4 "Cache hit / reuse"; G3 |
| B2 | First-class runner-side guard in `_loader_inputs` (`specs.py:84`) raises `InputResolutionError` on any `^[a-z][a-z0-9+.-]*://` token; covers both bypass paths (`run_specification_from_json` cli.py:1247, campaign manifest rows). New task **T7b** + gate **G4b**. | §4 "Runner-side unresolved-URI guard"; §8 T7b; §9; G4b |
| M1 | Any fetcher exception deletes the partial artifact / per-accession subdir before warn/skip or re-raise; closes m3 (no corrupt hit). | §4 "Partial-download cleanup"; G4 |
| M2 | Entry list deduped pre-fetch by `(scheme, accession, fmt)` (not just post-resolution local paths) — kills cold-cache double-fetch. | §3 "Mixed inputs" |
| M3 | `--input-type file` suppresses scheme detection (force-all-local); `pdb`/`afdb`/`mdcath` apply only to schemeless tokens and hard-error on conflicting explicit scheme. Two G2 matrix cells. | §3 precedence rules 2-3; G2 |
| M4 | `proxide_fetch` calls fetchers via module attribute (`proxide.io.fetch_rcsb`); tests patch `aminx.io.proxide_fetch.proxide.io.*`; target named in G3/G4/G6. | §4 "Mock boundary"; G3/G4/G6 |
| m1 | G1 rows: empty accession -> BadParameter; `file://host/path` -> strip host to `/path`; `s3://`/`http(s)://` -> BadParameter (no silent local fallback). | §3 "Grammar edge cases"; G1 |
| m2 | mkdir+write wrapped in `InputResolutionError` mapping (permission/full-disk); per-accession subdirs mitigate concurrent-write races. | §4 "Cache dir"; G4 |
| m3 | Subsumed by M1 — partial-download cleanup guarantees no corrupt file persists as a hit. | §4 "Partial-download cleanup"; G4 |
| m4 | G5 (local golden) and G8 (spec-emit JSON) goldens captured from `main` HEAD before T1 and committed as fixtures (frozen bytes). | G5, G8 |
| m5 | Dropped `cache_path/inputs` tier (collides with JAX compilation cache prep.py:129-130); precedence is `--cache-dir` -> `AMINX_CACHE_DIR` -> `~/.cache/aminx/inputs` (XDG). | §4 "Cache dir" |
| n1 | `--input-type` supersedes the research note's `--input-source`; do not re-add. | §11 |
| n3 | T9 "examples match grammar" is a checkable doctest on §3. | §8 T9 |
