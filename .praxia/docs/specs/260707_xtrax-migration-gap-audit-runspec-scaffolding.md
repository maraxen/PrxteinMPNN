# xtrax migration gap audit: RunSpec write-only scaffolding + cardinality-divergence pattern

**Task:** 260707_xtrax-migration-gap-audit · **Status:** findings complete, remediation not chosen ·
**Scope:** aminx-wide fan-out audit for gaps of the same class as
`.praxia/docs/specs/260706_samples-axis-planner-cardinality-mismatch.md` (that document's finding is
folded in and *revised* here — see Finding D). **Extended same-day** with a second pass (Finding E)
checking a different question: not "is a migration broken/incomplete," but "does aminx duplicate
functionality xtrax already ships, by never having adopted it at all." Not part of EPIC #1541's phase
DAG; a cross-cutting audit prompted by it.

## How this audit was run

Four Claude Haiku recon subagents were fanned out in parallel, each scoped to a different slice:
(A) `RunSpec`'s other `eqx.Module` sub-configs beyond `BatchingConfig`, (B) staleness of
`.praxia/docs/plans/260614_runspec-migration-map.md`'s migration-status claims, (C) a spot-check of
EPIC #1541's P0/P1/P2 "DONE" claims, (D) a broader sweep of `host/`, `inference/`, `model/`, `potts/`,
`training/` for new instances of the same anti-pattern. All four returned; findings below are
consolidated and, where a claim was surprising enough to warrant it, independently re-verified via
direct `rg`/`Read` rather than taken at face value (per this project's "verify the measurement
pipeline" discipline).

**Before the fan-out**, the `praxia rig-run --flow recon --backend vllm/qwythos` microflow was tried
first as a cheaper alternative, per instruction. It failed three times in a row on pure infrastructure
issues, each diagnosed and fixed in turn:
1. `recon_contract.yaml` was never synced to `~/.praxia/workflows/` — fixed via
   `praxia dw install-templates --overwrite`.
2. `--backend vllm` alone doesn't match the "vllm/*" routing prefix — needs `vllm/<full-model-id>`.
3. The `vllm` provider itself was registered only in the *praxia* repo's own project-local
   `.praxia/backends.toml`, invisible to every other project. Promoted to the global
   `~/.praxia/backends.toml` (user-confirmed) so it resolves from any project going forward.

Once all three were fixed and the flow actually ran, it failed anyway — 6 rounds of "repeat action
detected" on `ripgrep`, the harness's repeat-guard forcing terminal-actions-only 5 times, then a
walltime timeout at 120s with **zero findings, confidence 0, `recon.jsonl` never written**. Qwythos-9B
could not complete a moderately complex, multi-file cross-reference query (9 fields × grep-and-classify)
at all. This matches the routing rule's own stated fallback condition ("confidence < 0.7 or multi-hop
synthesis needed") — confidence here was exactly 0. Claude Haiku subagents were used for the actual
audit instead, and did the work in ~2-3.5 minutes each with concrete, checkable evidence.

**Caveat on subagent output:** even Haiku's output required correction. Lane B's blanket "DEAD_CODE
(0 reads)" label for `noise_batch_size` and `multi_state_temperature` was too strong — direct `rg`
confirms both are read constantly via the *flat* `SamplingSpecification`/`RunSpecification` fields
(e.g. `host/plan.py:770: getattr(spec, "multi_state_temperature", 1.0)`); what's actually dead is only
the `RunSpec.tied.multi_state_temperature` / `RunSpec.batching.noise_batch_size` *sub-config copy* —
consistent with Finding A's pattern, not evidence the underlying feature is unused. Treat every
"dead code" claim below as "the RunSpec sub-config copy is dead," not "nobody uses this."

---

## Finding A: `RunSpec`'s write-only sub-config scaffolding is far broader than the original BatchingConfig finding

`src/aminx/run/spec.py` defines 11 `eqx.Module` sub-configs on `RunSpec` (`io`, `resource`,
`multistate`, `ligand`, `tied`, `grid`, `batching`, `averaging`, `precision`, `plan`, `sampling`),
all populated by `build_run_spec()` (lines 292-404) from the flat `SamplingSpecification`/
`RunSpecification` dataclass. Auditing read-sites for every field across all 11 (BatchingConfig's 9
fields were already fully confirmed dead in the prior session pass) finds **27 total fields, across 5
of the 11 sub-configs, with zero read sites anywhere outside their own construction**:

| Sub-config | Dead fields | Live fields |
|---|---|---|
| `BatchingConfig` (9 fields) | **all 9** — `batch_size`, `samples_batch_size`, `samples_chunk_size`, `noise_batch_size`, `temperature_batch_size`, `jacobian_batch_size`, `combine_batch_size`, `apc_batch_size`, `apc_residue_batch_size` | none |
| `TiedPositionsConfig` (5 fields) | **all 5** — `tied_positions`, `pass_mode`, `multi_state_temperature`, `tie_group_map`, `structure_mapping` | none |
| `AveragingConfig` (4 fields) | **all 4** — `average_node_features`, `average_encoding_mode`, `average_encodings`, `state_weights` | none |
| `GridLineageConfig` (6 fields) | 5 — `campaign_mode`, `job_id`, `chunk_id`, `sample_start`, `sample_count` | `grid_mode` (`run_spec_portable_json.py:177`) |
| `LigandConfig` (5 fields) | 4 — `use_side_chain_context`, `ligand_conditioning`, `sidechain_conditioning`, `context_path` | `model_family` (guard check, `run_spec_portable_json.py:185`) |
| `IOConfig`, `ResourceConfig`, `MultistateConfig`, `PrecisionConfig`, `PlannerTopology`, `SamplingConfig` | none | all fields read (mostly via `run_spec_portable_json.py`'s serialization round-trip, plus real production reads for `SamplingConfig`'s fields — see Finding below) |

Two important qualifiers:
- `SamplingConfig` (`num_samples`, `random_seed`, `return_logits`, `backbone_noise`, `temperature`,
  etc.) is the one sub-config whose fields are read constantly in real production call sites
  (`host/kernel_dispatch.py`, `host/runner.py`, `host/campaign.py`, `host/_sampling_grid_lineage.py`) —
  this is the part of the RS-1 migration that genuinely landed. It's the counter-example that shows
  the other 5 sub-configs' dead status is a real gap, not an audit-methodology artifact.
- Most of the "read" sites for `IOConfig`/`ResourceConfig`/`MultistateConfig` are exclusively inside
  `run/run_spec_portable_json.py`'s own serialize/deserialize round-trip (lines ~177-210) — i.e. the
  value gets written into `RunSpec`, then read back out only to be re-serialized to JSON, with no
  confirmed production consumer downstream of that JSON. This wasn't independently chased further;
  flagged as an open question below.

**Total: 27 of 72 RunSpec sub-config fields (BatchingConfig's 9 + these 18) are populated at
construction time and never read again anywhere in the codebase.** This is the "RS track"
(RunSpec-unification migration, `.praxia/docs/specs/260611_runspec-unification.md`) in a state the
spec index still describes as "Active... blocks T4.1" — the audit suggests "stalled, partially
inverted" is more accurate (see Finding B).

---

## Finding B: the RS-1 migration map (`260614_runspec-migration-map.md`, `status: complete`) is substantially wrong, in both directions

This doc claims 67 fields inventoried: 22 already migrated, 16 to migrate, 9 RS-gaps, 21 protein-only.
Auditing every row marked "already migrated":

**All 22 "already migrated" rows are stale.** 19 still read the old flat field directly at the cited
call site (e.g. `plan.py:367-368`'s `resolve_chunk_size` still does
`if hasattr(spec, "samples_chunk_size") and spec.samples_chunk_size: return int(spec.samples_chunk_size)`,
never touching `spec.run_spec.batching.samples_chunk_size`); the other 3 (`noise_batch_size`,
`average_encoding_mode`, `multi_state_temperature`) are dead in the specific sub-config-copy sense
described in the caveat above. Representative stale rows (file/line as cited in the map, field, claimed
target, actual status):

| File | Field | Claimed target | Actual |
|---|---|---|---|
| kernel_dispatch.py:31-33 | `tie_group_map`, `structure_mapping`, `state_weights` | `tied.*` / `averaging.state_weights` | old flat field, 1-3 real reads each |
| _sampling_helper.py:39-42 | `model_family`, `ligand_context_path`, `ligand_conditioning`, `sidechain_conditioning` | `ligand.*` | old flat field, 2-8 real reads each |
| runner.py:48,51,61 | `grid_mode`, `multi_state_strategy`, `apc_residue_batch_size` | `grid.*` / `batching.*` | old flat field, 1-5 real reads each |
| plan.py:122 | `samples_chunk_size` | `batching.samples_chunk_size` | old flat field (already confirmed prior session) |
| _sampling_grid_lineage.py:110-114 | `sample_count`, `sample_start`, `chunk_id`, `job_id` | `grid.*` | old flat field, 1 read each |

**In the other direction**, a spot-check of 6 fields the map lists under "16 to migrate" found they
are *already* fully migrated and read exclusively via `spec.run_spec.sampling.<field>`:
`backbone_noise` (8 sites), `temperature` (5 sites), `num_samples` (2 sites), `random_seed` (6 sites),
`return_logits` (11 sites), `compute_pseudo_perplexity` (1 site) — independently re-confirmed via `rg`.

The "9 RS-gaps" section spot-checked as accurate (`sequences_to_score`, `distance_matrix`,
`jacobian_mode`, `compute_apc` are genuinely still flat-only, as claimed).

**Recommendation:** this doc's `status: complete` frontmatter is actively misleading — it should be
changed to `status: outdated`, and any planning decision that relied on its "22 migrated" figure
(e.g. scoping for T4.1, which the RS track spec says this blocks) should be revisited against the
table above, not the doc.

---

## Finding C: EPIC #1541 P0/P1/P2 "DONE" claims — sampled, no new gaps found, coverage partial

A time-boxed sample of `.praxia/docs/specs/260611_aminx-xtrax-refactor.md`'s P0/P1/P2 claims (P3/P4
already independently verified in prior session passes) found:
- **P1 training (ResumableState on xtrax):** confirmed genuinely live — `trainer.py:19,173,207,209,795`
  import and use `xtrax.training.ResumableState`, no parallel legacy path detected.
- **P2/P3-adjacent dispatch:** confirmed `factory.py` calls `make_axis_dispatch_via_xtrax` (not the
  legacy `make_axis_dispatch`) at lines 91/95/99 — the legacy function's continued presence in
  `tiling/dispatch.py` is the deliberate, documented T2.GATE reference implementation, not a gap.

**Not checked** (explicitly out of the time-boxed sample, not confirmed either way): T0.3 (SM120 L3
smoke), T0.4 (ruff TID251 boundary lint wiring), T2.5b (heterogeneous axis injection), T2.GATE's actual
test execution (bit-for-bit golden, recompile tripwire, cluster bench), T2.6 (StageBundle wrap-adapter
sink multiplicity/order), T1.4 (checkpoint round-trip golden). None of these should be read as "passed
implicitly" — they're simply unaudited.

---

## Finding D: the default production dispatch path has *no* size-safety fallback at all — this revises and strengthens 260706's cardinality-mismatch finding

The prior session's cardinality-mismatch spec analyzed `_safe_map(_run_one_sample, sample_keys,
batch_size=samples_bs)` (`kernel_dispatch.py:434,531` in that document's line numbering) — the *legacy*
dispatch path. Tracing `kernel_dispatch.py` fully during this audit surfaces that this is **not** the
default path:

- `PlannerTopology.use_unified_driver` defaults to `True` everywhere it's set (`run/specs.py:504`,
  `cli.py:663,1180`, `run_spec_portable_json.py:155`), and `spec.run_spec.plan.use_unified_driver`
  (`kernel_dispatch.py:185`) gates which path `_sample_batch` takes: `if _use_unified and
  plan.stage_set.encoding_fusion is None:` (line 187) — i.e. the unified path runs whenever the driver
  flag is at its default AND encoding fusion isn't configured (not independently confirmed how common
  the latter condition is in production configs, but it is not a rare/exotic combination).
- The unified path dispatches via `_dispatch_axis` (lines 250, 261, 334, 355, 371), whose `Vmap` branch
  (`kernel_dispatch.py:69-70`) is:
  ```python
  if strategy_name == "Vmap":
      return jax.vmap(body)(xs)
  ```
  **Unconditional.** No `num_elements`/`batch_size` comparison of any kind — unlike `safe_map`, which
  at least routes through an `if batch_size is None or batch_size == 0 or num_elements <= batch_size`
  check before deciding to skip chunking (`utils/safe_map.py:49`, the mechanism the original finding
  was built around). When `make_sampling_planner`'s `N_SAMPLES` decision is `Vmap` (chosen because the
  small, disconnected `samples_batch_size` fit the memory budget), the unified path hands the *actual*
  `sample_keys`/`key_samples` array — sized by `target_num_samples`, resolved independently — straight
  to `jax.vmap`, with no gate of any kind in between.

This means the originally-documented mechanism (a `batch_size == 0` sentinel silently defeating
`safe_map`'s check) is the *milder*, non-default variant of this bug. The default, more commonly hit
variant skips the check-with-a-defeatable-condition entirely and goes straight to an unconditional
`jax.vmap` call. Same root cause (disconnected `samples_batch_size` vs. `samples_chunk_size`/
`num_samples`), same two files, but the severity/likelihood framing in `260706_samples-axis-planner-
cardinality-mismatch.md` should be read as understating the default-path risk — **that document should
be amended** to describe both the unified (`_dispatch_axis`, unconditional) and legacy (`safe_map`,
defeatable-condition) manifestations, not just the latter.

No other new Pattern-A instances were found in `host/`, `inference/`, `model/`, `potts/`, `training/`
outside this one; no new Pattern-B (write-only scaffolding) instances were found outside the RunSpec
sub-configs already covered in Finding A.

---

## Finding E: unadopted xtrax capabilities — duplicated local implementations, not just broken migrations

Findings A-D all concern places aminx *tried* to adopt xtrax and the adoption is broken or
incomplete. This is a different question: of xtrax's 15 top-level subpackages (`checkpoint`, `cli`,
`data`, `distributed`, `eda`, `engine`, `inference`, `io`, `run`, `safety`, `sparse`, `stages`,
`tiling`, `training`, `transforms` — confirmed via `pkgutil.iter_modules` against the installed
0.4.0a1), aminx imports from only 6 (`tiling`, `run`, `stages`, `training`, `checkpoint`, `eda`). The
other 9 were checked for local-duplicate signal via targeted `rg` + direct source comparison:

**Confirmed duplicate effort — `xtrax.transforms.safe_map`/`safe_scan`.** aminx's
`src/aminx/utils/safe_map.py` and `src/aminx/utils/safe_scan.py` are local reimplementations, not
uses of `xtrax.transforms.safe_map`/`safe_scan` (installed at
`.venv/.../xtrax/transforms/{map,scan}.py`). Diffing the two `safe_map`s directly: xtrax's canonical
version treats `batch_size=None` as its only "always vmap" sentinel and has **no** special case for
`batch_size=0` — passing `0` hits `n % batch_size` and raises `ZeroDivisionError`. aminx's fork
*added* a `batch_size == 0` "always vmap" branch on top of xtrax's semantics, to interoperate with
its own legacy int-based dispatch calling convention (`_legacy_batch_size`'s Vmap-sentinel
translation, see the 260706 document). **This addition is the exact mechanism that lets the
cardinality-divergence bug (Finding D and the 260706 document) fail silently instead of loudly** —
had aminx been calling xtrax's canonical `safe_map` directly with a real `batch_size` value (no `0`
sentinel), the same disconnected-cardinality input would raise `ZeroDivisionError` at the `lax.map`
boundary rather than quietly vmapping an oversized array. `safe_scan` shows the same divergence
pattern in miniature: xtrax's version validates the actual leading-axis length is nonzero; aminx's
only checks whether the pytree has zero leaves at all (a materially weaker check — an array of shape
`(0, N)` passes aminx's check but would be caught by xtrax's). **This directly changes the
remediation picture for the cardinality bug**: migrating dispatch onto xtrax's canonical `safe_map`
(forcing every caller to pass a real batch size, eliminating the `0`-sentinel path entirely) is a
strictly cleaner fix than patching the local fork's condition — not merely "another option" among
the three listed in the 260706 document's Remediation section, but the one most consistent with how
xtrax itself expects this primitive to be called.

**Confirmed duplicate effort — `xtrax.engine.Engine`.** `xtrax.engine` ships a generic `Engine`
class (`fit`/`fit_sync`/`eval`, taking a `Trainer`/`TrainStepLike` + callbacks) built specifically to
orchestrate a training loop. `src/aminx/training/trainer.py` (939 lines) hand-rolls its own loop
(`train()` at line 707, `train_step()` at 293, `eval_step()` at 571) instead of using it — despite
already importing *other* pieces of `xtrax.training` (`adamw_with_schedule`/`make_optimizer`,
`ResumableState`) and `xtrax.checkpoint.orbax` into that same custom loop. Not independently assessed
whether `Engine.fit()` could be a drop-in replacement for aminx's specific training requirements
(e.g. custom eval cadence, mixed-precision handling at `setup_mixed_precision`) — flagged as real
duplicated orchestration code, not confirmed as a safe swap.

**Lower-urgency duplicate effort — `xtrax.safety.safe_norm`/`safe_reciprocal`.**
(`Float[Array,...], eps=1e-8) -> Float[Array,...]`, computing `sqrt(sum(x**2)+eps**2)` and
`1/(x+eps)` respectively). aminx has 10+ scattered ad-hoc epsilon-guard patterns doing the same
thing by hand across `inference/decode/_kernel.py:164`, `parity/evidence.py:349-407`,
`inference/decode/ste.py:318`, `inference/optimize_ste.py:220,237`, `inference/logits.py:114-115`,
`utils/coordinates.py:161`, `utils/reverse_jac.py:47`, `utils/structure_metrics.py:108`,
`host/averaging.py:505`, `utils/normalize.py:21,80`. Real duplication, but a style/consolidation
question rather than a correctness gap — none of these were found to have the wrong epsilon or a
missing guard, just independently-invented instances of the same pattern.

**Checked, no gap found:** `xtrax.sparse` (`SparseConfig`/`sparsify_model`/etc. — zero local
sparsity/pruning code anywhere in aminx), `xtrax.distributed` (`init_dist`/`ShardingPolicy`/etc. —
zero local device-mesh/sharding code), `xtrax.data` (`DataModule` — aminx's own
`create_protein_dataset` in `host/prep.py` wasn't found to structurally overlap on inspection, though
not deeply diffed), `xtrax.inference` (axis/signature inference — `AxisRole`/`BundleSchema`/
`infer_bundle`/`synthesize_axes` — zero references anywhere in aminx, no evidence of a hand-written
equivalent), `xtrax.cli` (batch-plan visualization CLI — aminx's own `cli.py` doesn't expose plan
visualization via any command, so this reads as an unused feature, not duplicated effort). None of
these show a local reimplementation competing with the xtrax version — they read as genuinely unused
capability with unclear current applicability, not wasted effort. `xtrax.io`'s
`BoundedCallbackHandler`/`async_indexed_stream` (same names as aminx's pre-existing
`_vendored_callbacks.py`) was already examined by the P4 scoping spec this session and confirmed to
be a naming collision with different semantics (aminx: sync FIFO backlog / sync int-range chunker;
xtrax: async semaphore-bounded executor backing `Engine.fit()` / async prefetch generator) — not a
migration target, already resolved, no new action here.

---

## Consolidated file:line findings

| # | File:line | Issue | Class |
|---|---|---|---|
| 1 | `run/spec.py:84-96` (`BatchingConfig`) | 9/9 fields dead | Write-only scaffolding |
| 2 | `run/spec.py:63-70` (`TiedPositionsConfig`) | 5/5 fields dead | Write-only scaffolding |
| 3 | `run/spec.py:98-104` (`AveragingConfig`) | 4/4 fields dead | Write-only scaffolding |
| 4 | `run/spec.py:73-81` (`GridLineageConfig`) | 5/6 fields dead | Write-only scaffolding |
| 5 | `run/spec.py:53-60` (`LigandConfig`) | 4/5 fields dead | Write-only scaffolding |
| 6 | `.praxia/docs/plans/260614_runspec-migration-map.md` | 22/22 "migrated" rows stale; 6+ "to migrate" rows already done | Documentation drift |
| 7 | `host/kernel_dispatch.py:69-70`, `:100-371` | Unified dispatch path (`use_unified_driver=True` default) vmaps `sample_keys` unconditionally when planner picks Vmap for a disconnected cardinality | Cardinality divergence (default path, no fallback) |
| 8 | `host/kernel_dispatch.py:421,518` (legacy path) | Same divergence via `safe_map`'s defeatable `batch_size==0` branch | Cardinality divergence (legacy path, already documented 260706) |
| 9 | `utils/safe_map.py`, `utils/safe_scan.py` | Local forks of `xtrax.transforms.safe_map`/`safe_scan`; fork *added* the `batch_size==0` sentinel that lets finding #7/#8 fail silently instead of raising | Unadopted xtrax capability (duplicate implementation) |
| 10 | `training/trainer.py` (939 lines, `train()`/`train_step()`/`eval_step()`) | Hand-rolled training loop instead of `xtrax.engine.Engine.fit()`, despite already using other `xtrax.training`/`xtrax.checkpoint` pieces inside it | Unadopted xtrax capability (duplicate implementation) |
| 11 | 10+ sites across `inference/`, `utils/`, `host/averaging.py` | Ad-hoc epsilon-guard patterns (`x + 1e-8` etc.) instead of `xtrax.safety.safe_norm`/`safe_reciprocal` | Unadopted xtrax capability (style/consolidation) |

---

## Open questions

- Are `IOConfig`/`ResourceConfig`/`MultistateConfig`'s "read" sites (exclusively inside
  `run_spec_portable_json.py`'s serialize round-trip) actually consumed by anything downstream of the
  JSON they produce, or is that also effectively write-only once one more hop is traced?
- How common is `plan.stage_set.encoding_fusion is None` in real production sampling configs? This
  determines how often Finding D's unconditional-vmap path is actually hit vs. falling through to the
  legacy path.
- Was `average_encoding_mode`'s explicit "Dead field; kept for backward compat, never used" comment
  (`run/spec.py:367`) written when RS-1 landed, or is it older/unrelated context? If older, it's a
  precedent that this project already has a convention for marking fields dead-on-purpose — the other
  26 write-only fields in Finding A have no such marker and were not (as far as this audit found)
  deliberately deprecated, which argues they're accidental scaffolding rather than intentional legacy.
- Whether the vllm/qwythos backend's zero-findings failure on this exact task class is specific to
  Qwythos-9B or would recur with the default `ollama/qwen3.6:latest` recon backend — not tested this
  session (the fan-out went straight to Claude subagents once vllm/qwythos hit zero confidence).

## Remediation

**Item (3), the cardinality-divergence bug, is now FIXED (2026-07-07)** — see the "Remediation
implemented" section at the top of `260706_samples-axis-planner-cardinality-mismatch.md`.
`make_sampling_planner` now takes the real per-call sample count as `n_samples_override`, resolved and
passed by `kernel_dispatch._sample_batch` before planning rather than after. This fixes *both*
manifestations at once, including finding #7's `_dispatch_axis` unconditional-`jax.vmap` path — once
the cardinality the planner reasons about is correct, a Vmap decision genuinely is safe for the real
array, so `_dispatch_axis` having no independent runtime fallback check stops being a problem specific
to that path. The `xtrax.transforms.safe_map` adoption angle (item 4 / Finding E) was deliberately
**not** folded into this fix — it's a separate, larger-blast-radius change (every `aminx.utils.safe_map`
call site, not just this one) that remains valuable as defense-in-depth (turns a future
disconnected-cardinality mistake into a loud crash instead of a silent bypass) but wasn't required to
close this specific bug.

Given the scale surfaced (27 dead fields across 5 sub-configs, a migration-status doc that's backwards
on both its "done" and "not done" claims), the next decision is priority/sequencing across the three
remaining workstreams: (1) fix or formally abandon the RS-1 sub-config migration, (2) re-author the
migration map from actual code state, (4) adopt `xtrax.transforms.safe_map`/`safe_scan` and evaluate
`xtrax.engine.Engine` as a `trainer.py` replacement.
