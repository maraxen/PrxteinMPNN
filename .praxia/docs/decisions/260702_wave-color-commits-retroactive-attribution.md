---
title: Retroactive attribution — 5 wave-color commits landed in aminx, driven by mpnn_ext Epic WAVE
decision_id: 260702_wave-color-commits-retroactive-attribution
date: 2026-07-02
status: Accepted
decision_type: process
relates_to: mpnn_ext/.praxia/docs/roadmaps/consolidated-cross-project/260702_00-mandate.md
---

## Status: Accepted

## Context

`mpnn_ext`'s Epic WAVE (chromatic wave-color decoding-order scheduling, praxia epic `#2871`)
was explicitly migrated out of `tev_design` into `mpnn_ext` on 2026-06-30. `aminx`'s own
`daily.jsonl` entry for that session (`260630_wave-color-scheduling-session-close`) instructs
future work to file W0/WAVE work under `mpnn_ext`, not `aminx` or `tev_design`.

Despite that instruction, five commits on branch `worktree-w0-wave-color` implemented Epic WAVE
requirements directly in `aminx`, with no epic-ID reference in any commit message and no
cross-linking decision doc at the time each commit landed:

| Commit | Summary |
|---|---|
| `54d6d84` | feat(wave-color): W0.1 LogitFingerprint EDA return + W0.2 xtrax>=0.3.0 + W0.3 schedule selector |
| `0be59ef` | fix(decode): real causal masking + G>1 chromatic-wave decode in AutoregressiveDecode |
| `4060e9d` | feat(decode): add explicit wave override to build_inference_bundle |
| `0670197` | fix(decode): generate_wave_ar_mask sentinel leaked omitted positions in partial schedules |
| `1cec556` | build: add cuda12 optional-dependency group for GPU-enabled jaxlib |

This is exactly the failure mode the consolidated cross-project roadmap's boundary-enforcement
mechanism (`mpnn_ext/.praxia/docs/roadmaps/consolidated-cross-project/260702_00-mandate.md`
§4.1 item 2) exists to prevent going forward — a policy stated once, in prose, that failed
anyway because nothing enforced it mechanically. Backlog `#2954` (this project) tracks landing
that enforcement lint.

## Decision

These five commits are **kept in place** — no revert, no replay onto a different branch/repo.
`aminx` is the correct implementation location for MPNN model-library code regardless of which
project's epic drives the requirement (per the roadmap's ownership map, `aminx` is pure
infra/substrate that implements research-epic requirements; it does not need to "own" the
research question to be the right place for the code).

What was missing was **attribution**, not placement. This document retroactively supplies it:
all five commits implement requirements of `mpnn_ext` Epic WAVE (`#2871`), specifically Phase 0
(instrumentation + wiring: W0.1 LogitFingerprint EDA return, W0.2 xtrax pin, W0.3 schedule
selector wiring) and follow-on bugfixes surfaced during Stage 1a (`W1a.1`) confirmatory work
(real causal masking for `G>1` chromatic-wave decode, the `generate_wave_ar_mask` sentinel leak,
an explicit wave override on `build_inference_bundle`, and the `cuda12` optional-dependency
group needed to run the `W1a.1` confirmatory campaign on cluster GPU).

## Consequences

- No code or history change required.
- `mpnn_ext/.praxia/docs/roadmaps/wave-color-scheduling/260630_00-mandate.md` (Epic WAVE's own
  mandate doc) should carry a reciprocal cross-link back to this document and these five SHAs,
  so the attribution is discoverable from either direction.
- Future wave-color-adjacent commits in `aminx` are subject to backlog `#2954`'s lint once
  landed — this document is a one-time retroactive fix, not a template for skipping the lint
  going forward.

## Out of scope

Re-litigating whether Epic WAVE's implementation *should* live in `aminx` vs. a hypothetical
`mpnn_ext`-owned fork of the decode path — the consolidated roadmap's ownership map already
settled this (`aminx` = implementation substrate for any research-epic requirement, regardless
of which project's epic originates the requirement).
