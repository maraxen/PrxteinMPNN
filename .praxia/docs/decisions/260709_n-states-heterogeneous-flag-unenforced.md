---
title: N_STATES.heterogeneous=True is unenforced and currently unenforceable — decide relabel vs. implement
status: Open — decision deferred, not blocking any current work
date: 2026-07-09
related: spec ../specs/260709_mbr-consensus-reranking-composition.md (PR #92)
---

# Decision needed: `N_STATES.heterogeneous=True` — relabel or implement

## Finding

`N_STATES` (`src/aminx/tiling/axes.py:45-51`) is registered with `heterogeneous=True` ("Shapes vary
across states"). This flag is **not connected to the real state-encoding production path** — confirmed
by independent oracle-level review during PR #92's design work, via three separate checks:

- `make_encode_fn` (`inference/encode.py:267-295`) selects between `_VmapEncode`/`_ScanEncode` via a
  **plain boolean** `use_rolling_state`, sourced directly from a spec field at `host/plan.py:790-791` —
  no `AxisSpec`, `BatchPlanner`, or `bucket_boundaries` consulted anywhere in this factory.
- The one real heterogeneous-axis guard in this codebase (`_plan_with_joint_budget`, `host/plan.py:106-116`,
  which force-fixes heterogeneous axes to `SafeMap`) only fires for axes actually passed to it —
  `make_sampling_planner` (`plan.py:229-236`) passes exactly `N_STRUCTURES, N_SAMPLES, N_TEMPERATURES,
  N_NOISES`. `N_STATES` is not among them.
- `host/kernel_dispatch.py`'s `decision_for` calls (`kernel_dispatch.py:196-199,275-278`) cover the same
  four axes — never `n_states`.

**More fundamental than "unenforced":** `InferenceBundle`/`GeometryBundle` cannot even represent
genuinely ragged states — `coords`, `atom_37`, `physics_features`, and masks are all single stacked
arrays with a uniform leading `S` axis (`encode.py:52,118-132`). There is no code path by which a
caller could construct a bundle with per-state-varying shapes in the first place; you'd fail at array
construction, long before `make_encode_fn` or any planner logic runs. No existing test (checked all
`n_states`-referencing test files) exercises genuinely ragged states — every one builds a uniform stack.

**Practical consequence today: zero blast radius.** Every real caller (confirmed: tev_design's necklace
campaign via `build_canonical_bundle.py`, `N_CANONICAL=214`) pre-pads to a uniform shape before ever
constructing an `InferenceBundle`. The `heterogeneous=True` label describes an aspiration — that states
might conceptually differ in shape — that the current data model does not support, rather than a live
enforcement gap protecting against a real, reachable failure.

## Why this surfaced now

PR #92 (MBR post-hoc consensus reranking) went through five rounds of design correction, one of which
(round 4) assumed `N_STATES.heterogeneous=True` meant "dispatch this axis via xtrax's `BatchPlanner`,
which will correctly demote to a padding-aware strategy" — this is false; `BatchPlanner`/`SafeMap` cannot
ragged-iterate at all, and nothing in this codebase would have caught that assumption being wrong until
an actual heterogeneous input was attempted (which, per the above, cannot currently even be constructed).
The registry flag reads as a live contract but isn't wired to anything that would honor it.

## Decision options (not decided here — flagging for deliberate choice)

1. **Relabel**: set `N_STATES.heterogeneous=False` (or remove the field / add a comment clarifying it's
   currently aspirational) and correct the module docstring at `tiling/axes.py` accordingly. Cheapest;
   makes the registry honest about current behavior. Loses the "this axis might need special handling
   someday" signal for future readers.
2. **Implement**: wire real padding/bucketing support into `make_encode_fn`/`_VmapEncode` so a genuinely
   heterogeneous-state caller (should one ever exist) is handled correctly and automatically, honoring
   the registry's existing claim. Real engineering work on a live production code path (encode/AR
   sampling) — its own test/parity/JIT burden, a different risk class than adding new post-hoc scoring
   code, and explicitly out of scope for PR #92 (see that PR's Provenance section, round 6).

## Non-decision for now

PR #92's `decode_states_unfused`/MBR design takes the cheap, safe path: it explicitly asserts/documents
that its states arrive pre-padded to a uniform shape (a stated precondition, not an implicit assumption)
rather than depending on `N_STATES`'s registry flag meaning anything operationally. This document exists
so the underlying registry/production-path gap isn't lost, decided by default, or rediscovered the hard
way a sixth time.
