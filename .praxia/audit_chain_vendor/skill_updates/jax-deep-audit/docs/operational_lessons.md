# Operational lessons — 260826 chain-selection-vendor-superset-audit

Process discipline learned while running this skill's methodology for real, beyond the
checker mechanics documented elsewhere. These are execution-environment and resolver-judgment
lessons, not new invariants.

## Resolver fix-strategy: loud refusal over guessing correctness-sensitive semantics

When a differential probe confirms a field is silently inert (the F002/F005/FA2/FA3 shape),
the resolver has two options: wire it up for real, or make the gap loud (raise, naming the
field, pointing at a surface that does honor it). Default to the loud fix whenever:

- The correct semantics are genuinely ambiguous (more than one plausible interpretation, and
  the difference matters), AND
- The surrounding code has documented history of subtle correctness bugs from unverified
  changes (comments describing a specific measured leakage, a specific nats/sigma effect
  size, a specific seed-pairing requirement — anything indicating the code owner already paid
  a real cost to get the current behavior right).

A loud refusal is reversible and honest: it costs a caller an explicit error instead of a
silent wrong answer, and it can be replaced by a real implementation later once the owner
resolves the semantics question. A guessed implementation that turns out wrong is much more
expensive to find, because it looks like it works.

Only implement the fix for real when the aggregation/masking change is separable from the
correctness-sensitive core (e.g. "exclude these positions from the final reduction" bolted
onto an already-computed per-position array) AND you can verify the change against the same
kind of evidence the surrounding code already trusts (bit-identity on the untouched paths,
not just "the new code path runs without crashing").

## A single stale identifier can be a mix of load-bearing and cosmetic references

Don't assume every occurrence of a suspicious pin/SHA/identifier carries the same risk. Split
occurrences into:

1. **Functional** — code that actively uses the value (clones a commit, resolves a path,
   drives a runtime decision). Stale here is a real bug.
2. **Documentation/provenance** — a comment or metadata field recording what a *frozen,
   already-committed* artifact was derived from at the time. Stale here is a cosmetic
   inaccuracy; "fixing" it by pointing at a newer value would misrepresent history.

`grep` finds every occurrence but conflates the two. Read each occurrence's surrounding code
before deciding it needs to change — one pass, this session, over-corrected an initial "the
pin is stale" finding into "half these references were already fixed and the other half is
correctly historical, not stale at all."

## Offloading differential probes to remote compute

Real bit-identity/differential probes need to actually load models and run inference — often
more compute/disk than the auditing session's own sandboxed environment should carry. Lessons
from running this on a shared lab workstation (not the audit sandbox itself):

- **Check per-GPU occupancy before pinning a device.** `nvidia-smi --query-gpu=index,memory.used,memory.total`
  before setting `CUDA_VISIBLE_DEVICES` — a shared box can have some GPUs fully claimed by an
  unrelated long-running job (e.g. a serving process) while others sit idle.
- **`ssh host "command"` often needs an explicit login shell** (`ssh host "bash -lc 'command'"`)
  to pick up `uv`/`cargo`/etc. on `$PATH` — a bare non-interactive SSH command frequently skips
  the profile that sets it up, producing a confusing "command not found" that looks like a
  missing install.
- **Git LFS pointer stubs look like real files until you read their content.** A `git clone`
  without `git-lfs` installed leaves LFS-tracked binaries (checkpoints, weights) as small text
  stub files (`version https://git-lfs.github.com/spec/v1 ...`) with the RIGHT file size shown
  by `ls -la` in some tooling but garbage content — a decompression/parse error two layers away
  from the actual cause is the tell. If another checkout on the same host already has the real
  content, copying the real files over the stubs is a legitimate low-risk workaround when
  installing `git-lfs` isn't an option (no sudo, shared box).
- **A repo's `git log --all` commit count is a fast sanity check for "does this pinned SHA
  still exist."** A vendor repo with a rewritten/reset history can leave a previously-valid pin
  permanently unreachable with no local signal beyond a plain "not a valid object name" — check
  `git log --all --oneline | wc -l` against what you'd expect before concluding the pin itself
  (versus your local clone) is the problem.

## Sandbox / git-worktree isolation interactions

- A git worktree shares `.git/config` (and other metadata) with the primary checkout, which
  can sit **outside** the sandbox's write-allowed paths even though the worktree's own working
  directory is inside them. A denied write there can surface as a phantom character-device
  file (`crw-rw-rw- nobody nogroup`) at paths like `.git/config.lock` or `.gitmodules` instead
  of a clean permission error — that signature is itself the tell to retry the git operation
  with the sandbox disabled, not a sign of real filesystem corruption.
- A worktree-isolated session cannot write outside its own worktree directory at all — this
  applies to genuinely shared, version-control-external tooling directories (like this very
  skill's live install location) as much as it does to source code. There is no in-session
  workaround that respects the isolation boundary other than staging the change inside the
  worktree/branch and handing it to the user (or a future non-isolated session) to apply.
