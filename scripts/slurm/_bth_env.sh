# Source from SLURM scripts: source scripts/slurm/_bth_env.sh
# Sets BTH_PROJECT_SLUG, BTH_PROJECT_ROOT, BTH_WORKSPACE_ROOT, and BTH_CATALOG_DIR so bth runs transparently in batch jobs.
set -euo pipefail
export BTH_PROJECT_SLUG="aminx"

# Deterministic workspace filesystem root, resolved rather than hardcoded.
#
# The requirement (spec 260611) is that this be an ABSOLUTE path and not depend
# on `git rev-parse --show-toplevel`: in a SLURM spool dir that resolves to an
# unrelated repo or fails outright. That still holds. What changed (260817) is
# HOW the absolute path is obtained -- it used to be the literal
# `/home/marielle/projects/aminx`, which is this workstation's path and does not
# exist on the cluster (engaging's home is /home/maarxaru, and the project
# arrives there as an rsync'd copy with no .git at all). Any batch job sourcing
# the old version got a BTH_PROJECT_ROOT pointing at a nonexistent directory.
#
# Resolution order (revised 260817 after measuring what SLURM_SUBMIT_DIR actually
# contains -- see below; do not reorder these without re-measuring):
#   1. This file's own location, two levels up (scripts/slurm/ -> project root).
#      Reliable in every launch path tested, including a spool dir, because the
#      sourced file's path is known regardless of cwd.
#   2. SLURM_SUBMIT_DIR, but ONLY if it actually looks like this project (has a
#      pyproject.toml). Used as a cross-check/override, never blindly.
#
# Why SLURM_SUBMIT_DIR is not preferred, contra the general cluster guidance
# ("use SLURM_SUBMIT_DIR, not BASH_SOURCE, which resolves to the spool dir"):
# that advice assumes you ran `sbatch` FROM the project. `myxcel submit-job`
# does not -- measured on job 20640802, SLURM_SUBMIT_DIR was
# /orcd/home/002/maarxaru, i.e. the home directory, and a job that did
# `cd "$SLURM_SUBMIT_DIR"` then failed with "can't open
# .../scripts/benchmarks/...py: No such file or directory". Trusting it here
# would have pointed BTH_PROJECT_ROOT at the home dir -- a different wrong
# answer than the hardcoded path this replaced, but still wrong.
_BTH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/pyproject.toml" ]]; then
  _BTH_ROOT="${SLURM_SUBMIT_DIR}"
fi

export BTH_PROJECT_ROOT="${_BTH_ROOT}"
export BTH_WORKSPACE_ROOT="${_BTH_ROOT}"

# $HOME, not a literal home directory: the catalog lives under the running
# user's home on whichever machine the job lands on.
export BTH_CATALOG_DIR="${BTH_CATALOG_DIR:-${HOME}/.bth/catalog}"

unset _BTH_ROOT
