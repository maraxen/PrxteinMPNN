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
# Resolution order:
#   1. SLURM_SUBMIT_DIR  -- set by sbatch to the submission directory, which is
#      the project root for `myxcel submit-job` and for `sbatch` run from the
#      project. Preferred because it is exactly the "pin it, don't discover it"
#      property the original comment asked for.
#   2. This file's own location, two levels up (scripts/slurm/ -> project root).
#      Covers `source`-ing outside a SLURM job, where SLURM_SUBMIT_DIR is unset.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  _BTH_ROOT="${SLURM_SUBMIT_DIR}"
else
  _BTH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

export BTH_PROJECT_ROOT="${_BTH_ROOT}"
export BTH_WORKSPACE_ROOT="${_BTH_ROOT}"

# $HOME, not a literal home directory: the catalog lives under the running
# user's home on whichever machine the job lands on.
export BTH_CATALOG_DIR="${BTH_CATALOG_DIR:-${HOME}/.bth/catalog}"

unset _BTH_ROOT
