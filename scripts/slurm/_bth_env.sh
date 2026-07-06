# Source from SLURM scripts: source scripts/slurm/_bth_env.sh
# Sets BTH_PROJECT_SLUG, BTH_PROJECT_ROOT, BTH_WORKSPACE_ROOT, and BTH_CATALOG_DIR so bth runs transparently in batch jobs.
set -euo pipefail
export BTH_PROJECT_SLUG="aminx"
export BTH_PROJECT_ROOT="/home/marielle/projects/aminx"
# Deterministic workspace filesystem root: in a SLURM spool dir, `git rev-parse
# --show-toplevel` may resolve to an unrelated repo (or fail), so pin it to the
# absolute project root for worktree-aware resolution (spec 260611).
export BTH_WORKSPACE_ROOT="/home/marielle/projects/aminx"
export BTH_CATALOG_DIR="/home/marielle/.bth/catalog"
