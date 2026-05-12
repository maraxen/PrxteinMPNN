#!/bin/bash
#SBATCH --job-name=refactor-validation
#SBATCH --partition=mit_quicktest
#SBATCH --time=0:10:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=outputs/logs/slurm/%j.out
#SBATCH --error=outputs/logs/slurm/%j.err

# Refactoring validation on cluster
# Validates: StageSet, Executor, parity regression
# Partition: mit_quicktest (15min max)
# Expected runtime: ~10 min total

set -e

cd ~/projects/tev_design/prxteinmpnn

echo "=========================================="
echo "Refactoring Validation - Cluster Run"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: mit_quicktest"
echo "Time allocated: $SLURM_TIMELIMIT"
echo "=========================================="
echo ""

# Verify uv is available
uv --version
echo ""

# Run validation script
bash scripts/test_refactoring_validation.sh

echo ""
echo "=========================================="
echo "✓ Validation Complete"
echo "=========================================="
