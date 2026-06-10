#!/bin/bash
#SBATCH --job-name=pottsmpnn_recapture
#SBATCH --array=0-9
#SBATCH --output=outputs/logs/slurm/recapture_%A_%a.out
#SBATCH --error=outputs/logs/slurm/recapture_%A_%a.err
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=short

_PROJ="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
_MISTYPOTTS="${_PROJ}/../mistypotts"
_PDB="${_MISTYPOTTS}/tests/data/pdb/1crn.pdb"
_CKPT_ROOT="${_MISTYPOTTS}/.tmp/PottsMPNN"
_OUT="${_PROJ}/outputs/weights/pottsmpnn"
mkdir -p "${_OUT}" "${_PROJ}/outputs/logs/slurm"

# Checkpoint manifest: 10 variants
# Format: "checkpoint_relative_path|output_stem"
CHECKPOINTS=(
  "vanilla_model_weights/pottsmpnn_20.pt|pottsmpnn_vanilla_20"
  "vanilla_model_weights/pottsmpnn_30.pt|pottsmpnn_vanilla_30"
  "vanilla_model_weights/pottsmpnn_msa_20.pt|pottsmpnn_vanilla_msa_20"
  "soluble_model_weights/sol_pottsmpnn_20.pt|pottsmpnn_sol_20"
  "soluble_model_weights/sol_pottsmpnn_30.pt|pottsmpnn_sol_30"
  "soluble_model_weights/sol_pottsmpnn_msa_20.pt|pottsmpnn_sol_msa_20"
  "ca_model_weights/v_48_002.pt|pottsmpnn_ca_002"
  "ca_model_weights/v_48_010.pt|pottsmpnn_ca_010"
  "ca_model_weights/v_48_020.pt|pottsmpnn_ca_020"
  "ft_model_weights/potts_ft.pt|pottsmpnn_ft"
)

IFS='|' read -r CKPT_REL STEM <<< "${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}"
CKPT="${_CKPT_ROOT}/${CKPT_REL}"
OUT_FILE="${_OUT}/${STEM}.eqx.zst"

echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID}: ${STEM}"
echo "  checkpoint: ${CKPT}"
echo "  output:     ${OUT_FILE}"

_NODE="${SLURM_JOB_NODELIST:-$(hostname -s)}"
if [[ "${_NODE}" == *node4007* ]] || [[ "${_NODE}" == *node4008* ]]; then
    export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_shard_autotuning=false"
fi

uv run python "${_PROJ}/scripts/recapture/pottsmpnn_to_eqx.py" \
  --checkpoint "${CKPT}" \
  --pdb "${_PDB}" \
  --pottsmpnn-root "${_CKPT_ROOT}" \
  --out "${OUT_FILE}" \
  --device cpu

echo "[$(date)] Done: exit $?"
