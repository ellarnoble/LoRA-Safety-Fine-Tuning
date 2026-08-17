#!/bin/bash
# Submit all 3 full-parameter fine-tuning jobs (one per model).
#
# Usage: ./submit_all_full.sh [--dry-run]

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

MODELS=(mistral7b qwen7b falcon7b)

for MODEL in "${MODELS[@]}"; do
    JOB_NAME="${MODEL}_full"
    CMD=(sbatch --job-name="$JOB_NAME" --export=ALL,MODEL="$MODEL" slurm/train_full.sbatch)
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "${CMD[@]}"
    else
        "${CMD[@]}"
    fi
done
