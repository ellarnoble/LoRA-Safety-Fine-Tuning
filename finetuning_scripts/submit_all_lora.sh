#!/bin/bash
# Submit all 36 LoRA fine-tuning jobs (3 models x 4 ranks x 3 placements).
#
# Usage: ./submit_all_lora.sh [--dry-run]
#   --dry-run   print the sbatch commands instead of submitting them

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

MODELS=(mistral7b qwen7b falcon7b)
RANKS=(1 4 16 64)
PLACEMENTS=(early middle late)

for MODEL in "${MODELS[@]}"; do
    for RANK in "${RANKS[@]}"; do
        for PLACEMENT in "${PLACEMENTS[@]}"; do
            JOB_NAME="${MODEL}_r${RANK}_${PLACEMENT}"
            CMD=(sbatch --job-name="$JOB_NAME" \
                --export=ALL,MODEL="$MODEL",RANK="$RANK",PLACEMENT="$PLACEMENT" \
                slurm/train_lora.sbatch)
            if [[ "$DRY_RUN" == "1" ]]; then
                echo "${CMD[@]}"
            else
                "${CMD[@]}"
            fi
        done
    done
done
