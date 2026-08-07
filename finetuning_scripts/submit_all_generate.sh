#!/bin/bash
# Submit generation jobs for all 42 conditions
#
# Usage: ./submit_all_generate.sh [--dry-run]

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

DEPEND_ARGS=()
if [[ -n "${DEPEND:-}" ]]; then
    DEPEND_ARGS=(--dependency="$DEPEND")
fi

MODELS=(mistral7b qwen7b falcon7b)
RANKS=(1 4 16 64)
PLACEMENTS=(early middle late)

submit() {
    local JOB_NAME="$1"; shift
    local CMD=(sbatch --job-name="$JOB_NAME" "${DEPEND_ARGS[@]}" --export="ALL,$1" slurm/generate.sbatch)
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "${CMD[@]}"
    else
        "${CMD[@]}"
    fi
}

for MODEL in "${MODELS[@]}"; do
    submit "${MODEL}_gen_baseline" "MODEL=$MODEL,CONDITION=baseline"
    submit "${MODEL}_gen_full" "MODEL=$MODEL,CONDITION=full"
    for RANK in "${RANKS[@]}"; do
        for PLACEMENT in "${PLACEMENTS[@]}"; do
            submit "${MODEL}_gen_r${RANK}_${PLACEMENT}" \
                "MODEL=$MODEL,CONDITION=lora,RANK=$RANK,PLACEMENT=$PLACEMENT"
        done
    done
done
