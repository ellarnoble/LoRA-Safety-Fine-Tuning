#!/bin/bash
# Submit LlamaGuard evaluation jobs for all 42 conditions' generate.py outputs.
#
# Usage: ./submit_all_evaluate.sh [--dry-run]
#
# Run this AFTER the corresponding generate.py jobs have written their
# *_responses.jsonl files. Input/output/debug paths are derived from
# config.py (config.generate_output_file / config.safety_eval_paths) --
# nothing to configure here beyond editing config.py itself.

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
    local CMD=(sbatch --job-name="$JOB_NAME" "${DEPEND_ARGS[@]}" --export="ALL,$1" slurm/evaluate.sbatch)
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "${CMD[@]}"
    else
        "${CMD[@]}"
    fi
}

for MODEL in "${MODELS[@]}"; do
    submit "${MODEL}_eval_baseline" "MODEL=$MODEL,CONDITION=baseline"
    submit "${MODEL}_eval_full" "MODEL=$MODEL,CONDITION=full"
    for RANK in "${RANKS[@]}"; do
        for PLACEMENT in "${PLACEMENTS[@]}"; do
            submit "${MODEL}_eval_r${RANK}_${PLACEMENT}" \
                "MODEL=$MODEL,CONDITION=lora,RANK=$RANK,PLACEMENT=$PLACEMENT"
        done
    done
done
