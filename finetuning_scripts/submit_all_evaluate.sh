#!/bin/bash --login
#SBATCH --job-name=evaluate_all
#SBATCH -p gpuA40GB
#SBATCH -G 1
#SBATCH -n 1
#SBATCH -t 2-00:00:00
#SBATCH -o logs/evaluate_all_%j.out
#SBATCH -e logs/evaluate_all_%j.err

# Run this AFTER the corresponding generate.py jobs have written their
# *_responses.jsonl files (submit_all_generate.sh). Conditions whose
# generate.py output doesn't exist yet are skipped with a warning rather
# than failing the whole job. Already-evaluated conditions (results file
# already present) are also skipped, so a requeued/resubmitted job picks up
# where it left off.
#
# Usage:
#   sbatch submit_all_evaluate.sh          # submit as a SLURM job
#   ./submit_all_evaluate.sh --dry-run     # print the python commands, run nothing

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

# Activate the venv named by config.VENV_PATH
REPO_ROOT="$(cd .. && pwd)"
export PYTHONPATH="$REPO_ROOT"
VENV_PATH=$(python3 -c "import config; print(config.VENV_PATH)")
source "$VENV_PATH/bin/activate"

MODELS=(mistral7b qwen7b falcon7b)
RANKS=(1 4 16 64)
PLACEMENTS=(early middle late)

# Args: MODEL CONDITION [RANK PLACEMENT]
run_condition() {
    local MODEL="$1" CONDITION="$2" RANK="${3:-}" PLACEMENT="${4:-}"
    local LABEL="${MODEL}_eval_${CONDITION}"
    local ARGS=(--model "$MODEL" --condition "$CONDITION")
    local RANK_PY="None" PLACEMENT_PY="None"
    if [[ "$CONDITION" == "lora" ]]; then
        LABEL="${MODEL}_eval_r${RANK}_${PLACEMENT}"
        ARGS+=(--rank "$RANK" --placement "$PLACEMENT")
        RANK_PY="$RANK"
        PLACEMENT_PY="'$PLACEMENT'"
    fi

    local INPUT_FILE RESULTS_FILE
    read -r INPUT_FILE RESULTS_FILE <<< "$(python3 -c "
import config
input_file = config.generate_output_file('$MODEL', '$CONDITION', $RANK_PY, $PLACEMENT_PY)
results_file, _ = config.safety_eval_paths('$MODEL', '$CONDITION', $RANK_PY, $PLACEMENT_PY)
print(input_file, results_file)
")"

    if [[ ! -f "$INPUT_FILE" ]]; then
        echo "WARNING: $INPUT_FILE not found (run submit_all_generate.sh first), skipping $LABEL"
        return
    fi
    if [[ -f "$RESULTS_FILE" ]]; then
        echo "Skipping $LABEL: already evaluated at $RESULTS_FILE"
        return
    fi

    echo "=== $LABEL ==="
    CMD=(python Llamaguard_evaluate.py "${ARGS[@]}")
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "${CMD[@]}"
    else
        "${CMD[@]}"
    fi
}

for MODEL in "${MODELS[@]}"; do
    run_condition "$MODEL" baseline
    run_condition "$MODEL" full
    for RANK in "${RANKS[@]}"; do
        for PLACEMENT in "${PLACEMENTS[@]}"; do
            run_condition "$MODEL" lora "$RANK" "$PLACEMENT"
        done
    done
done

echo "All 42 evaluation conditions finished."
