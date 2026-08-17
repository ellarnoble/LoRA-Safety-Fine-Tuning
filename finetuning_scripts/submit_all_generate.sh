#!/bin/bash --login
#SBATCH --job-name=generate_all
#SBATCH -p gpuA40GB
#SBATCH -G 1
#SBATCH -n 1
#SBATCH -t 2-00:00:00
#SBATCH -o logs/generate_all_%j.out
#SBATCH -e logs/generate_all_%j.err

# Run this AFTER the corresponding LoRA/full-FT training jobs have produced
# their checkpoints. Conditions whose model checkpoint doesn't exist yet
# are skipped with a warning rather than failing the whole job.
# Already-generated conditions (output .jsonl already present) are also
# skipped, so a requeued/resubmitted job picks up where it left off.
#
# Usage:
#   sbatch submit_all_generate.sh        
#   ./submit_all_generate.sh --dry-run    

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
    local LABEL="${MODEL}_gen_${CONDITION}"
    local ARGS=(--model "$MODEL" --condition "$CONDITION")
    local RANK_PY="None" PLACEMENT_PY="None"
    if [[ "$CONDITION" == "lora" ]]; then
        LABEL="${MODEL}_gen_r${RANK}_${PLACEMENT}"
        ARGS+=(--rank "$RANK" --placement "$PLACEMENT")
        RANK_PY="$RANK"
        PLACEMENT_PY="'$PLACEMENT'"
    fi

    local MODEL_READY MODEL_DIR OUT_FILE
    read -r MODEL_READY MODEL_DIR OUT_FILE <<< "$(python3 -c "
import config
model_dir = config.resolve_model_dir('$MODEL', '$CONDITION', $RANK_PY, $PLACEMENT_PY)
if '$CONDITION' == 'lora':
    ready = (model_dir / 'adapter_model.safetensors').exists()
elif '$CONDITION' == 'full':
    ready = (model_dir / 'config.json').exists()
else:
    ready = model_dir.is_dir()
out_file = config.generate_output_file('$MODEL', '$CONDITION', $RANK_PY, $PLACEMENT_PY)
print(int(ready), model_dir, out_file)
")"

    if [[ "$MODEL_READY" != "1" ]]; then
        echo "WARNING: $MODEL_DIR not ready, skipping $LABEL"
        return
    fi
    if [[ -f "$OUT_FILE" ]]; then
        echo "Skipping $LABEL: already generated at $OUT_FILE"
        return
    fi

    echo "=== $LABEL ==="
    CMD=(python Generate.py "${ARGS[@]}")
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

echo "All 42 generation conditions finished."
