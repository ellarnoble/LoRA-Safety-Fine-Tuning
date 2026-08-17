#!/bin/bash --login
#SBATCH --job-name=lora_all
#SBATCH -p gpuA40GB
#SBATCH -G 1
#SBATCH -n 1
#SBATCH -t 4-00:00:00
#SBATCH -o logs/lora_all_%j.out
#SBATCH -e logs/lora_all_%j.err

# Runs all 36 LoRA fine-tuning conditions (3 models x 4 ranks x 3
# placements), one after another, by calling LoRA_finetune.py directly.
# Already-trained conditions (adapter_model.safetensors already present)
# are skipped, so a requeued/resubmitted job picks up where it left off.
#
# Usage:
#   sbatch submit_all_lora.sh        
#   ./submit_all_lora.sh --dry-run  

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

for MODEL in "${MODELS[@]}"; do
    for RANK in "${RANKS[@]}"; do
        for PLACEMENT in "${PLACEMENTS[@]}"; do
            OUT_DIR=$(python3 -c "import config; print(config.lora_output_dir('$MODEL', $RANK, '$PLACEMENT'))")

            if [[ -f "$OUT_DIR/adapter_model.safetensors" ]]; then
                echo "Skipping ${MODEL}_r${RANK}_${PLACEMENT}: already trained at $OUT_DIR"
                continue
            fi

            echo "=== ${MODEL}_r${RANK}_${PLACEMENT} ==="
            CMD=(python LoRA_finetune.py --model "$MODEL" --rank "$RANK" --placement "$PLACEMENT")
            if [[ "$DRY_RUN" == "1" ]]; then
                echo "${CMD[@]}"
            else
                "${CMD[@]}"
            fi
        done
    done
done

echo "All 36 LoRA conditions finished."
