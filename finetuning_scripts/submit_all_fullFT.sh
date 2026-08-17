#!/bin/bash --login
#SBATCH --job-name=full_ft_all
#SBATCH -p gpuA40GB
#SBATCH -G 1
#SBATCH -n 1
#SBATCH -t 4-00:00:00
#SBATCH -o logs/full_ft_all_%j.out
#SBATCH -e logs/full_ft_all_%j.err

# Runs all 3 full-parameter fine-tuning conditions (one per model), one
# after another, by calling Full_finetune.py directly. Already-trained
# conditions (config.json already present in the output dir) are skipped,
# so a requeued/resubmitted job picks up where it left off.
#
# Usage:
#   sbatch submit_all_fullFT.sh          # submit as a SLURM job
#   ./submit_all_fullFT.sh --dry-run     # print the python commands, run nothing

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

for MODEL in "${MODELS[@]}"; do
    OUT_DIR=$(python3 -c "import config; print(config.full_output_dir('$MODEL'))")

    if [[ -f "$OUT_DIR/config.json" ]]; then
        echo "Skipping ${MODEL}_full: already trained at $OUT_DIR"
        continue
    fi

    echo "=== ${MODEL}_full ==="
    CMD=(python Full_finetune.py --model "$MODEL")
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "${CMD[@]}"
    else
        "${CMD[@]}"
    fi
done

echo "All 3 full fine-tune conditions finished."
