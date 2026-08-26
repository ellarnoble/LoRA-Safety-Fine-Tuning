#!/bin/bash --login
#SBATCH --job-name=shapley_mistral_all3
#SBATCH -p gpuA40GB
#SBATCH -G 3
#SBATCH -n 4
#SBATCH -t 4-00:00:00
#SBATCH -o job_%j.out
#SBATCH -e job_%j.err

echo "Job started on $(hostname)"

# cd to this script's own directory (attribution_analysis/) so it works
# regardless of where sbatch was invoked from.
cd "$(dirname "$0")"

# Activate the venv named by config.VENV_PATH
REPO_ROOT="$(cd .. && pwd)"
VENV_PATH=$(PYTHONPATH="$REPO_ROOT" python3 -c "import config; print(config.VENV_PATH)")
source "$VENV_PATH/bin/activate"

CUDA_VISIBLE_DEVICES=0 python owen_shapley_attribution.py r64_middle > shapley_r64_middle.log 2>&1 &
PID1=$!
CUDA_VISIBLE_DEVICES=1 python owen_shapley_attribution.py full > shapley_full.log 2>&1 &
PID2=$!
CUDA_VISIBLE_DEVICES=2 python owen_shapley_attribution.py r1_late > shapley_r1_late.log 2>&1 &
PID3=$!

wait $PID1; echo "r64_middle condition exited with status $?"
wait $PID2; echo "full FT condition exited with status $?"
wait $PID3; echo "r1_late condition exited with status $?"

echo "All three conditions finished."
