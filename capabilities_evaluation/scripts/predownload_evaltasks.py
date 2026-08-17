"""
Pre-download IFEval and MMLU datasets/configs into the local HF cache

Run this ONCE (with internet access) before submitting the GPU job

"""

from lm_eval.tasks import TaskManager
from lm_eval import evaluator

# This triggers lm-eval to resolve task configs and download the
# underlying HF datasets (ifeval, mmlu) into ~/.cache/huggingface/
# without actually running any model.

task_manager = TaskManager()

print("Resolving and downloading task data for: ifeval, mmlu ...")
task_dict = task_manager.load_task_or_group(["ifeval", "mmlu"])

# Force materialization of the underlying datasets by accessing them
for task_name, task_obj in task_dict.items():
    print(f"  - {task_name}: dataset loaded ({len(task_obj.dataset) if hasattr(task_obj, 'dataset') else 'group'})")

print("Done. Datasets cached under ~/.cache/huggingface/")
