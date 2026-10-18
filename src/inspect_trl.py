"""Print the installed TRL GRPO API surface we rely on (config fields, trainer args, reward calling)."""
import dataclasses
import inspect

import peft
import transformers
import trl
from trl import GRPOConfig, GRPOTrainer

print("trl", trl.__version__, "peft", peft.__version__, "transformers", transformers.__version__)
names = {f.name for f in dataclasses.fields(GRPOConfig)}
used = ["use_vllm", "vllm_mode", "vllm_gpu_memory_utilization", "num_generations", "max_completion_length",
        "beta", "reward_weights", "temperature", "log_completions", "lr_scheduler_type", "warmup_steps",
        "gradient_checkpointing", "save_total_limit", "report_to", "scale_rewards", "loss_type",
        "multi_objective_aggregation", "generation_batch_size", "steps_per_generation", "chat_template_kwargs"]
print("fields used/considered:", {k: k in names for k in used})
print("reward/vllm/norm fields:", sorted(n for n in names if any(s in n for s in ("reward", "vllm", "scale", "aggreg", "loss"))))
print("GRPOTrainer.__init__:", list(inspect.signature(GRPOTrainer.__init__).parameters))
src = inspect.getsource(GRPOTrainer)
for key in ("def _calculate_rewards", "reward_kwargs", "scale_rewards"):
    i = src.find(key)
    print(f"\n--- {key} ---\n" + (src[i:i + 900] if i >= 0 else "not found"))
