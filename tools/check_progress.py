
import json
from pathlib import Path

runs_file = Path("experiments/runs.jsonl")
if not runs_file.exists():
    print("No runs.jsonl found. Experiments not started yet.")
    exit(0)

runs = []
with open(runs_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            try:
                runs.append(json.loads(line))
            except:
                pass

print(f"Total runs completed: {len(runs)}")
print("\nRecent runs:")
for i, r in enumerate(runs[-10:], 1):
    print(f"  {i}. {r.get('run_id', 'unknown')}")

baseline_runs = [r for r in runs if r.get('run_id', '').startswith('baseline_')]
ablation_runs = [r for r in runs if r.get('run_id', '').startswith('ablation_')]

print(f"\nBreakdown:")
print(f"  Baselines: {len(baseline_runs)}")
print(f"  Ablation: {len(ablation_runs)}")


expected_ablation = 6
expected_baselines = 9

if len(ablation_runs) > 0:
    print(f"\nAblation progress: {len(ablation_runs)}/{expected_ablation}")
if len(baseline_runs) > 0:
    print(f"Baselines progress: {len(baseline_runs)}/{expected_baselines}")
