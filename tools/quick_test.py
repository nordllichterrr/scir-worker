import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.run_all_experiments import run_baselines, run_ablation_study
from src.evaluation_config import N_QUICK_TEST_USERS

print("Running quick test: n_users=%d, seed 42 (not for paper)" % N_QUICK_TEST_USERS)
print("Paper uses n_users=%d and 10 seeds — see experiments/EVALUATION_PROTOCOL.md" % 1000)

run_baselines(n_users=N_QUICK_TEST_USERS, seeds=[42])
run_ablation_study(n_users=N_QUICK_TEST_USERS, seeds=[42])

print("\nDone! Check experiments/runs.jsonl for results")
