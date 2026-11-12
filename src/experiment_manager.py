
import time, json, os
from pathlib import Path
from .utils import logger, save_json
from .run_experiment import run_experiment  
from .metrics import coverage, intra_list_diversity, compute_item_popularity
import datetime

RESULTS_DIR = Path("results")
EXPERIMENTS_DIR = Path("experiments")
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

def save_metadata(metadata: dict, filename: str):
    p = EXPERIMENTS_DIR / filename
    save_json(p, metadata)
    logger.info("Saved metadata to %s", p)

def run_multiple_seeds(config: dict, seeds=None, n_users=None):

    from .evaluation_config import N_TEST_USERS, EVAL_SEEDS
    if n_users is None:
        n_users = N_TEST_USERS
    if seeds is None:
        seeds = EVAL_SEEDS
    logger.warning("run_multiple_seeds is deprecated. Use run_all_experiments.py instead.")
    
    summary = []
    for seed in seeds:
        t0 = time.time()
        logger.info("Running seed %s", seed)

        results = run_experiment(n_users=n_users, seed=seed, config={
            "baseline": None,
            "use_reranker": True,
            "topk": 10
        })
        elapsed = time.time() - t0

        all_rec_lists = [[r['item_id'] for r in results[uid]] for uid in results.keys() if results[uid]]
   
        cov = coverage(all_rec_lists, config.get("n_items", 49157))
        summary.append({
            "seed": seed,
            "elapsed": elapsed,
            "coverage": cov,
            "n_users": n_users,
            "date": datetime.datetime.utcnow().isoformat()
        })
        fname = f"run_seed_{seed}_{int(time.time())}.json"
        save_json(RESULTS_DIR / fname, {"results": results, "meta": {"seed": seed, "elapsed": elapsed}})
    save_metadata({"config": config, "summary": summary}, "experiment_summary.json")
    return summary
