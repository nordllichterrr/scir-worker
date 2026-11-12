
import json
from pathlib import Path
from .metrics import coverage, intra_list_diversity, compute_item_popularity
from .utils import load_json, save_json, logger
import numpy as np
import csv

RESULTS_DIR = Path("results")
EXPERIMENTS_DIR = Path("experiments")

def load_latest_results():
    files = sorted(RESULTS_DIR.glob("run_seed_*.json"))
    if not files:
      
        f = RESULTS_DIR / "demo_rankings.json"
        if f.exists():
            return [f]
        return []
    return files

def aggregate_and_write_csv():
    files = load_latest_results()
    rows = []
    for f in files:
        j = load_json(f)
        results = j.get("results") or j
     
        rec_lists = [[r["item_id"] for r in results[str(uid)]] for uid in range(len(results))]
        cov = coverage(rec_lists, j.get("meta", {}).get("n_items", 49157))

        rows.append({
            "file": str(f.name),
            "n_users": len(results),
            "coverage": cov
        })

    out = EXPERIMENTS_DIR / "aggregated_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = rows[0].keys() if rows else ["file","n_users","coverage"]
    with open(out, "w", newline="", encoding="utf-8") as csvf:
        w = csv.DictWriter(csvf, fieldnames=list(keys))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    logger.info("Wrote aggregated CSV to %s", out)
    return out

if __name__ == "__main__":
    aggregate_and_write_csv()
