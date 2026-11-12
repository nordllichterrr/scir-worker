import json
import csv
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from collections import defaultdict

RUNS_LOG_PATH = Path("experiments") / "runs.jsonl"
OUTPUT_JSON = Path("experiments") / "master_results.json"


def load_results_data(run_id: str) -> Dict[str, Any]:

    results_path = Path("results") / f"{run_id}.json"
    if not results_path.exists():
        return {}
    
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    
    if isinstance(data, dict):
        return data
    

    return {}


def load_runs() -> List[Dict[str, Any]]:
    if not RUNS_LOG_PATH.exists():
        return []
    
    runs = []
    with open(RUNS_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    runs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return runs


def load_per_user_csv(run_id: str) -> List[Dict[str, Any]]:
    csv_path = Path("experiments") / f"{run_id}_per_user.csv"
    if not csv_path.exists():
        return []
    
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def bootstrap_ci(data: List[float], n_bootstrap: int = 1000, confidence: float = 0.95) -> tuple:
    if not data or len(data) == 0:
        return None, None
    
    n = len(data)
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return float(lower), float(upper)


def compute_per_item_exposure(runs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    item_exposure = defaultdict(lambda: {"count": 0, "users": set()})
    
    for run in runs:
        run_id = run.get("run_id")
        per_user = load_per_user_csv(run_id)
        
        for row in per_user:
            user_id = row.get("user", "")
            rec_ids_str = row.get("rec_ids", "")
            if rec_ids_str and rec_ids_str != "None":
                rec_ids = rec_ids_str.split()
                for item_id in rec_ids[:10]:
                    item_exposure[item_id]["count"] += 1
                    item_exposure[item_id]["users"].add(user_id)
    
    result = {}
    for item_id, data in item_exposure.items():
        result[item_id] = {
            "exposure_count": data["count"],
            "unique_users": len(data["users"])
        }
    
    return result


def build_master_results():
    runs = load_runs()
    

    datasets = {run.get("config", {}).get("dataset", "serendipity") for run in runs}
    dataset_str = ",".join(sorted(datasets)) if datasets else "serendipity"

    master = {
        "meta": {
            "project": "cold-start-algorithm",
            "dataset": dataset_str,
            "n_runs": len(runs)
        },
        "runs": []
    }
    
    for run in runs:
        run_id = run.get("run_id")
        config = run.get("config", {})
        metrics = run.get("metrics", {})
        diagnostics = run.get("diagnostics", {})
        
        per_user_rows = load_per_user_csv(run_id)
        
        hr10_per_user = []
        ndcg10_per_user = []
        recall50_per_user = []
        recall200_per_user = []
        recall1000_per_user = []
        
        per_user_detail = {}
        
        results_data = load_results_data(run_id)
        reranker_scores_all = results_data.get("reranker_scores", {})
        
        for row in per_user_rows:
            user_id = row.get("user", "")
            
            hr_val = row.get("hr@10", "0")
            ndcg_val = row.get("ndcg@10", "0")
            
            try:
                hr10_per_user.append(float(hr_val))
                ndcg10_per_user.append(float(ndcg_val))
            except (ValueError, TypeError):
                pass
            
            user_scores = reranker_scores_all.get(user_id, {})
            
            per_user_detail[user_id] = {
                "hr@10": float(hr_val) if hr_val else 0.0,
                "ndcg@10": float(ndcg_val) if ndcg_val else 0.0,
                "rec_ids": row.get("rec_ids", ""),
                "hits": row.get("hits", ""),
                "hit_bool": int(row.get("hit_bool", 0)),
                "reranker_scores": user_scores
            }
        
        hr10_mean = metrics.get("hr@10", {}).get("mean", 0.0) if isinstance(metrics.get("hr@10"), dict) else 0.0
        hr10_std = metrics.get("hr@10", {}).get("std", 0.0) if isinstance(metrics.get("hr@10"), dict) else 0.0
        
        ndcg10_mean = metrics.get("ndcg@10", {}).get("mean", 0.0) if isinstance(metrics.get("ndcg@10"), dict) else 0.0
        ndcg10_std = metrics.get("ndcg@10", {}).get("std", 0.0) if isinstance(metrics.get("ndcg@10"), dict) else 0.0
        
        hr10_ci_lower, hr10_ci_upper = bootstrap_ci(hr10_per_user) if hr10_per_user else (None, None)
        ndcg10_ci_lower, ndcg10_ci_upper = bootstrap_ci(ndcg10_per_user) if ndcg10_per_user else (None, None)
        
        run_record = {
            "run_id": run_id,
            "timestamp": run.get("timestamp"),
            "config": config,
            "metrics": {
                "hr@10": {
                    "mean": hr10_mean,
                    "std": hr10_std,
                    "ci_95_lower": hr10_ci_lower,
                    "ci_95_upper": hr10_ci_upper,
                    "per_user": hr10_per_user,
                    "n_users": len(hr10_per_user)
                },
                "ndcg@10": {
                    "mean": ndcg10_mean,
                    "std": ndcg10_std,
                    "ci_95_lower": ndcg10_ci_lower,
                    "ci_95_upper": ndcg10_ci_upper,
                    "per_user": ndcg10_per_user,
                    "n_users": len(ndcg10_per_user)
                }
            },
            "diagnostics": diagnostics,
            "per_user_detail": per_user_detail,
            "files": run.get("files", {})
        }
        
        master["runs"].append(run_record)
    
    item_exposure = compute_per_item_exposure(runs)
    master["per_item_exposure"] = item_exposure
    
    return master


def main():
    print("Building master results file...")
    master = build_master_results()
    
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2, ensure_ascii=False)
    
    print(f"Saved master results to {OUTPUT_JSON}")
    print(f"Total runs: {len(master['runs'])}")
    print(f"Total items exposed: {len(master['per_item_exposure'])}")
    print("Done!")


if __name__ == "__main__":
    main()
