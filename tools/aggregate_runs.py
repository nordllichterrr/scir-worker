import json
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any
import numpy as np

RUNS_LOG_PATH = Path("experiments") / "runs.jsonl"
OUT_CSV = Path("experiments") / "aggregated_results.csv"
OUT_JSON = Path("experiments") / "aggregated_results.json"


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


def get_config_key(run: Dict[str, Any]) -> str:
    config = run.get("config", {})
    baseline = config.get("baseline")
    use_reranker = config.get("use_reranker", False)
    pool_size = config.get("candidate_pool_size", 1000)
    
    if baseline:
        return f"{baseline}_no_reranker"
    elif use_reranker:
        return f"ours_with_reranker_pool{pool_size}"
    else:
        return f"candidates_only_pool{pool_size}"


def aggregate_runs() -> Dict[str, Any]:
    runs = load_runs()
    
    grouped = defaultdict(list)
    for run in runs:
        key = get_config_key(run)
        grouped[key].append(run)
    
    aggregated = {}
    
    for config_key, config_runs in grouped.items():
        if len(config_runs) == 0:
            continue
        
        hr10_vals = []
        ndcg10_vals = []
        recall50_vals = []
        recall200_vals = []
        recall1000_vals = []
        unique_top1_vals = []
        coverage10_vals = []
        exposure_gini_vals = []
        exposure_entropy_vals = []
        novelty_vals = []
        serendipity10_vals = []
        top10_share_vals = []
        long_tail_coverage_vals = []
        
        first_config = config_runs[0].get("config", {})
        
        for run in config_runs:
            metrics = run.get("metrics", {})
            diagnostics = run.get("diagnostics", {})
            
            hr10 = metrics.get("hr@10", {})
            if isinstance(hr10, dict):
                hr10_vals.append(hr10.get("mean", 0.0))
            else:
                hr10_vals.append(float(hr10) if hr10 is not None else 0.0)
            
            ndcg10 = metrics.get("ndcg@10", {})
            if isinstance(ndcg10, dict):
                ndcg10_vals.append(ndcg10.get("mean", 0.0))
            else:
                ndcg10_vals.append(float(ndcg10) if ndcg10 is not None else 0.0)
            
            recall50_vals.append(diagnostics.get("recall@50"))
            recall200_vals.append(diagnostics.get("recall@200"))
            recall1000_vals.append(diagnostics.get("recall@1000"))
            
            unique_top1_vals.append(diagnostics.get("unique_top1"))
            coverage10_vals.append(diagnostics.get("catalog_coverage_at_10"))
            exposure_gini_vals.append(diagnostics.get("exposure_gini"))
            exposure_entropy_vals.append(diagnostics.get("exposure_entropy"))
            novelty_vals.append(diagnostics.get("mean_self_information_novelty"))
            serendipity10_vals.append(diagnostics.get("serendipity@10"))
            top10_share_vals.append(diagnostics.get("top10_share"))
            long_tail_coverage_vals.append(diagnostics.get("long_tail_coverage"))
        
        def compute_stats(vals):
            filtered = [v for v in vals if v is not None]
            if not filtered:
                return None, None
            return float(np.mean(filtered)), float(np.std(filtered))
        
        hr10_mean, hr10_std = compute_stats(hr10_vals)
        ndcg10_mean, ndcg10_std = compute_stats(ndcg10_vals)
        recall50_mean, recall50_std = compute_stats(recall50_vals)
        recall200_mean, recall200_std = compute_stats(recall200_vals)
        recall1000_mean, recall1000_std = compute_stats(recall1000_vals)
        unique_top1_mean, unique_top1_std = compute_stats(unique_top1_vals)
        coverage10_mean, coverage10_std = compute_stats(coverage10_vals)
        exposure_gini_mean, exposure_gini_std = compute_stats(exposure_gini_vals)
        exposure_entropy_mean, exposure_entropy_std = compute_stats(exposure_entropy_vals)
        novelty_mean, novelty_std = compute_stats(novelty_vals)
        serendipity10_mean, serendipity10_std = compute_stats(serendipity10_vals)
        top10_share_mean, top10_share_std = compute_stats(top10_share_vals)
        long_tail_mean, long_tail_std = compute_stats(long_tail_coverage_vals)
        
        baseline = first_config.get("baseline")
        use_reranker = first_config.get("use_reranker", False)
        
        if baseline:
            model_name = baseline
        elif use_reranker:
            model_name = "ours_with_reranker"
        else:
            model_name = "candidates_only"
        
        aggregated[config_key] = {
            "model": model_name,
            "n_seeds": len(config_runs),
            "config": {
                "baseline": baseline,
                "use_reranker": use_reranker,
                "candidate_pool_size": first_config.get("candidate_pool_size", 1000),
                "topk": first_config.get("topk", 10)
            },
            "metrics": {
                "hr@10": {
                    "mean": hr10_mean,
                    "std": hr10_std,
                    "values": hr10_vals
                },
                "ndcg@10": {
                    "mean": ndcg10_mean,
                    "std": ndcg10_std,
                    "values": ndcg10_vals
                }
            },
            "diagnostics": {
                "recall@50": {
                    "mean": recall50_mean,
                    "std": recall50_std
                },
                "recall@200": {
                    "mean": recall200_mean,
                    "std": recall200_std
                },
                "recall@1000": {
                    "mean": recall1000_mean,
                    "std": recall1000_std
                },
                "unique_top1": {
                    "mean": unique_top1_mean,
                    "std": unique_top1_std
                },
                "catalog_coverage_at_10": {
                    "mean": coverage10_mean,
                    "std": coverage10_std
                },
                "exposure_gini": {
                    "mean": exposure_gini_mean,
                    "std": exposure_gini_std
                },
                "exposure_entropy": {
                    "mean": exposure_entropy_mean,
                    "std": exposure_entropy_std
                },
                "mean_self_information_novelty": {
                    "mean": novelty_mean,
                    "std": novelty_std
                },
                "serendipity@10": {
                    "mean": serendipity10_mean,
                    "std": serendipity10_std
                },
                "top10_share": {
                    "mean": top10_share_mean,
                    "std": top10_share_std
                },
                "long_tail_coverage": {
                    "mean": long_tail_mean,
                    "std": long_tail_std
                },
            }
        }
    
    return aggregated


def save_results(aggregated: Dict[str, Any]):
    rows = []
    for config_key, data in aggregated.items():
        row = {
            "model": data["model"],
            "n_seeds": data["n_seeds"],
            "baseline": data["config"]["baseline"] or "",
            "use_reranker": data["config"]["use_reranker"],
            "candidate_pool_size": data["config"]["candidate_pool_size"],
            "hr@10_mean": data["metrics"]["hr@10"]["mean"],
            "hr@10_std": data["metrics"]["hr@10"]["std"],
            "ndcg@10_mean": data["metrics"]["ndcg@10"]["mean"],
            "ndcg@10_std": data["metrics"]["ndcg@10"]["std"],
            "recall@50_mean": data["diagnostics"]["recall@50"]["mean"],
            "recall@50_std": data["diagnostics"]["recall@50"]["std"],
            "recall@200_mean": data["diagnostics"]["recall@200"]["mean"],
            "recall@200_std": data["diagnostics"]["recall@200"]["std"],
            "recall@1000_mean": data["diagnostics"]["recall@1000"]["mean"],
            "recall@1000_std": data["diagnostics"]["recall@1000"]["std"],
            "unique_top1_mean": data["diagnostics"]["unique_top1"]["mean"],
            "unique_top1_std": data["diagnostics"]["unique_top1"]["std"],
            "coverage_at_10_mean": data["diagnostics"]["catalog_coverage_at_10"]["mean"],
            "coverage_at_10_std": data["diagnostics"]["catalog_coverage_at_10"]["std"],
            "exposure_gini_mean": data["diagnostics"]["exposure_gini"]["mean"],
            "exposure_gini_std": data["diagnostics"]["exposure_gini"]["std"],
            "exposure_entropy_mean": data["diagnostics"]["exposure_entropy"]["mean"],
            "exposure_entropy_std": data["diagnostics"]["exposure_entropy"]["std"],
            "novelty_self_info_mean": data["diagnostics"]["mean_self_information_novelty"]["mean"],
            "novelty_self_info_std": data["diagnostics"]["mean_self_information_novelty"]["std"],
            "serendipity@10_mean": data["diagnostics"]["serendipity@10"]["mean"],
            "serendipity@10_std": data["diagnostics"]["serendipity@10"]["std"],
            "top10_share_mean": data["diagnostics"]["top10_share"]["mean"],
            "top10_share_std": data["diagnostics"]["top10_share"]["std"],
            "long_tail_coverage_mean": data["diagnostics"]["long_tail_coverage"]["mean"],
            "long_tail_coverage_std": data["diagnostics"]["long_tail_coverage"]["std"],
        }
        rows.append(row)
    
    rows.sort(key=lambda x: (x["use_reranker"], x["model"]))
    
    if rows:
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved aggregated results to {OUT_CSV}")
    
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2, ensure_ascii=False)
    print(f"Saved aggregated results to {OUT_JSON}")


def main():
    print("Aggregating runs from runs.jsonl...")
    aggregated = aggregate_runs()
    
    print(f"\nFound {len(aggregated)} unique configurations:")
    for key, data in aggregated.items():
        print(f"  {key}: {data['n_seeds']} seeds")
    
    save_results(aggregated)
    print("\nDone!")


if __name__ == "__main__":
    main()
