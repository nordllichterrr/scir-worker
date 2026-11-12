from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

try:
    from .plot_style import apply_paper_style, save_fig_paper, AXES_FONTSIZE, TITLE_FONTSIZE
except ImportError:
    from plot_style import apply_paper_style, save_fig_paper, AXES_FONTSIZE, TITLE_FONTSIZE


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


PROJECT_ROOT = _project_root()
RUNS_JSONL = PROJECT_ROOT / "experiments" / "runs.jsonl"
SPLIT_METADATA = PROJECT_ROOT / "experiments" / "split_metadata.json"
OUT_DIR = PROJECT_ROOT / "experiments" / "hardest_cold_start"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    runs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return runs


def _load_split_metadata() -> Dict[str, Any]:
    if not SPLIT_METADATA.exists():
        return {}
    with open(SPLIT_METADATA, "r", encoding="utf-8") as f:
        return json.load(f)


def _identify_hardest_users(split_metadata: Dict[str, Any]) -> List[str]:
 
    user_meta = split_metadata.get("user_meta", {})
    scenario_to_users = split_metadata.get("scenario_to_users", {})
    train_items = set(split_metadata.get("train_items", []))
    
    new_users_set = set(scenario_to_users.get("new_users", []))
    new_items_set = set(scenario_to_users.get("new_items", []))
    
    hardest = list(new_users_set & new_items_set)
    
    return hardest


def _get_config_key(run: Dict[str, Any]) -> str:
    c = run.get("config", {})
    baseline = c.get("baseline")
    use_reranker = c.get("use_reranker", False)
    pool_size = c.get("candidate_pool_size", 1000)
    if baseline:
        return f"{baseline}_no_reranker"
    return f"ours_with_reranker_pool{pool_size}" if use_reranker else f"candidates_only_pool{pool_size}"


def _get_model_display_name(config_key: str) -> str:
    if config_key.startswith("ours_with_reranker"):
        return "Ours+Rerank"
    if config_key.startswith("candidates_only"):
        return "Candidates"
    base = config_key.replace("_no_reranker", "").replace("_", " ").title()
    return base[:15]


def _extract_segment_metrics(
    runs: List[Dict[str, Any]],
    hardest_users: List[str],
    split_metadata: Dict[str, Any],
) -> Dict[str, Dict[str, Dict[str, Any]]]:

    user_meta = split_metadata.get("user_meta", {})
    scenario_to_users = split_metadata.get("scenario_to_users", {})
    
    hardest_set = set(hardest_users)
    new_users_set = set(scenario_to_users.get("new_users", []))
    new_items_set = set(scenario_to_users.get("new_items", []))

    new_users_only = list(new_users_set - hardest_set)
    new_items_only = list(new_items_set - hardest_set)
    warm_users = [uid for uid in split_metadata.get("user_meta", {}).keys() 
                  if uid not in new_users_set and uid not in new_items_set]
    
    by_config: Dict[str, List[Dict]] = defaultdict(list)
    for run in runs:
        key = _get_config_key(run)
        by_config[key].append(run)
    
    result: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for config_key, config_runs in by_config.items():
        for run in config_runs:
            run_id = run.get("run_id", "")
            per_user_csv = PROJECT_ROOT / "experiments" / f"{run_id}_per_user.csv"
            
            if not per_user_csv.exists():
                continue
            
            import csv
            with open(per_user_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    uid = str(row.get("user", row.get("user_id", "")))
                    
                    try:
                        hr = float(row.get("hr@10", 0) or 0)
                        ndcg = float(row.get("ndcg@10", 0) or 0)
                    except (ValueError, TypeError):
                        continue
                    
                    if uid in hardest_set:
                        segment = "hardest"
                    elif uid in new_users_only:
                        segment = "new_users_only"
                    elif uid in new_items_only:
                        segment = "new_items_only"
                    elif uid in warm_users:
                        segment = "warm"
                    else:
                        continue
                    
                    result[config_key][segment]["hr"].append(hr)
                    result[config_key][segment]["ndcg"].append(ndcg)

    aggregated = {}
    for config_key, segment_dict in result.items():
        aggregated[config_key] = {}
        for segment, metric_dict in segment_dict.items():
            aggregated[config_key][segment] = {}
            for metric_name, vals in metric_dict.items():
                if vals:
                    aggregated[config_key][segment][metric_name] = {
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)) if len(vals) > 1 else 0.0,
                        "n": len(vals),
                    }
    
    return aggregated


def plot_segment_comparison(
    aggregated: Dict[str, Dict[str, Dict[str, Any]]],
    out_dir: Path,
) -> None:

    apply_paper_style()
    
    if not aggregated:
        return
    
    segments_order = ["hardest", "new_users_only", "new_items_only", "warm"]
    key_models = sorted([k for k in aggregated.keys() if "ours" in k or "candidates_only" in k or k.startswith("popularity")])[:5]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    for metric_idx, (metric_name, ax) in enumerate([("hr", axes[0]), ("ndcg", axes[1])]):
        x = np.arange(len(segments_order))
        width = 0.8 / max(len(key_models), 1)
        
        for i, model_key in enumerate(key_models):
            means = []
            stds = []
            for seg in segments_order:
                m = aggregated.get(model_key, {}).get(seg, {}).get(metric_name, {})
                means.append(m.get("mean", 0) if m else 0)
                stds.append(m.get("std", 0) if m else 0)
            
            offset = (i - len(key_models) / 2) * width + width / 2
            bars = ax.bar(x + offset, means, width, yerr=stds, capsize=2, 
                         label=_get_model_display_name(model_key))
        
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_", " ").title() for s in segments_order], 
                          fontsize=AXES_FONTSIZE, rotation=15, ha="right")
        ax.set_ylabel(f"{metric_name.upper()}@10 (mean ± std)", fontsize=AXES_FONTSIZE)
        ax.set_title(f"Hardest Cold-Start vs Other Segments: {metric_name.upper()}@10", 
                    fontsize=TITLE_FONTSIZE)
        ax.legend(fontsize=AXES_FONTSIZE - 1, loc="best", ncol=2)
        ax.grid(alpha=0.3, axis="y")
    
    plt.tight_layout()
    save_fig_paper(out_dir / "hardest_cold_start_comparison")
    plt.close()


def main():
    runs = _read_jsonl(RUNS_JSONL)
    if not runs:
        print(f"No runs found in {RUNS_JSONL}. Run experiments first.")
        return
    
    split_metadata = _load_split_metadata()
    if not split_metadata:
        print(f"Split metadata not found at {SPLIT_METADATA}. Run create_splits first.")
        return
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    hardest_users = _identify_hardest_users(split_metadata)
    
    if not hardest_users:
        print("No hardest cold-start users found (new_users AND new_items simultaneously).")
        print("This may indicate the dataset has few such users, or split configuration.")
        return
    
    print(f"Found {len(hardest_users)} hardest cold-start users (new_users AND new_items)")
    
    aggregated = _extract_segment_metrics(runs, hardest_users, split_metadata)
    
    if not aggregated:
        print("No metrics extracted. Ensure per-user CSV files exist.")
        return
    

    report_data = {
        "hardest_users_count": len(hardest_users),
        "hardest_users_sample": hardest_users[:20],  
        "metrics_by_model": aggregated,
    }
    
    with open(OUT_DIR / "hardest_cold_start_report.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    

    md_lines = [
        "# Hardest Cold-Start Analysis",
        "",
        "Анализ самого жёсткого cold-start режима: **new users + new items одновременно**.",
        "",
        f"## Summary",
        "",
        f"- **Hardest users:** {len(hardest_users)} (new_users AND new_items)",
        "",
        "## Metrics Comparison",
        "",
        "| Model | Hardest | New Users Only | New Items Only | Warm |",
        "|-------|---------|----------------|----------------|------|",
    ]
    
    key_models = sorted([k for k in aggregated.keys() if "ours" in k or "candidates_only" in k or k.startswith("popularity")])[:5]
    for model_key in key_models:
        row = [_get_model_display_name(model_key)]
        for seg in ["hardest", "new_users_only", "new_items_only", "warm"]:
            m = aggregated.get(model_key, {}).get(seg, {}).get("hr", {})
            val = m.get("mean") if m else None
            n = m.get("n", 0) if m else 0
            row.append(f"{val:.4f} (n={n})" if val is not None else "—")
        md_lines.append("| " + " | ".join(row) + " |")
    
    md_lines.extend([
        "",
        "## Conclusions",
        "",
        "1. **Hardest segment (new_users + new_items):** Самый сложный режим; показывает, где метод действительно выигрывает.",
        "2. **Comparison:** Hardest vs new_users_only vs new_items_only показывает вклад каждого фактора.",
        "3. **Warm users:** Baseline для сравнения; ожидаем, что collaborative методы здесь сильнее.",
        "",
    ])
    
    with open(OUT_DIR / "hardest_cold_start_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    

    plot_segment_comparison(aggregated, OUT_DIR)
    
    print(f"Hardest cold-start analysis saved to {OUT_DIR}")
    print("Files:")
    print("  - hardest_cold_start_comparison.{pdf,svg,png}")
    print("  - hardest_cold_start_report.{md,json}")


if __name__ == "__main__":
    main()
