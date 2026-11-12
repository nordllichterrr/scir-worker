from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

try:
    from .plot_style import apply_paper_style, save_fig_paper, AXES_FONTSIZE, TITLE_FONTSIZE
except ImportError:
    from plot_style import apply_paper_style, save_fig_paper, AXES_FONTSIZE, TITLE_FONTSIZE


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


PROJECT_ROOT = _project_root()
RUNS_JSONL = PROJECT_ROOT / "experiments" / "runs.jsonl"
RESULTS_DIR = PROJECT_ROOT / "results"
GT_PATH = PROJECT_ROOT / "experiments" / "ground_truth.json"
OUT_DIR = PROJECT_ROOT / "experiments" / "score_calibration"


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


def _load_gt() -> Dict[str, List[str]]:
    if not GT_PATH.exists():
        return {}
    with open(GT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("data", data)


def _load_results(run_id: str) -> Dict[str, Any]:
    path = RESULTS_DIR / f"{run_id}.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_retrieval_scores(results_data: Dict[str, Any], gt: Dict[str, List[str]]) -> Dict[str, Dict[str, List[float]]]:
    out = defaultdict(lambda: defaultdict(list))
    
    reranker_scores = results_data.get("reranker_scores", {})
    candidate_pools = results_data.get("candidate_pools", {})
    
    for uid, item_scores in reranker_scores.items():
        uid_str = str(uid)
        gt_items = set(str(x) for x in gt.get(uid_str, []))
        candidates = candidate_pools.get(uid_str, [])
        if not candidates:
            continue
        
        
        cand_set = set(str(c) for c in candidates)
        if gt_items and not (gt_items & cand_set):
            continue
        
        for item_id, score in item_scores.items():
            out["reranker"][uid_str].append(float(score))
    
    
    return dict(out)


def _analyze_score_distributions(scores_by_source: Dict[str, Dict[str, List[float]]]) -> Dict[str, Any]:

    analysis = {}
    
    for source, user_scores in scores_by_source.items():
        all_scores = []
        for uid, scores in user_scores.items():
            all_scores.extend(scores)
        
        if not all_scores:
            continue
        
        arr = np.array(all_scores)
        analysis[source] = {
            "n_samples": len(all_scores),
            "n_users": len(user_scores),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p95": float(np.percentile(arr, 95)),
        }
    
    return analysis


def _check_sorting_preservation(
    results_data: Dict[str, Any],
    gt: Dict[str, List[str]],
) -> Dict[str, Any]:

    reranker_scores = results_data.get("reranker_scores", {})
    candidate_pools = results_data.get("candidate_pools", {})
    results = results_data.get("results", {})
    
    if not reranker_scores or not results:
        return {"status": "insufficient_data"}
    
    n_users_checked = 0
    n_users_correct_sorting = 0
    n_users_top1_hit = 0
    
    per_user_metrics = []
    
    for uid, item_scores in reranker_scores.items():
        uid_str = str(uid)
        gt_items = set(str(x) for x in gt.get(uid_str, []))
        candidates = candidate_pools.get(uid_str, [])
        final_recs = results.get(uid_str, [])
        
        if not gt_items or not candidates or not final_recs:
            continue
        
        cand_set = set(str(c) for c in candidates)
        if not (gt_items & cand_set):
            continue
        
        n_users_checked += 1
        

        scored_items = [(str(iid), float(score)) for iid, score in item_scores.items()]
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        if scored_items and str(scored_items[0][0]) in gt_items:
            n_users_top1_hit += 1
        
        relevant_scores = [s for iid, s in scored_items if str(iid) in gt_items]
        irrelevant_scores = [s for iid, s in scored_items if str(iid) not in gt_items]
        
        if relevant_scores and irrelevant_scores:
            correct_pairs = 0
            total_pairs = 0
            for r_score in relevant_scores:
                for irr_score in irrelevant_scores:
                    total_pairs += 1
                    if r_score > irr_score:
                        correct_pairs += 1
            
            pairwise_acc = correct_pairs / total_pairs if total_pairs > 0 else 0.0
            
            if pairwise_acc > 0.5: 
                n_users_correct_sorting += 1
            
            per_user_metrics.append({
                "user_id": uid_str,
                "pairwise_accuracy": pairwise_acc,
                "n_relevant": len(relevant_scores),
                "n_irrelevant": len(irrelevant_scores),
                "top1_hit": 1 if (scored_items and str(scored_items[0][0]) in gt_items) else 0,
            })
    
    return {
        "n_users_checked": n_users_checked,
        "n_users_correct_sorting": n_users_correct_sorting,
        "sorting_preservation_rate": n_users_correct_sorting / n_users_checked if n_users_checked > 0 else 0.0,
        "top1_hit_rate": n_users_top1_hit / n_users_checked if n_users_checked > 0 else 0.0,
        "per_user_metrics": per_user_metrics[:100], 
    }


def _check_calibration(
    results_data: Dict[str, Any],
    gt: Dict[str, List[str]],
) -> Dict[str, Any]:
    reranker_scores = results_data.get("reranker_scores", {})
    candidate_pools = results_data.get("candidate_pools", {})
    
    if not reranker_scores:
        return {"status": "insufficient_data"}
    
    all_scores = []
    all_labels = []
    
    for uid, item_scores in reranker_scores.items():
        uid_str = str(uid)
        gt_items = set(str(x) for x in gt.get(uid_str, []))
        candidates = candidate_pools.get(uid_str, [])
        
        if not candidates:
            continue
        
        cand_set = set(str(c) for c in candidates)
        if gt_items and not (gt_items & cand_set):
            continue
        
        for item_id, score in item_scores.items():
            item_id_str = str(item_id)
            all_scores.append(float(score))
            all_labels.append(1.0 if item_id_str in gt_items else 0.0)
    
    if not all_scores or sum(all_labels) == 0:
        return {"status": "insufficient_data"}
    
    scores_arr = np.array(all_scores)
    labels_arr = np.array(all_labels)
    
    def sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))
    
    probs = sigmoid(scores_arr)
    
    brier = float(np.mean((probs - labels_arr) ** 2))
 
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.clip(
        np.searchsorted(bin_edges[1:-1], probs, side="right"), 0, n_bins - 1
    )
    ece = 0.0
    n = len(probs)
    for b in range(n_bins):
        mask = bin_indices == b
        if not np.any(mask):
            continue
        acc = float(np.mean(labels_arr[mask]))
        conf = float(np.mean(probs[mask]))
        weight = float(np.sum(mask)) / n
        ece += weight * abs(acc - conf)
    
    try:
        if len(np.unique(labels_arr)) > 1:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(labels_arr, scores_arr))
        else:
            auc = None
    except ImportError:
        auc = None
        if len(np.unique(labels_arr)) > 1:
            sorted_indices = np.argsort(scores_arr)[::-1]
            sorted_labels = labels_arr[sorted_indices]
            n_pos = int(np.sum(sorted_labels))
            n_neg = len(sorted_labels) - n_pos
            if n_pos > 0 and n_neg > 0:
                ranks = np.arange(1, len(sorted_labels) + 1)
                sum_ranks_pos = float(np.sum(ranks[sorted_labels == 1]))
                auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    
    return {
        "brier_score": brier,
        "ece": ece,
        "auc": auc,
        "n_samples": len(all_scores),
        "positive_rate": float(np.mean(labels_arr)),
    }


def plot_score_distributions(
    analysis: Dict[str, Any],
    out_dir: Path,
) -> None:
    apply_paper_style()
    
    if not analysis:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    sources = list(analysis.keys())
    if not sources:
        return
    
    ax = axes[0]
    for source in sources:
        pass
    ax.set_title("Score Distributions (placeholder)", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Score", fontsize=AXES_FONTSIZE)
    ax.set_ylabel("Density", fontsize=AXES_FONTSIZE)
    ax.grid(alpha=0.3)

    ax = axes[1]
    if sources:
        stats_names = ["mean", "std", "p50"]
        x = np.arange(len(stats_names))
        width = 0.8 / len(sources)
        for i, source in enumerate(sources):
            vals = [analysis[source].get(s, 0) for s in stats_names]
            offset = (i - len(sources) / 2) * width + width / 2
            ax.bar(x + offset, vals, width, label=source)
        ax.set_xticks(x)
        ax.set_xticklabels(stats_names, fontsize=AXES_FONTSIZE)
        ax.set_ylabel("Value", fontsize=AXES_FONTSIZE)
        ax.set_title("Score Statistics Comparison", fontsize=TITLE_FONTSIZE)
        ax.legend(fontsize=AXES_FONTSIZE - 1)
        ax.grid(alpha=0.3, axis="y")
    
    ax = axes[2]
    if sources:
        percentiles = ["p25", "p50", "p75", "p95"]
        x = np.arange(len(percentiles))
        width = 0.8 / len(sources)
        for i, source in enumerate(sources):
            vals = [analysis[source].get(p, 0) for p in percentiles]
            offset = (i - len(sources) / 2) * width + width / 2
            ax.bar(x + offset, vals, width, label=source)
        ax.set_xticks(x)
        ax.set_xticklabels(percentiles, fontsize=AXES_FONTSIZE)
        ax.set_ylabel("Score", fontsize=AXES_FONTSIZE)
        ax.set_title("Score Percentiles", fontsize=TITLE_FONTSIZE)
        ax.legend(fontsize=AXES_FONTSIZE - 1)
        ax.grid(alpha=0.3, axis="y")
    
    axes[3].axis("off")
    
    plt.tight_layout()
    save_fig_paper(out_dir / "score_distributions_comparison")
    plt.close()


def main():
    runs = _read_jsonl(RUNS_JSONL)
    if not runs:
        print(f"No runs found in {RUNS_JSONL}. Run experiments first.")
        return
    
    gt = _load_gt()
    if not gt:
        print(f"Ground truth not found at {GT_PATH}. Run create_splits first.")
        return
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    report = {
        "runs_analyzed": [],
        "summary": {},
    }
    
    all_scores_by_source = defaultdict(lambda: defaultdict(list))
    
    for run in runs:
        cfg = run.get("config", {}) or {}
        run_id = run.get("run_id", "")
       
        if not cfg.get("use_reranker") or cfg.get("baseline"):
            continue
        
        results_data = _load_results(run_id)
        if not results_data:
            continue

        scores_by_source = _extract_retrieval_scores(results_data, gt)
        if not scores_by_source:
            continue

        for source, user_scores in scores_by_source.items():
            for uid, scores in user_scores.items():
                all_scores_by_source[source][uid].extend(scores)
        
        run_analysis = {
            "run_id": run_id,
            "config": {
                "retrieval_mode": cfg.get("retrieval_mode", "ann"),
                "candidate_pool_size": cfg.get("candidate_pool_size", 1000),
            },
            "score_distributions": _analyze_score_distributions(scores_by_source),
            "sorting_preservation": _check_sorting_preservation(results_data, gt),
            "calibration": _check_calibration(results_data, gt),
        }
        
        report["runs_analyzed"].append(run_analysis)
    
    if report["runs_analyzed"]:
        all_sorting_rates = [
            r["sorting_preservation"].get("sorting_preservation_rate", 0)
            for r in report["runs_analyzed"]
            if r["sorting_preservation"].get("status") != "insufficient_data"
        ]
        all_top1_rates = [
            r["sorting_preservation"].get("top1_hit_rate", 0)
            for r in report["runs_analyzed"]
            if r["sorting_preservation"].get("status") != "insufficient_data"
        ]
        all_brier = [
            r["calibration"].get("brier_score", 1.0)
            for r in report["runs_analyzed"]
            if r["calibration"].get("status") != "insufficient_data"
        ]
        all_ece = [
            r["calibration"].get("ece", 1.0)
            for r in report["runs_analyzed"]
            if r["calibration"].get("status") != "insufficient_data"
        ]
        
        report["summary"] = {
            "n_runs": len(report["runs_analyzed"]),
            "mean_sorting_preservation_rate": float(np.mean(all_sorting_rates)) if all_sorting_rates else None,
            "mean_top1_hit_rate": float(np.mean(all_top1_rates)) if all_top1_rates else None,
            "mean_brier_score": float(np.mean(all_brier)) if all_brier else None,
            "mean_ece": float(np.mean(all_ece)) if all_ece else None,
        }
    

    global_distributions = _analyze_score_distributions(dict(all_scores_by_source))
    
    with open(OUT_DIR / "score_calibration_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "report": report,
            "global_distributions": global_distributions,
        }, f, indent=2, ensure_ascii=False)
    
    md_lines = [
        "# Score Calibration and Comparability Analysis",
        "",
        "Проверка 'смысла скоринга': сопоставимость, калибровка, сохранение сортировки.",
        "",
        "## Summary",
        "",
    ]
    
    if report["summary"]:
        summ = report["summary"]
        md_lines.extend([
            f"- **Runs analyzed:** {summ.get('n_runs', 0)}",
            f"- **Mean sorting preservation rate:** {summ.get('mean_sorting_preservation_rate', 0):.2%}",
            f"- **Mean top-1 hit rate:** {summ.get('mean_top1_hit_rate', 0):.2%}",
            f"- **Mean Brier score:** {summ.get('mean_brier_score', 0):.4f} (lower = better)",
            f"- **Mean ECE:** {summ.get('mean_ece', 0):.4f} (lower = better)",
            "",
        ])
    
    md_lines.extend([
        "## Score Distributions",
        "",
    ])
    
    if global_distributions:
        md_lines.append("| Source | Mean | Std | Min | Max | P25 | P50 | P75 | P95 |")
        md_lines.append("|--------|------|-----|-----|-----|-----|-----|-----|-----|")
        for source, stats_dict in global_distributions.items():
            md_lines.append(
                f"| {source} | {stats_dict.get('mean', 0):.4f} | "
                f"{stats_dict.get('std', 0):.4f} | {stats_dict.get('min', 0):.4f} | "
                f"{stats_dict.get('max', 0):.4f} | {stats_dict.get('p25', 0):.4f} | "
                f"{stats_dict.get('p50', 0):.4f} | {stats_dict.get('p75', 0):.4f} | "
                f"{stats_dict.get('p95', 0):.4f} |"
            )
        md_lines.append("")
    
    md_lines.extend([
        "## Per-Run Analysis",
        "",
    ])
    
    for run_analysis in report["runs_analyzed"][:10]:
        md_lines.extend([
            f"### {run_analysis['run_id']}",
            "",
            f"- **Retrieval mode:** {run_analysis['config'].get('retrieval_mode', 'unknown')}",
            f"- **Sorting preservation:** {run_analysis['sorting_preservation'].get('sorting_preservation_rate', 0):.2%}",
            f"- **Top-1 hit rate:** {run_analysis['sorting_preservation'].get('top1_hit_rate', 0):.2%}",
            f"- **Brier score:** {run_analysis['calibration'].get('brier_score', 0):.4f}",
            f"- **ECE:** {run_analysis['calibration'].get('ece', 0):.4f}",
            "",
        ])
    
    md_lines.extend([
        "## Conclusions",
        "",
        "1. **Sorting preservation:** Проверка, что relevant items имеют более высокие скоры, чем irrelevant.",
        "2. **Calibration:** Brier score и ECE показывают, насколько хорошо скоры предсказывают relevance.",
        "3. **Score distributions:** Сравнение распределений скоров из разных источников (если доступны).",
        "",
    ])
    
    with open(OUT_DIR / "score_calibration_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    
    plot_score_distributions(global_distributions, OUT_DIR)
    
    print(f"Score calibration analysis saved to {OUT_DIR}")
    print("Files:")
    print("  - score_calibration_report.{md,json}")
    print("  - score_distributions_comparison.{pdf,svg,png}")


if __name__ == "__main__":
    main()
