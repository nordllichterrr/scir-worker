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
OUT_DIR = PROJECT_ROOT / "experiments" / "multiobjective"


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


def _extract_lambda_runs(runs: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
    two_head = []
    debias = []
    for r in runs:
        cfg = r.get("config", {}) or {}
        th = cfg.get("two_head_config")
        div = cfg.get("diversify_config") or {}
        
        if th and isinstance(th, dict):
            alpha = th.get("alpha")
            mode = th.get("mode", "scalarize")
            if alpha is not None or mode == "pareto_balanced":
                two_head.append({
                    "run": r,
                    "lambda_type": "two_head_alpha",
                    "lambda_value": alpha if alpha is not None else "pareto_balanced",
                    "mode": mode,
                })

        pop_alpha = div.get("popularity_penalty_alpha")
        if pop_alpha is not None and pop_alpha > 0:
            debias.append({
                "run": r,
                "lambda_type": "popularity_penalty_alpha",
                "lambda_value": float(pop_alpha),
            })
    
    return two_head, debias


def _get_metric(run: Dict[str, Any], path: str) -> Optional[float]:
    parts = path.split(".")
    current = run
    for p in parts:
        if isinstance(current, dict):
            current = current.get(p)
        else:
            return None
        if current is None:
            return None
    try:
        return float(current) if current is not None else None
    except (TypeError, ValueError):
        return None


def _aggregate_by_lambda(lambda_runs: List[Dict], metric_paths: Dict[str, str]) -> Dict[Any, Dict[str, Any]]:
    by_lambda: Dict[Any, List[Dict]] = defaultdict(list)
    for lr in lambda_runs:
        lam_val = lr["lambda_value"]
        by_lambda[lam_val].append(lr["run"])
    
    agg = {}
    for lam_val, group in by_lambda.items():
        metrics_dict = {}
        for name, path in metric_paths.items():
            vals = [_get_metric(r, path) for r in group]
            vals = [v for v in vals if v is not None]
            if vals:
                metrics_dict[name] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)) if len(vals) > 1 else 0.0,
                    "n": len(vals),
                }
            else:
                metrics_dict[name] = None
        agg[lam_val] = metrics_dict
    
    return agg


def plot_pareto_trajectories(
    two_head_agg: Dict[Any, Dict[str, Any]],
    debias_agg: Dict[Any, Dict[str, Any]],
    out_dir: Path,
) -> None:

    apply_paper_style()
    
    objectives = [
        ("ndcg@10", "serendipity@10", "nDCG@10 (accuracy)", "Serendipity@10 (novelty)", "accuracy_vs_serendipity"),
        ("hr@10", "serendipity@10", "HR@10 (accuracy)", "Serendipity@10 (novelty)", "accuracy_vs_serendipity_hr"),
        ("ndcg@10", "coverage", "nDCG@10 (accuracy)", "Catalog Coverage (diversity)", "accuracy_vs_coverage"),
        ("hr@10", "coverage", "HR@10 (accuracy)", "Catalog Coverage (diversity)", "accuracy_vs_coverage_hr"),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (x_met, y_met, x_lab, y_lab, title_suf) in enumerate(objectives):
        ax = axes[idx]
        

        if two_head_agg:
            xs, ys, labels = [], [], []

            sorted_keys = sorted(
                [k for k in two_head_agg.keys() if isinstance(k, (int, float))],
                key=lambda x: float(x)
            )
            if "pareto_balanced" in two_head_agg:
                sorted_keys.append("pareto_balanced")
            
            for lam in sorted_keys:
                m = two_head_agg.get(lam)
                if not m:
                    continue
    
                xm = m.get(x_met)
                if not xm and x_met == "coverage":
                    xm = m.get("catalog_coverage_at_10")
                ym = m.get(y_met)
                if not ym and y_met == "coverage":
                    ym = m.get("catalog_coverage_at_10")
                if xm and ym and xm.get("mean") is not None and ym.get("mean") is not None:
                    xs.append(xm["mean"])
                    ys.append(ym["mean"])
                    if isinstance(lam, (int, float)):
                        labels.append(f"α={lam:.2f}")
                    else:
                        labels.append("Pareto-balanced")
            
            if xs:
                ax.plot(xs, ys, "o-", linewidth=2, markersize=8, label="Two-head (relevance+novelty)", color="#ff7f0e")
       
                for i, (x, y, lbl) in enumerate(zip(xs, ys, labels)):
                    if i == 0 or i == len(xs) - 1 or (len(xs) > 2 and i == len(xs) // 2):
                        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=AXES_FONTSIZE - 2)

        if debias_agg:
            xs_d, ys_d = [], []
            sorted_keys_d = sorted([k for k in debias_agg.keys() if isinstance(k, (int, float))])
            for lam in sorted_keys_d:
                m = debias_agg.get(lam)
                if not m:
                    continue
                xm = m.get(x_met)
                if not xm and x_met == "coverage":
                    xm = m.get("catalog_coverage_at_10")
                ym = m.get(y_met)
                if not ym and y_met == "coverage":
                    ym = m.get("catalog_coverage_at_10")
                if xm and ym and xm.get("mean") is not None and ym.get("mean") is not None:
                    xs_d.append(xm["mean"])
                    ys_d.append(ym["mean"])
            
            if xs_d:
                ax.plot(xs_d, ys_d, "s--", linewidth=2, markersize=6, label="Popularity penalty", color="#2ca02c", alpha=0.7)
        
        ax.set_xlabel(x_lab, fontsize=AXES_FONTSIZE)
        ax.set_ylabel(y_lab, fontsize=AXES_FONTSIZE)
        ax.set_title(f"Pareto: {title_suf.replace('_', ' ').title()}", fontsize=TITLE_FONTSIZE)
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=AXES_FONTSIZE - 1, loc="best")
    
    plt.tight_layout()
    save_fig_paper(out_dir / "multiobjective_pareto_trajectories")
    plt.close()


def generate_policy_guide(
    two_head_agg: Dict[Any, Dict[str, Any]],
    debias_agg: Dict[Any, Dict[str, Any]],
    out_dir: Path,
) -> None:

    policy = {
        "two_head_alpha": {},
        "popularity_penalty_alpha": {},
        "recommendations": [],
    }
    

    if two_head_agg:
        sorted_alphas = sorted([k for k in two_head_agg.keys() if isinstance(k, (int, float))], key=float)
        if sorted_alphas:

            goals = {
                "max_accuracy": ("ndcg@10", "max"),
                "max_serendipity": ("serendipity@10", "max"),
                "balanced": ("ndcg@10", "balanced"),  
            }
            
            for goal_name, (metric, strategy) in goals.items():
                best_alpha = None
                best_score = None
                if strategy == "max":
                    for alpha in sorted_alphas:
                        m = two_head_agg.get(alpha)
                        if m and m.get(metric):
                            score = m[metric]["mean"]
                            if best_score is None or score > best_score:
                                best_score = score
                                best_alpha = alpha
                elif strategy == "balanced":
                    
                    best_alpha = None
                    best_balanced = None
                    for alpha in sorted_alphas:
                        m = two_head_agg.get(alpha)
                        if m and m.get("ndcg@10") and m.get("serendipity@10"):
                            acc = m["ndcg@10"]["mean"]
                            ser = m["serendipity@10"]["mean"]
                            balanced = 2 * acc * ser / (acc + ser) if (acc + ser) > 0 else 0
                            if best_balanced is None or balanced > best_balanced:
                                best_balanced = balanced
                                best_alpha = alpha
                
                if best_alpha is not None:
                    m = two_head_agg.get(best_alpha)
                    policy["two_head_alpha"][goal_name] = {
                        "alpha": float(best_alpha),
                        "ndcg@10": m.get("ndcg@10", {}).get("mean") if m and m.get("ndcg@10") else None,
                        "hr@10": m.get("hr@10", {}).get("mean") if m and m.get("hr@10") else None,
                        "serendipity@10": m.get("serendipity@10", {}).get("mean") if m and m.get("serendipity@10") else None,
                        "coverage": m.get("coverage", {}).get("mean") if m and m.get("coverage") else None,
                        "catalog_coverage_at_10": m.get("catalog_coverage_at_10", {}).get("mean") if m and m.get("catalog_coverage_at_10") else None,
                    }
    
    if debias_agg:
        sorted_alphas = sorted([k for k in debias_agg.keys() if isinstance(k, (int, float))], key=float)
        if sorted_alphas:

            for goal_name in ["max_accuracy", "max_diversity", "balanced"]:
                best_alpha = None
                best_score = None
                if goal_name == "max_accuracy":
                    for alpha in sorted_alphas:
                        m = debias_agg.get(alpha)
                        if m and m.get("ndcg@10"):
                            score = m["ndcg@10"]["mean"]
                            if best_score is None or score > best_score:
                                best_score = score
                                best_alpha = alpha
                elif goal_name == "max_diversity":
                    for alpha in sorted_alphas:
                        m = debias_agg.get(alpha)
                        if m and m.get("coverage"):
                            score = m["coverage"]["mean"]
                            if best_score is None or score > best_score:
                                best_score = score
                                best_alpha = alpha
                elif goal_name == "balanced":
                    best_alpha = None
                    best_balanced = None
                    for alpha in sorted_alphas:
                        m = debias_agg.get(alpha)
                        if m and m.get("ndcg@10") and m.get("coverage"):
                            acc = m["ndcg@10"]["mean"]
                            div = m["coverage"]["mean"]
                            balanced = 2 * acc * div / (acc + div) if (acc + div) > 0 else 0
                            if best_balanced is None or balanced > best_balanced:
                                best_balanced = balanced
                                best_alpha = alpha
                
                if best_alpha is not None:
                    m = debias_agg.get(best_alpha)
                    policy["popularity_penalty_alpha"][goal_name] = {
                        "alpha": float(best_alpha),
                        "ndcg@10": m.get("ndcg@10", {}).get("mean") if m and m.get("ndcg@10") else None,
                        "hr@10": m.get("hr@10", {}).get("mean") if m and m.get("hr@10") else None,
                        "coverage": m.get("coverage", {}).get("mean") if m and m.get("coverage") else None,
                        "catalog_coverage_at_10": m.get("catalog_coverage_at_10", {}).get("mean") if m and m.get("catalog_coverage_at_10") else None,
                    }
    

    recommendations = []
    if policy["two_head_alpha"]:
        recommendations.append("## Two-head reranker (relevance + novelty)")
        recommendations.append("")
        recommendations.append("| Goal | α (alpha) | nDCG@10 | Serendipity@10 | Coverage |")
        recommendations.append("|------|-----------|---------|----------------|----------|")
        for goal, data in policy["two_head_alpha"].items():
            recommendations.append(
                f"| {goal.replace('_', ' ').title()} | {data['alpha']:.2f} | "
                f"{data.get('ndcg@10', 0):.4f} | {data.get('serendipity@10', 0):.4f} | "
                f"{data.get('coverage', 0):.4f} |"
            )
        recommendations.append("")
        recommendations.append("**Policy:**")
        recommendations.append("- **Max accuracy:** Use α=1.0 (pure relevance)")
        recommendations.append("- **Max serendipity:** Use α=0.0 (pure novelty)")
        recommendations.append("- **Balanced:** Use α≈0.5 (equal weight)")
        recommendations.append("")
    
    if policy["popularity_penalty_alpha"]:
        recommendations.append("## Popularity penalty (accuracy vs diversity)")
        recommendations.append("")
        recommendations.append("| Goal | α (penalty) | nDCG@10 | Coverage |")
        recommendations.append("|------|-------------|---------|----------|")
        for goal, data in policy["popularity_penalty_alpha"].items():
            recommendations.append(
                f"| {goal.replace('_', ' ').title()} | {data['alpha']:.2f} | "
                f"{data.get('ndcg@10', 0):.4f} | {data.get('coverage', 0):.4f} |"
            )
        recommendations.append("")
        recommendations.append("**Policy:**")
        recommendations.append("- **Max accuracy:** Use α=0.0 (no penalty)")
        recommendations.append("- **Max diversity:** Use α≥0.5 (strong penalty)")
        recommendations.append("- **Balanced:** Use α≈0.3 (moderate penalty)")
        recommendations.append("")
    
 
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "multiobjective_policy_guide.json", "w", encoding="utf-8") as f:
        json.dump(policy, f, indent=2, ensure_ascii=False)
    
    md_lines = [
        "# Multi-objective Policy Guide",
        "",
        "Рекомендации по выбору λ (alpha) для разных целей.",
        "",
    ] + recommendations
    
    with open(out_dir / "multiobjective_policy_guide.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))


def main():
    runs = _read_jsonl(RUNS_JSONL)
    if not runs:
        print(f"No runs found in {RUNS_JSONL}. Run experiments first.")
        return
    
    two_head_runs, debias_runs = _extract_lambda_runs(runs)
    

    metric_paths = {
        "ndcg@10": "metrics.ndcg@10.mean",
        "hr@10": "metrics.hr@10.mean",
        "serendipity@10": "diagnostics.serendipity@10",
        "coverage": "diagnostics.coverage",
        "catalog_coverage_at_10": "diagnostics.catalog_coverage_at_10",
        "mean_popularity_rank": "diagnostics.mean_popularity_rank",
    }
    
    two_head_agg = {}
    if two_head_runs:
        two_head_agg = _aggregate_by_lambda(two_head_runs, metric_paths)
    
    debias_agg = {}
    if debias_runs:
        debias_agg = _aggregate_by_lambda(debias_runs, metric_paths)
    
    if not two_head_agg and not debias_agg:
        print("No λ-controlled runs found. Run:")
        print("  - python -m tools.run_pareto_sweep --run")
        print("  - python -m src.run_debias_sweep")
        return
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    plot_pareto_trajectories(two_head_agg, debias_agg, OUT_DIR)
    generate_policy_guide(two_head_agg, debias_agg, OUT_DIR)
    
    print(f"Multi-objective analysis saved to {OUT_DIR}")
    print("Files:")
    print("  - multiobjective_pareto_trajectories.{pdf,svg,png}")
    print("  - multiobjective_policy_guide.{md,json}")


if __name__ == "__main__":
    main()
