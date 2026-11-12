import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

try:
    from .plot_style import apply_paper_style, short_name, MODEL_ORDER, save_fig_paper, AXES_FONTSIZE, TITLE_FONTSIZE
except ImportError:
    from plot_style import apply_paper_style, short_name, MODEL_ORDER, save_fig_paper, AXES_FONTSIZE, TITLE_FONTSIZE

import seaborn as sns
sns.set_style("whitegrid")

AGGREGATED_JSON = Path("experiments") / "aggregated_results.json"
RUNS_LOG_PATH = Path("experiments") / "runs.jsonl"
OUTPUT_DIR = Path("experiments") / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_aggregated() -> Dict[str, Any]:
    if not AGGREGATED_JSON.exists():
        raise FileNotFoundError(f"Aggregated results not found. Run tools/aggregate_runs.py first.")
    with open(AGGREGATED_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def _ci95(metrics_key: str, data: Dict) -> tuple:
    m = data.get("metrics", {}).get(metrics_key, {})
    mean = m.get("mean")
    if mean is None:
        return None, None
    if m.get("ci_95_lower") is not None and m.get("ci_95_upper") is not None:
        return mean - m["ci_95_lower"], m["ci_95_upper"] - mean
    std = m.get("std") or 0
    h = 1.96 * std
    return h, h


def plot_hr10_bar_chart(aggregated: Dict[str, Any]):
    apply_paper_style()
    models, means, err_lo, err_hi = [], [], [], []
    for model_name in MODEL_ORDER:
        for key, data in aggregated.items():
            if data["model"] == model_name:
                models.append(short_name(model_name))
                mean = data["metrics"]["hr@10"]["mean"]
                means.append(mean)
                lo, hi = _ci95("hr@10", data)
                err_lo.append(lo if lo is not None else 0)
                err_hi.append(hi if hi is not None else 0)
                break
    if not models:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(models))
    ax.bar(x, means, yerr=[err_lo, err_hi], capsize=3, alpha=0.7, color="steelblue", edgecolor="black")
    ax.set_xlabel("Model", fontsize=AXES_FONTSIZE, fontweight="bold")
    ax.set_ylabel("HR@10", fontsize=AXES_FONTSIZE, fontweight="bold")
    ax.set_title("Hit Rate@10 (95% CI)", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.grid(axis="y", alpha=0.3)
    for i, mean in enumerate(means):
        ax.text(i, mean + (err_hi[i] or 0) + 0.01, f"{mean:.3f}", ha="center", va="bottom", fontsize=AXES_FONTSIZE)
    plt.tight_layout()
    save_fig_paper(OUTPUT_DIR / "hr10_bar_chart")
    plt.close()


def plot_recall_curves(aggregated: Dict[str, Any]):
    apply_paper_style()
    pool_sizes = {}
    for key, data in aggregated.items():
        pool_size = data["config"].get("candidate_pool_size")
        if pool_size is None:
            continue
        if pool_size not in pool_sizes:
            pool_sizes[pool_size] = {"recall@50": [], "recall@200": [], "recall@1000": []}
        diag = data.get("diagnostics", {})
        for k in [50, 200, 1000]:
            v = diag.get(f"recall@{k}", {})
            if isinstance(v, dict) and v.get("mean") is not None:
                pool_sizes[pool_size][f"recall@{k}"].append(v["mean"])
    if not pool_sizes:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    Ks = [50, 200, 1000]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for idx, pool_size in enumerate(sorted(pool_sizes.keys())):
        recalls = []
        for k in Ks:
            vals = pool_sizes[pool_size].get(f"recall@{k}", [])
            recalls.append(np.mean(vals) if vals else 0.0)
        ax.plot(Ks, recalls, marker="o", linewidth=2, markersize=6, label=f"Pool={pool_size}", color=colors[idx % len(colors)])
    ax.set_xlabel("K (top-K candidates)", fontsize=AXES_FONTSIZE, fontweight="bold")
    ax.set_ylabel("Recall@K", fontsize=AXES_FONTSIZE, fontweight="bold")
    ax.set_title("Recall@K vs pool size", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_fig_paper(OUTPUT_DIR / "recall_curves")
    plt.close()


def plot_unique_top1_histogram(aggregated: Dict[str, Any]):
    apply_paper_style()
    models, means, err_lo, err_hi = [], [], [], []
    n_users = None
    for model_name in MODEL_ORDER:
        for key, data in aggregated.items():
            if data["model"] == model_name:
                models.append(short_name(model_name))
                diag = data.get("diagnostics", {})
                u = diag.get("unique_top1", {})
                mean = u.get("mean") if u.get("mean") is not None else 0
                means.append(mean)
                std = u.get("std") or 0
                h = 1.96 * std
                err_lo.append(h)
                err_hi.append(h)
                if n_users is None and "n_users" in data.get("config", {}):
                    n_users = data["config"]["n_users"]
                break
    if not models:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    y_pos = np.arange(len(models))
    ax.barh(y_pos, means, xerr=[err_lo, err_hi], capsize=3, alpha=0.7, color="coral", edgecolor="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlabel("Unique top-1 items", fontsize=AXES_FONTSIZE, fontweight="bold")
    ax.set_ylabel("Model", fontsize=AXES_FONTSIZE, fontweight="bold")
    title = "Unique top-1 items (lower = more bias)"
    if n_users is not None:
        title += f" (N = {n_users} users)"
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    for i, val in enumerate(means):
        ax.text(val + (err_hi[i] or 0) + 0.5, i, f"{int(val)}", va="center", fontsize=AXES_FONTSIZE)
    plt.tight_layout()
    save_fig_paper(OUTPUT_DIR / "unique_top1_histogram")
    plt.close()


def plot_ndcg10_comparison(aggregated: Dict[str, Any]):
    apply_paper_style()
    models, means, err_lo, err_hi = [], [], [], []
    for model_name in MODEL_ORDER:
        for key, data in aggregated.items():
            if data["model"] == model_name:
                models.append(short_name(model_name))
                mean = data["metrics"]["ndcg@10"]["mean"]
                means.append(mean)
                lo, hi = _ci95("ndcg@10", data)
                err_lo.append(lo if lo is not None else 0)
                err_hi.append(hi if hi is not None else 0)
                break
    if not models:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(models))
    ax.bar(x, means, yerr=[err_lo, err_hi], capsize=3, alpha=0.7, color="mediumseagreen", edgecolor="black")
    ax.set_xlabel("Model", fontsize=AXES_FONTSIZE, fontweight="bold")
    ax.set_ylabel("nDCG@10", fontsize=AXES_FONTSIZE, fontweight="bold")
    ax.set_title("nDCG@10 (95% CI)", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.grid(axis="y", alpha=0.3)
    for i, mean in enumerate(means):
        ax.text(i, mean + (err_hi[i] or 0) + 0.005, f"{mean:.3f}", ha="center", va="bottom", fontsize=AXES_FONTSIZE)
    plt.tight_layout()
    save_fig_paper(OUTPUT_DIR / "ndcg10_bar_chart")
    plt.close()


def main():
    print("Loading aggregated results...")
    try:
        aggregated = load_aggregated()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run: python -m tools.aggregate_runs")
        return
    
    print(f"Generating plots from {len(aggregated)} configurations...")
    
    plot_hr10_bar_chart(aggregated)
    plot_ndcg10_comparison(aggregated)
    plot_recall_curves(aggregated)
    plot_unique_top1_histogram(aggregated)
    
    print(f"\nAll plots saved to {OUTPUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
