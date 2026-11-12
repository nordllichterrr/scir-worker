import json
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt

try:
    from .plot_style import apply_paper_style, save_fig_paper, AXES_FONTSIZE, TITLE_FONTSIZE
except ImportError:
    from plot_style import apply_paper_style, save_fig_paper, AXES_FONTSIZE, TITLE_FONTSIZE

AGGREGATED_JSON = Path("experiments") / "aggregated_results.json"
OUTPUT_DIR = Path("experiments") / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_aggregated() -> Dict[str, Any]:
    if not AGGREGATED_JSON.exists():
        raise FileNotFoundError(f"Aggregated results not found. Run tools/aggregate_runs.py first.")
    with open(AGGREGATED_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_ndcg_vs_exposure_gini(aggregated: Dict[str, Any]) -> None:
    apply_paper_style()
    xs, ys, labels = [], [], []
    for key, data in aggregated.items():
        m = data.get("metrics", {}).get("ndcg@10", {})
        d = data.get("diagnostics", {}).get("exposure_gini", {})
        x = m.get("mean")
        y = d.get("mean")
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
        model = data.get("model", key)
        pool = data.get("config", {}).get("candidate_pool_size")
        labels.append(f"{model}_pool{pool}" if pool is not None else model)
    if not xs:
        print("No data for nDCG@10 vs Exposure Gini.")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(ys, xs, alpha=0.8)
    for x, y, lab in zip(ys, xs, labels):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(3, 3), fontsize=AXES_FONTSIZE)
    ax.set_xlabel("Exposure Gini (lower = less bias)", fontsize=AXES_FONTSIZE, fontweight="bold")
    ax.set_ylabel("nDCG@10 (higher = better)", fontsize=AXES_FONTSIZE, fontweight="bold")
    ax.set_title("nDCG@10 vs Exposure Gini", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_fig_paper(OUTPUT_DIR / "tradeoff_ndcg10_vs_exposure_gini")
    plt.close()


def plot_hr_vs_unique_top1(aggregated: Dict[str, Any]) -> None:
    apply_paper_style()
    xs, ys, labels = [], [], []
    for key, data in aggregated.items():
        m = data.get("metrics", {}).get("hr@10", {})
        d = data.get("diagnostics", {}).get("unique_top1", {})
        x = m.get("mean")
        y = d.get("mean")
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
        model = data.get("model", key)
        pool = data.get("config", {}).get("candidate_pool_size")
        labels.append(f"{model}_pool{pool}" if pool is not None else model)
    if not xs:
        print("No data for HR@10 vs Unique Top-1.")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(xs, ys, alpha=0.8)
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(3, 3), fontsize=AXES_FONTSIZE)
    ax.set_xlabel("HR@10 (higher = better)", fontsize=AXES_FONTSIZE, fontweight="bold")
    ax.set_ylabel("Unique top-1 items (higher = less collapse)", fontsize=AXES_FONTSIZE, fontweight="bold")
    ax.set_title("HR@10 vs Unique Top-1", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_fig_paper(OUTPUT_DIR / "tradeoff_hr10_vs_unique_top1")
    plt.close()


def main():
    print("Loading aggregated results for trade-off plots...")
    try:
        aggregated = load_aggregated()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run: python -m tools.aggregate_runs")
        return

    print("Plotting nDCG@10 vs Exposure Gini...")
    plot_ndcg_vs_exposure_gini(aggregated)

    print("Plotting HR@10 vs Unique Top-1...")
    plot_hr_vs_unique_top1(aggregated)

    print(f"\nTrade-off plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

