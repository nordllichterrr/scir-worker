import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

import matplotlib.pyplot as plt

try:
    from .plot_style import apply_paper_style, save_fig_paper, AXES_FONTSIZE, TITLE_FONTSIZE, MODEL_DISPLAY_NAMES, MODEL_ORDER
except ImportError:
    from plot_style import apply_paper_style, save_fig_paper, AXES_FONTSIZE, TITLE_FONTSIZE
    MODEL_DISPLAY_NAMES = {}
    MODEL_ORDER = []

AGGREGATED_JSON = Path("experiments") / "aggregated_results.json"
OUTPUT_DIR = Path("experiments") / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_aggregated() -> Dict[str, Any]:
    if not AGGREGATED_JSON.exists():
        raise FileNotFoundError(f"Aggregated results not found. Run tools/aggregate_runs.py first.")
    with open(AGGREGATED_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def get_model_display_name(key: str, config: Dict) -> str:
    baseline = config.get("baseline")
    use_reranker = config.get("use_reranker", False)
    pool_size = config.get("candidate_pool_size")
    
    if baseline:
        name = MODEL_DISPLAY_NAMES.get(baseline, baseline)
    elif use_reranker:
        name = "Ours+Rerank"
    else:
        name = "Candidates"
    
    if pool_size and pool_size != 1000:
        name += f"_p{pool_size}"
    return name


def plot_pareto_front(
    aggregated: Dict[str, Any],
    x_metric: str,
    y_metric: str,
    x_label: str,
    y_label: str,
    title: str,
    filename: str,
    x_higher_better: bool = True,
    y_higher_better: bool = True,
) -> None:
    """
    Plot Pareto-front: each point is a model configuration.
    x_metric/y_metric: paths like "diagnostics.serendipity@10.mean" or "metrics.ndcg@10.mean"
    """
    apply_paper_style()
    
    points: List[Tuple[float, float, str, str]] = []
    
    for key, data in aggregated.items():
        x_val = _get_nested(data, x_metric)
        y_val = _get_nested(data, y_metric)
        
        if x_val is None or y_val is None:
            continue
        
        config = data.get("config", {})
        display_name = get_model_display_name(key, config)
        points.append((x_val, y_val, display_name, key))
    
    if not points:
        print(f"No data for {title}.")
        return
    
    xs, ys, labels, keys = zip(*points)
    
    fig, ax = plt.subplots(figsize=(7, 5))

    colors = []
    for key in keys:
        config = aggregated[key].get("config", {})
        if config.get("baseline"):
            colors.append("#1f77b4")  
        elif config.get("use_reranker"):
            colors.append("#ff7f0e") 
        else:
            colors.append("#2ca02c") 
    
    scatter = ax.scatter(xs, ys, c=colors, alpha=0.7, s=100, edgecolors="black", linewidths=0.5)
    
    for x, y, lab, _ in zip(xs, ys, labels, keys):
        ax.annotate(
            lab,
            (x, y),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=AXES_FONTSIZE - 1,
            alpha=0.8,
        )
    
    if x_higher_better and y_higher_better:
        pareto_points = _compute_pareto_front(list(zip(xs, ys)))
        if pareto_points:
            pareto_xs, pareto_ys = zip(*pareto_points)
            sorted_pareto = sorted(zip(pareto_xs, pareto_ys), key=lambda p: p[0])
            if len(sorted_pareto) > 1:
                px, py = zip(*sorted_pareto)
                ax.plot(px, py, "r--", alpha=0.5, linewidth=1.5, label="Pareto front")
                ax.legend(fontsize=AXES_FONTSIZE - 1)
    
    ax.set_xlabel(x_label, fontsize=AXES_FONTSIZE, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=AXES_FONTSIZE, fontweight="bold")
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    save_fig_paper(OUTPUT_DIR / filename)
    plt.close()


def _get_nested(data: Dict, path: str) -> Any:
    parts = path.split(".")
    current = data
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
        if current is None:
            return None
    return current


def _compute_pareto_front(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:

    if not points:
        return []
    
    pareto = []
    for px, py in points:
        is_dominated = False
        for ox, oy in points:
            if (ox, oy) == (px, py):
                continue
            if ox >= px and oy >= py and (ox > px or oy > py):
                is_dominated = True
                break
        if not is_dominated:
            pareto.append((px, py))
    
    return pareto


def main():
    print("Loading aggregated results for serendipity/novelty trade-off plots...")
    try:
        aggregated = load_aggregated()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run: python -m tools.aggregate_runs")
        return
    
    print("Plotting Pareto-fronts for serendipity/novelty trade-offs...")
    
    plot_pareto_front(
        aggregated,
        x_metric="diagnostics.serendipity@10.mean",
        y_metric="metrics.ndcg@10.mean",
        x_label="Serendipity@10 (higher = more serendipitous)",
        y_label="nDCG@10 (higher = better accuracy)",
        title="nDCG@10 vs Serendipity@10 (Pareto-front)",
        filename="pareto_ndcg10_vs_serendipity10",
        x_higher_better=True,
        y_higher_better=True,
    )
    
    plot_pareto_front(
        aggregated,
        x_metric="diagnostics.serendipity@10.mean",
        y_metric="metrics.hr@10.mean",
        x_label="Serendipity@10 (higher = more serendipitous)",
        y_label="HR@10 (higher = better accuracy)",
        title="HR@10 vs Serendipity@10 (Pareto-front)",
        filename="pareto_hr10_vs_serendipity10",
        x_higher_better=True,
        y_higher_better=True,
    )
    
    plot_pareto_front(
        aggregated,
        x_metric="diagnostics.catalog_coverage_at_10.mean",
        y_metric="metrics.ndcg@10.mean",
        x_label="Catalog Coverage@10 (higher = more diverse)",
        y_label="nDCG@10 (higher = better accuracy)",
        title="nDCG@10 vs Catalog Coverage@10 (Pareto-front)",
        filename="pareto_ndcg10_vs_coverage10",
        x_higher_better=True,
        y_higher_better=True,
    )
    
    plot_pareto_front(
        aggregated,
        x_metric="diagnostics.catalog_coverage_at_10.mean",
        y_metric="metrics.hr@10.mean",
        x_label="Catalog Coverage@10 (higher = more diverse)",
        y_label="HR@10 (higher = better accuracy)",
        title="HR@10 vs Catalog Coverage@10 (Pareto-front)",
        filename="pareto_hr10_vs_coverage10",
        x_higher_better=True,
        y_higher_better=True,
    )
    
    plot_pareto_front(
        aggregated,
        x_metric="diagnostics.mean_self_information_novelty.mean",
        y_metric="metrics.ndcg@10.mean",
        x_label="Novelty (mean self-information, higher = more novel)",
        y_label="nDCG@10 (higher = better accuracy)",
        title="nDCG@10 vs Novelty (Pareto-front)",
        filename="pareto_ndcg10_vs_novelty",
        x_higher_better=True,
        y_higher_better=True,
    )
    
    plot_pareto_front(
        aggregated,
        x_metric="diagnostics.mean_self_information_novelty.mean",
        y_metric="metrics.hr@10.mean",
        x_label="Novelty (mean self-information, higher = more novel)",
        y_label="HR@10 (higher = better accuracy)",
        title="HR@10 vs Novelty (Pareto-front)",
        filename="pareto_hr10_vs_novelty",
        x_higher_better=True,
        y_higher_better=True,
    )
    
    print(f"\nSerendipity/novelty trade-off plots saved to {OUTPUT_DIR}")
    print("Files:")
    print("  - pareto_ndcg10_vs_serendipity10.{pdf,svg,png}")
    print("  - pareto_hr10_vs_serendipity10.{pdf,svg,png}")
    print("  - pareto_ndcg10_vs_coverage10.{pdf,svg,png}")
    print("  - pareto_hr10_vs_coverage10.{pdf,svg,png}")
    print("  - pareto_ndcg10_vs_novelty.{pdf,svg,png}")
    print("  - pareto_hr10_vs_novelty.{pdf,svg,png}")


if __name__ == "__main__":
    main()
