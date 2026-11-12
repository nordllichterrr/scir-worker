
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt

try:
    from .plot_style import apply_paper_style, save_fig_paper, AXES_FONTSIZE, TITLE_FONTSIZE
except ImportError:
    from plot_style import apply_paper_style, save_fig_paper, AXES_FONTSIZE, TITLE_FONTSIZE

def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


SEGMENTATION_JSON = _project_root() / "experiments" / "segmentation" / "segmentation_results.json"
OUT_DIR = _project_root() / "experiments" / "segmentation"
BUCKET_ORDER = ("0", "1-2", "3-5", "6-10", "11-20", "21+")
SCENARIO_ORDER = ("new_users", "new_items", "both")


def get_model_display_name(config_key: str) -> str:
    if "ours_with_reranker" in config_key:
        return "Ours+Rerank"
    if "candidates_only" in config_key:
        return "Candidates"
    base = config_key.replace("_no_reranker", "").replace("_", " ").title()
    return base[:12]


def load_segmentation() -> Dict[str, Any]:
    if not SEGMENTATION_JSON.exists():
        raise FileNotFoundError(f"Run tools.segmentation_analysis first: {SEGMENTATION_JSON}")
    with open(SEGMENTATION_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_grouped_bar(
    segment_order: List[str],
    by_config: Dict[str, Dict[str, Dict]],
    metric: str,
    title: str,
    filename: str,
    xlabel: str = "Segment",
):
    apply_paper_style()
    configs = list(by_config.keys())
    configs_sorted = sorted(configs, key=lambda c: (0 if "ours" in c or "candidates_only" in c else 1, c))
    x = np.arange(len(segment_order))
    width = 0.8 / max(len(configs_sorted), 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, config_key in enumerate(configs_sorted):
        means = []
        stds = []
        for seg in segment_order:
            seg_data = by_config[config_key].get(seg, {})
            m = seg_data.get(metric)
            s = seg_data.get(metric.replace("mean", "std"))
            means.append(m if m is not None else 0.0)
            stds.append(s if s is not None else 0.0)
        offset = (i - len(configs_sorted) / 2) * width + width / 2
        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=2, label=get_model_display_name(config_key))
    ax.set_xticks(x)
    ax.set_xticklabels(segment_order, fontsize=AXES_FONTSIZE)
    ax.set_ylabel(metric.replace("_mean", "").upper() + " (mean ± std)", fontsize=AXES_FONTSIZE)
    ax.set_xlabel(xlabel, fontsize=AXES_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.legend(loc="upper right", fontsize=AXES_FONTSIZE - 1, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_fig_paper(OUT_DIR / filename)
    plt.close()


def main():
    data = load_segmentation()
    by_bucket = data.get("by_bucket", {})
    by_scenario = data.get("by_scenario", {})
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    buckets_used = [b for b in BUCKET_ORDER if any(by_bucket.get(c, {}).get(b) for c in by_bucket)]
    scenarios_used = [s for s in SCENARIO_ORDER if any(by_scenario.get(c, {}).get(s) for c in by_scenario)]

    if buckets_used:
        plot_grouped_bar(
            buckets_used,
            by_bucket,
            "hr_mean",
            "HR@10 by history bucket (train interactions)",
            "segmentation_hr_by_bucket",
            xlabel="Train interactions (bucket)",
        )
        plot_grouped_bar(
            buckets_used,
            by_bucket,
            "ndcg_mean",
            "nDCG@10 by history bucket (train interactions)",
            "segmentation_ndcg_by_bucket",
            xlabel="Train interactions (bucket)",
        )
    if scenarios_used:
        plot_grouped_bar(
            scenarios_used,
            by_scenario,
            "hr_mean",
            "HR@10 by cold-start scenario",
            "segmentation_hr_by_scenario",
            xlabel="Scenario",
        )
        plot_grouped_bar(
            scenarios_used,
            by_scenario,
            "ndcg_mean",
            "nDCG@10 by cold-start scenario",
            "segmentation_ndcg_by_scenario",
            xlabel="Scenario",
        )
    print(f"Segmentation plots saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
