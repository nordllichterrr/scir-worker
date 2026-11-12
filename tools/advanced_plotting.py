import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import seaborn as sns

try:
    from .plot_style import (
        apply_paper_style,
        short_name,
        MODEL_ORDER,
        bootstrap_ci,
        save_fig_paper,
        AXES_FONTSIZE,
        TITLE_FONTSIZE,
    )
except ImportError:
    from plot_style import (
        apply_paper_style,
        short_name,
        MODEL_ORDER,
        bootstrap_ci,
        save_fig_paper,
        AXES_FONTSIZE,
        TITLE_FONTSIZE,
    )

sns.set_style("whitegrid")
MASTER_JSON = Path("experiments") / "master_results.json"
AGGREGATED_JSON = Path("experiments") / "aggregated_results.json"
OUTPUT_DIR = Path("experiments") / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_master():
    if not MASTER_JSON.exists():
        raise FileNotFoundError(f"Master results not found. Run tools/build_master_results.py first.")
    with open(MASTER_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def load_aggregated():
    if not AGGREGATED_JSON.exists():
        raise FileNotFoundError(f"Aggregated results not found. Run tools/aggregate_runs.py first.")
    with open(AGGREGATED_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_recall_curves_detailed(master: Dict[str, Any]):
    apply_paper_style()
    runs_by_model = defaultdict(list)
    for run in master["runs"]:
        config = run.get("config", {})
        baseline = config.get("baseline")
        use_reranker = config.get("use_reranker", False)
        if baseline:
            model_key = baseline
        elif use_reranker:
            model_key = "ours_with_reranker"
        else:
            model_key = "candidates_only"
        runs_by_model[model_key].append(run)

    fig, ax = plt.subplots(figsize=(6, 4))
    Ks = [10, 20, 50, 100, 200, 500, 1000]
    plot_idx = 0
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(runs_by_model), 1)))

    for model_key in MODEL_ORDER:
        if model_key not in runs_by_model:
            continue
        model_runs = runs_by_model[model_key]
        all_recalls = {k: [] for k in Ks}
        for run in model_runs:
            diag = run.get("diagnostics", {})
            for k in [50, 200, 1000]:
                v = diag.get(f"recall@{k}")
                if v is not None:
                    all_recalls[k].append(float(v) if isinstance(v, (int, float)) else v.get("mean"))
        means, err_lo, err_hi, valid_ks = [], [], [], []
        for k in [50, 200, 1000]:
            if all_recalls[k]:
                m = np.mean(all_recalls[k])
                lo, hi = bootstrap_ci(all_recalls[k])
                means.append(m)
                err_lo.append(m - lo if lo is not None else 0)
                err_hi.append(hi - m if hi is not None else 0)
                valid_ks.append(k)
        if valid_ks:
            err = np.array([err_lo, err_hi])
            ax.errorbar(valid_ks, means, yerr=err, marker="o", linewidth=2, markersize=6,
                        label=short_name(model_key), color=colors[plot_idx % len(colors)], capsize=3)
            plot_idx += 1
    ax.set_xlabel("K (top-K candidates)", fontsize=AXES_FONTSIZE, fontweight="bold")
    ax.set_ylabel("Recall@K", fontsize=AXES_FONTSIZE, fontweight="bold")
    ax.set_title("Recall@K Curves by Model", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_fig_paper(OUTPUT_DIR / "recall_curves_detailed")
    plt.close()


def plot_hr_ndcg_distributions(master: Dict[str, Any]):
    apply_paper_style()
    runs_by_model = defaultdict(list)
    for run in master["runs"]:
        config = run.get("config", {})
        baseline = config.get("baseline")
        use_reranker = config.get("use_reranker", False)
        if baseline:
            model_key = baseline
        elif use_reranker:
            model_key = "ours_with_reranker"
        else:
            model_key = "candidates_only"
        runs_by_model[model_key].append(run)

    hr_data, hr_labels, hr_ci_lo, hr_ci_hi = [], [], [], []
    ndcg_data, ndcg_labels, ndcg_ci_lo, ndcg_ci_hi = [], [], [], []

    for model_key in MODEL_ORDER:
        if model_key not in runs_by_model:
            continue
        all_hr = []
        all_ndcg = []
        for run in runs_by_model[model_key]:
            hr_per = run.get("metrics", {}).get("hr@10", {}).get("per_user", [])
            ndcg_per = run.get("metrics", {}).get("ndcg@10", {}).get("per_user", [])
            if isinstance(hr_per, list):
                all_hr.extend([float(x) for x in hr_per])
            if isinstance(ndcg_per, list):
                all_ndcg.extend([float(x) for x in ndcg_per])
        if all_hr:
            hr_data.append(all_hr)
            hr_labels.append(short_name(model_key))
            lo, hi = bootstrap_ci(all_hr)
            hr_ci_lo.append(lo)
            hr_ci_hi.append(hi)
        if all_ndcg:
            ndcg_data.append(all_ndcg)
            ndcg_labels.append(short_name(model_key))
            lo, hi = bootstrap_ci(all_ndcg)
            ndcg_ci_lo.append(lo)
            ndcg_ci_hi.append(hi)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, len(hr_labels)))

    if hr_data:
        parts = ax1.violinplot(hr_data, positions=range(len(hr_data)), showmeans=False, showmedians=False)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
        ax1.set_xticks(range(len(hr_labels)))
        ax1.set_xticklabels(hr_labels)
        means = [np.mean(d) for d in hr_data]
        medians = [np.median(d) for d in hr_data]
        ax1.scatter(range(len(hr_data)), means, color="black", s=40, zorder=3, marker="o", label="Mean")
        ax1.scatter(range(len(hr_data)), medians, color="red", s=30, zorder=3, marker="^", label="Median")
        err_lo = [means[i] - (hr_ci_lo[i] if hr_ci_lo[i] is not None else means[i]) for i in range(len(means))]
        err_hi = [(hr_ci_hi[i] if hr_ci_hi[i] is not None else means[i]) - means[i] for i in range(len(means))]
        ax1.errorbar(range(len(means)), means, yerr=[err_lo, err_hi], fmt="none", color="black", capsize=2)
        ax1.set_ylabel("HR@10", fontsize=AXES_FONTSIZE, fontweight="bold")
        ax1.set_title("HR@10 (per user)", fontsize=TITLE_FONTSIZE, fontweight="bold")
        ax1.grid(axis="y", alpha=0.3)

    if ndcg_data:
        parts = ax2.violinplot(ndcg_data, positions=range(len(ndcg_data)), showmeans=False, showmedians=False)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
        ax2.set_xticks(range(len(ndcg_labels)))
        ax2.set_xticklabels(ndcg_labels)
        means = [np.mean(d) for d in ndcg_data]
        medians = [np.median(d) for d in ndcg_data]
        ax2.scatter(range(len(ndcg_data)), means, color="black", s=40, zorder=3, marker="o", label="Mean")
        ax2.scatter(range(len(ndcg_data)), medians, color="red", s=30, zorder=3, marker="^", label="Median")
        err_lo = [means[i] - (ndcg_ci_lo[i] if ndcg_ci_lo[i] is not None else means[i]) for i in range(len(means))]
        err_hi = [(ndcg_ci_hi[i] if ndcg_ci_hi[i] is not None else means[i]) - means[i] for i in range(len(means))]
        ax2.errorbar(range(len(means)), means, yerr=[err_lo, err_hi], fmt="none", color="black", capsize=2)
        ax2.set_ylabel("nDCG@10", fontsize=AXES_FONTSIZE, fontweight="bold")
        ax2.set_title("nDCG@10 (per user)", fontsize=TITLE_FONTSIZE, fontweight="bold")
        ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_fig_paper(OUTPUT_DIR / "hr_ndcg_distributions")
    plt.close()


def plot_top1_concentration(master: Dict[str, Any]):
    apply_paper_style()
    runs_by_model = defaultdict(list)
    n_users_total = None
    for run in master["runs"]:
        config = run.get("config", {})
        baseline = config.get("baseline")
        use_reranker = config.get("use_reranker", False)
        if baseline:
            model_key = baseline
        elif use_reranker:
            model_key = "ours_with_reranker"
        else:
            model_key = "candidates_only"
        runs_by_model[model_key].append(run)
        if n_users_total is None:
            n_users_total = len(run.get("per_user_detail", {}))

    models = []
    top1_means = []
    top1_ci_lo = []
    top1_ci_hi = []

    for model_key in MODEL_ORDER:
        if model_key not in runs_by_model:
            continue
        per_run_fractions = []
        for run in runs_by_model[model_key]:
            diag = run.get("diagnostics", {})
            top1_list = diag.get("top1_counts", [])
            all_top1_items = []
            for item_id, count in top1_list:
                all_top1_items.extend([item_id] * count)
            if all_top1_items:
                from collections import Counter
                counter = Counter(all_top1_items)
                total = sum(counter.values())
                top1_fraction = counter.most_common(1)[0][1] / total if total > 0 else 0
                per_run_fractions.append(top1_fraction)
        if per_run_fractions:
            models.append(short_name(model_key))
            top1_means.append(np.mean(per_run_fractions))
            lo, hi = bootstrap_ci(per_run_fractions)
            top1_ci_lo.append(lo if lo is not None else top1_means[-1])
            top1_ci_hi.append(hi if hi is not None else top1_means[-1])

    if not models:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    y_pos = np.arange(len(models))
    bars = ax.barh(y_pos, top1_means, xerr=[np.array(top1_means) - np.array(top1_ci_lo), np.array(top1_ci_hi) - np.array(top1_means)],
                   alpha=0.7, color="coral", edgecolor="black", capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlabel("Fraction of users with most common top-1 item", fontsize=AXES_FONTSIZE, fontweight="bold")
    ax.set_ylabel("Model", fontsize=AXES_FONTSIZE, fontweight="bold")
    title = "Top-1 concentration (higher = more bias)"
    if n_users_total is not None:
        title += f" (N = {n_users_total} users)"
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    for i, val in enumerate(top1_means):
        ax.text(val + 0.01, i, f"{val:.3f}", va="center", fontsize=AXES_FONTSIZE)
    plt.tight_layout()
    save_fig_paper(OUTPUT_DIR / "top1_concentration")
    plt.close()


def plot_item_exposure_distribution(master: Dict[str, Any]):
    apply_paper_style()
    exposure = master.get("per_item_exposure", {})
    if not exposure:
        print("No exposure data available")
        return

    def _count(data):
        return data.get("exposure_count", data.get("count", 0))

    exposure_counts = [_count(data) for data in exposure.values()]
    exposure_counts = [c for c in exposure_counts if c > 0]
    if not exposure_counts:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))


    ax1.set_xscale("log")
    bins_log = np.logspace(np.log10(max(1, min(exposure_counts))), np.log10(max(exposure_counts) + 1), 50)
    ax1.hist(exposure_counts, bins=bins_log, alpha=0.7, color="steelblue", edgecolor="black")
    ax1.set_xlabel("Exposure count (times in top-10)", fontsize=AXES_FONTSIZE, fontweight="bold")
    ax1.set_ylabel("Number of items", fontsize=AXES_FONTSIZE, fontweight="bold")
    ax1.set_title("Item exposure distribution", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax1.grid(alpha=0.3)


    sorted_exposure = sorted(exposure_counts, reverse=True)
    cumulative = np.cumsum(sorted_exposure)
    total = cumulative[-1]
    cumulative_pct = cumulative / total if total > 0 else cumulative
    ax2.plot(range(1, len(cumulative_pct) + 1), cumulative_pct, linewidth=2, color="darkred")
    ax2.axhline(y=0.8, color="gray", linestyle="--", alpha=0.6, label="80%")

    idx80 = np.searchsorted(cumulative_pct, 0.8, side="left")
    N80 = idx80 + 1 if idx80 < len(cumulative_pct) else len(cumulative_pct)
    ax2.axvline(x=N80, color="green", linestyle=":", linewidth=1.5, alpha=0.8)
    ax2.annotate(f"N = {N80}\n(80% exposure)", xy=(N80, 0.8), xytext=(N80 * 1.05, 0.75),
                 fontsize=AXES_FONTSIZE, ha="left")
    ax2.set_xlabel("Items (sorted by exposure)", fontsize=AXES_FONTSIZE, fontweight="bold")
    ax2.set_ylabel("Cumulative exposure fraction", fontsize=AXES_FONTSIZE, fontweight="bold")
    ax2.set_title("Exposure concentration", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    save_fig_paper(OUTPUT_DIR / "item_exposure_distribution")
    plt.close()


def main():
    print("Loading data...")
    try:
        master = load_master()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run: python -m tools.build_master_results")
        return
    
    print(f"Generating advanced visualizations from {len(master['runs'])} runs...")

    plot_recall_curves_detailed(master)
    plot_hr_ndcg_distributions(master)
    plot_top1_concentration(master)
    plot_item_exposure_distribution(master)

    print(f"\nAll plots saved to {OUTPUT_DIR} (PDF, SVG, PNG)")
    print("Done!")


if __name__ == "__main__":
    main()
