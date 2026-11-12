
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    from .plot_style import apply_paper_style, save_fig_paper, AXES_FONTSIZE, TITLE_FONTSIZE
except ImportError:
    from plot_style import apply_paper_style, save_fig_paper, AXES_FONTSIZE, TITLE_FONTSIZE

RUNS_LOG_PATH = Path("experiments") / "runs.jsonl"
OUTPUT_DIR = Path("experiments") / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_runs(path: Path) -> List[Dict[str, Any]]:
    runs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    runs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return runs


def extract_pool_size_runs(runs: List[Dict], use_reranker: bool = True) -> List[Dict[str, Any]]:
    out = []
    for r in runs:
        cfg = r.get("config", {})
        if cfg.get("baseline"):
            continue
        if bool(cfg.get("use_reranker", True)) != use_reranker:
            continue
        pool = cfg.get("candidate_pool_size")
        if pool is None:
            continue
        diag = r.get("diagnostics", {})
        rerank_mean = diag.get("rerank_time_mean")
        retrieval_mean = diag.get("retrieval_time_mean")
        if use_reranker and rerank_mean is None:
            continue
        out.append({
            "pool_size": pool,
            "run_id": r.get("run_id", ""),
            "hr10": (r.get("metrics") or {}).get("hr@10", {}) if isinstance(r.get("metrics", {}).get("hr@10"), dict) else {},
            "recall200": (diag.get("recall@200") or {}) if isinstance(diag.get("recall@200"), dict) else diag.get("recall@200"),
            "rerank_time_mean": rerank_mean,
            "retrieval_time_mean": retrieval_mean,
        })
    return out


def aggregate_by_pool_size(rows: List[Dict]) -> Dict[int, Dict[str, float]]:
    from collections import defaultdict
    by_pool = defaultdict(lambda: {"hr10": [], "recall200": [], "rerank_time": [], "retrieval_time": []})
    for r in rows:
        p = r["pool_size"]
        hr = r["hr10"]
        if isinstance(hr, dict):
            hr = hr.get("mean")
        if hr is not None:
            by_pool[p]["hr10"].append(float(hr))
        rec = r["recall200"]
        if isinstance(rec, dict):
            rec = rec.get("mean")
        if rec is not None:
            by_pool[p]["recall200"].append(float(rec))
        if r.get("rerank_time_mean") is not None:
            by_pool[p]["rerank_time"].append(float(r["rerank_time_mean"]))
        if r.get("retrieval_time_mean") is not None:
            by_pool[p]["retrieval_time"].append(float(r["retrieval_time_mean"]))
    result = {}
    for p, v in by_pool.items():
        result[p] = {
            "pool_size": p,
            "hr10_mean": np.mean(v["hr10"]) if v["hr10"] else None,
            "hr10_std": np.std(v["hr10"]) if v["hr10"] else None,
            "recall200_mean": np.mean(v["recall200"]) if v["recall200"] else None,
            "recall200_std": np.std(v["recall200"]) if v["recall200"] else None,
            "rerank_time_mean": np.mean(v["rerank_time"]) if v["rerank_time"] else None,
            "retrieval_time_mean": np.mean(v["retrieval_time"]) if v["retrieval_time"] else 0.0,
        }
        rt = result[p]["retrieval_time_mean"] or 0.0
        rr = result[p]["rerank_time_mean"] or 0.0
        result[p]["total_time_per_user"] = rt + rr
    return result


def sweet_spot_quality_per_time(data: Dict[int, Dict[str, Any]]) -> Tuple[int, str]:
    best_ratio = -1.0
    best_pool = None
    for p, v in data.items():
        t = v.get("total_time_per_user") or 0.0
        hr = v.get("hr10_mean") or 0.0
        if t > 0 and hr >= 0:
            ratio = hr / t
            if ratio > best_ratio:
                best_ratio = ratio
                best_pool = p
    return best_pool, f"HR@10/s (efficiency)" if best_pool is not None else "N/A"


def sweet_spot_knee(data: Dict[int, Dict[str, Any]], quality_key: str = "hr10_mean") -> Tuple[int, str]:
    pools = sorted(data.keys())
    if len(pools) < 2:
        return (pools[0], "only one") if pools else (None, "N/A")
    qual = [data[p].get(quality_key) or 0.0 for p in pools]
    times = [data[p].get("total_time_per_user") or 0.0 for p in pools]
    best_i = 0
    best_margin = -1e9
    for i in range(len(pools) - 1):
        dq = qual[i + 1] - qual[i]
        dt = times[i + 1] - times[i]
        if dt > 0 and dq >= 0:
            margin = dq / dt
            if margin > best_margin:
                best_margin = margin
                best_i = i + 1
    return pools[best_i], "knee (best marginal gain)"


def plot_quality_vs_time(
    data: Dict[int, Dict[str, Any]],
    quality_metric: str = "hr10_mean",
    out_path: Path = None,
    title_suffix: str = "",
) -> None:
    """Quality vs time: 95% CI (1.96*std), paper fonts, export PDF/SVG."""
    apply_paper_style()
    pools = sorted(data.keys())
    times = [data[p].get("total_time_per_user") or 0.0 for p in pools]
    quals = [data[p].get(quality_metric) or 0.0 for p in pools]
    std_key = quality_metric.replace("_mean", "_std")
    stds = [data[p].get(std_key) or 0.0 for p in pools]
    ci_half = [1.96 * s for s in stds]
    if not pools:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(times, quals, yerr=ci_half, marker="o", capsize=3, linewidth=2, markersize=6)
    for i, p in enumerate(pools):
        ax.annotate(f"pool={p}", (times[i], quals[i]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=AXES_FONTSIZE)
    ax.set_xlabel("Time per user (s) — retrieval + rerank", fontsize=AXES_FONTSIZE, fontweight="bold")
    ylab = quality_metric.replace("_mean", "").upper().replace("HR10", "HR@10").replace("RECALL200", "Recall@200")
    ax.set_ylabel(ylab, fontsize=AXES_FONTSIZE, fontweight="bold")
    ax.set_title("Quality vs time (pool_size)" + (f" — {title_suffix}" if title_suffix else ""), fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path_base = (out_path or OUTPUT_DIR / "quality_vs_time").with_suffix("")
    if out_path and out_path.suffix:
        path_base = out_path.with_suffix("")
    save_fig_paper(path_base)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot quality vs time and suggest sweet spot for pool_size.")
    parser.add_argument("--runs", type=Path, default=RUNS_LOG_PATH, help="Path to runs.jsonl")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR, help="Output directory for plots")
    parser.add_argument("--reranker", action="store_true", default=True, help="Use runs with reranker (default True)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.runs.exists():
        print(f"Runs file not found: {args.runs}")
        print("Run pool-size ablation first, e.g.: python -m src.run_ablation_pool_sizes --pool-sizes 200 500 1000 2000")
        return

    runs = load_runs(args.runs)
    rows = extract_pool_size_runs(runs, use_reranker=args.reranker)
    if not rows:
        print("No runs with pool_size and timing found. Ensure run_experiment logs retrieval_times and rerank_times.")
        return

    data = aggregate_by_pool_size(rows)
    if not data:
        print("No data after aggregating by pool_size.")
        return

    pool_eff, label_eff = sweet_spot_quality_per_time(data)
    pool_knee, label_knee = sweet_spot_knee(data)
    print("\n--- Sweet spot (quality vs time) ---")
    print(f"  Best efficiency (HR@10 per second): pool_size = {pool_eff} ({label_eff})")
    print(f"  Knee (marginal gain):              pool_size = {pool_knee} ({label_knee})")
    print("\nPer pool_size:")
    for p in sorted(data.keys()):
        v = data[p]
        t = v.get("total_time_per_user") or 0
        hr = v.get("hr10_mean") or 0
        r2 = v.get("recall200_mean") or 0
        print(f"  pool={p}: HR@10={hr:.4f}, Recall@200={r2:.4f}, time/user={t:.3f}s")

    summary = {
        "sweet_spot_efficiency": pool_eff,
        "sweet_spot_knee": pool_knee,
        "by_pool_size": data,
    }
    summary_path = args.out / "quality_vs_time_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {summary_path}")

    plot_quality_vs_time(data, quality_metric="hr10_mean", out_path=args.out / "quality_vs_time_hr10.png", title_suffix="HR@10")
    if any(data[p].get("recall200_mean") is not None for p in data):
        plot_quality_vs_time(data, quality_metric="recall200_mean", out_path=args.out / "quality_vs_time_recall200.png", title_suffix="Recall@200")


if __name__ == "__main__":
    main()
