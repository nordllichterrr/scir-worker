
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS_LOG_PATH = ROOT / "experiments" / "runs.jsonl"
OUTPUT_DIR = ROOT / "experiments" / "plots"


def load_runs(path: Path) -> List[Dict[str, Any]]:
    runs = []
    if not path.exists():
        return runs
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    runs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return runs


def run_sweep(
    n_users: int = 200,
    seeds: List[int] = None,
    alphas: List[float] = None,
    include_pareto_balanced: bool = True,
    dataset: str = "serendipity",
) -> None:
    from src.run_experiment import run_with_logging
    from src.evaluation_config import N_TEST_USERS

    if seeds is None:
        seeds = [42, 7]
    if alphas is None:
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    n_users = min(n_users, N_TEST_USERS) if n_users else N_TEST_USERS
    run_ids = []
    for alpha in alphas:
        for seed in seeds:
            run_id = f"pareto_alpha{alpha:.2f}_seed{seed}_n{n_users}"
            cfg = {
                "baseline": None,
                "use_reranker": True,
                "topk": 10,
                "candidate_pool_size": 500,
                "retrieval_mode": "ann",
                "dataset": dataset,
                "two_head_config": {"alpha": alpha, "mode": "scalarize"},
            }
            run_with_logging(run_id=run_id, n_users=n_users, seed=seed, config=cfg, dataset=dataset)
            run_ids.append(run_id)
    if include_pareto_balanced:
        for seed in seeds:
            run_id = f"pareto_pareto_balanced_seed{seed}_n{n_users}"
            cfg = {
                "baseline": None,
                "use_reranker": True,
                "topk": 10,
                "candidate_pool_size": 500,
                "retrieval_mode": "ann",
                "dataset": dataset,
                "two_head_config": {"mode": "pareto_balanced"},
            }
            run_with_logging(run_id=run_id, n_users=n_users, seed=seed, config=cfg, dataset=dataset)
            run_ids.append(run_id)
    print("Pareto sweep run_ids:", run_ids)


def extract_pareto_runs(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in runs:
        cfg = r.get("config", {})
        th = cfg.get("two_head_config")
        if not th or not isinstance(th, dict):
            continue
        diag = r.get("diagnostics", {})
        if not diag:
            continue
        metrics = r.get("metrics", {})
        hr = metrics.get("hr@10", {})
        ndcg = metrics.get("ndcg@10", {})
        alpha = th.get("alpha")
        mode = th.get("mode", "scalarize")
        label = f"α={alpha}" if alpha is not None else f"mode={mode}"
        out.append({
            "run_id": r.get("run_id", ""),
            "config": cfg,
            "diagnostics": diag,
            "metrics": metrics,
            "alpha": alpha,
            "mode": mode,
            "label": label,
            "hr_mean": hr.get("mean"),
            "ndcg_mean": ndcg.get("mean"),
            "coverage": diag.get("coverage"),
            "mean_popularity_rank": diag.get("mean_popularity_rank"),
        })
    return out


def aggregate_by_alpha(runs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_key: Dict[str, List[Dict]] = {}
    for r in runs:
        if r.get("alpha") is not None:
            key = f"alpha_{r['alpha']}"
        else:
            key = "pareto_balanced"
        by_key.setdefault(key, []).append(r)
    agg = {}
    for key, group in by_key.items():
        hr_vals = [x["hr_mean"] for x in group if x.get("hr_mean") is not None]
        ndcg_vals = [x["ndcg_mean"] for x in group if x.get("ndcg_mean") is not None]
        cov_vals = [x["coverage"] for x in group if x.get("coverage") is not None]
        mpr_vals = [x["mean_popularity_rank"] for x in group if x.get("mean_popularity_rank") is not None]
        agg[key] = {
            "label": group[0]["label"] if group else key,
            "hr_mean": float(np.mean(hr_vals)) if hr_vals else None,
            "hr_std": float(np.std(hr_vals)) if len(hr_vals) > 1 else None,
            "ndcg_mean": float(np.mean(ndcg_vals)) if ndcg_vals else None,
            "ndcg_std": float(np.std(ndcg_vals)) if len(ndcg_vals) > 1 else None,
            "coverage_mean": float(np.mean(cov_vals)) if cov_vals else None,
            "coverage_std": float(np.std(cov_vals)) if len(cov_vals) > 1 else None,
            "mean_pop_rank_mean": float(np.mean(mpr_vals)) if mpr_vals else None,
            "mean_pop_rank_std": float(np.std(mpr_vals)) if len(mpr_vals) > 1 else None,
            "n_runs": len(group),
        }
    return agg


def plot_pareto_curve(
    runs_path: Path = RUNS_LOG_PATH,
    out_dir: Path = OUTPUT_DIR,
    x_metric: str = "hr@10",
    y_metric: str = "coverage",
) -> None:
    import matplotlib.pyplot as plt

    runs = load_runs(runs_path)
    pareto_runs = extract_pareto_runs(runs)
    if not pareto_runs:
        print("No Pareto runs found (config.two_head_config with alpha or mode=pareto_balanced).")
        return
    agg = aggregate_by_alpha(pareto_runs)
    keys_order = sorted(
        [k for k in agg if k.startswith("alpha_")],
        key=lambda k: float(k.replace("alpha_", "")),
    ) + ([k for k in agg if k == "pareto_balanced"] or [])

    out_dir.mkdir(parents=True, exist_ok=True)


    x_means = []
    y_means = []
    x_stds = []
    y_stds = []
    labels = []
    for k in keys_order:
        v = agg[k]
        if x_metric == "hr@10":
            x_means.append(v.get("hr_mean"))
            x_stds.append(v.get("hr_std"))
        else:
            x_means.append(v.get("ndcg_mean"))
            x_stds.append(v.get("ndcg_std"))
        if y_metric == "coverage":
            y_means.append(v.get("coverage_mean"))
            y_stds.append(v.get("coverage_std"))
        else:
            y_means.append(v.get("mean_pop_rank_mean"))
            y_stds.append(v.get("mean_pop_rank_std"))
        labels.append(v.get("label", k))

    valid = [i for i in range(len(x_means)) if x_means[i] is not None and y_means[i] is not None]
    if not valid:
        print("Not enough data to plot (need both x and y metrics).")
        return
    x_means = np.array([x_means[i] for i in valid])
    y_means = np.array([y_means[i] for i in valid])
    x_stds = [x_stds[i] or 0 for i in valid]
    y_stds = [y_stds[i] or 0 for i in valid]
    labels = [labels[i] for i in valid]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(x_means, y_means, xerr=x_stds, yerr=y_stds, fmt="o-", capsize=4, linewidth=2, markersize=8)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (x_means[i], y_means[i]), textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.set_xlabel("Relevance (" + ("HR@10" if x_metric == "hr@10" else "nDCG@10") + ")", fontsize=12)
    ax.set_ylabel("Novelty / diversity (" + ("Coverage" if y_metric == "coverage" else "Mean popularity rank") + ")", fontsize=12)
    ax.set_title("Pareto: Relevance vs Novelty (two-headed reranker)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_file = out_dir / f"pareto_curve_{x_metric}_{y_metric}.png"
    fig.savefig(out_file, dpi=150)
    plt.close()
    print("Saved", out_file)


def main():
    ap = argparse.ArgumentParser(description="Pareto sweep (relevance vs novelty) and plot.")
    ap.add_argument("--run", action="store_true", help="Run experiments for multiple alphas")
    ap.add_argument("--plot", action="store_true", help="Plot Pareto curve from runs.jsonl")
    ap.add_argument("--n-users", type=int, default=200, help="Users per run (sweep)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7], help="Seeds for sweep")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0], help="Alpha values")
    ap.add_argument("--no-pareto-balanced", action="store_true", help="Skip mode=pareto_balanced run")
    ap.add_argument("--dataset", type=str, default="serendipity", choices=["serendipity", "taobao"], help="Dataset")
    ap.add_argument("--runs", type=Path, default=RUNS_LOG_PATH, help="runs.jsonl path for --plot")
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR, help="Output dir for plot")
    ap.add_argument("--x", choices=["hr@10", "ndcg@10"], default="hr@10", help="X axis metric")
    ap.add_argument("--y", choices=["coverage", "mean_pop_rank"], default="coverage", help="Y axis metric")
    args = ap.parse_args()

    if args.run:
        run_sweep(
            n_users=args.n_users,
            seeds=args.seeds,
            alphas=args.alphas,
            include_pareto_balanced=not args.no_pareto_balanced,
            dataset=args.dataset,
        )
    if args.plot:
        plot_pareto_curve(runs_path=args.runs, out_dir=args.out, x_metric=args.x, y_metric=args.y)
    if not args.run and not args.plot:
        ap.print_help()
        print("\nUse --run to run sweep, --plot to plot Pareto curve (or both).")


if __name__ == "__main__":
    main()
