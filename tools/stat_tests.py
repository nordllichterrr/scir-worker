import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np

from src.stats import bootstrap_ci, paired_bootstrap_test, paired_permutation_test, adjust_pvalues

RUNS_LOG_PATH = Path("experiments") / "runs.jsonl"
OUTPUT_DIR = Path("experiments") / "stat_tests"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_runs() -> List[Dict[str, Any]]:
    if not RUNS_LOG_PATH.exists():
        return []
    out = []
    with open(RUNS_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _model_key(run: Dict[str, Any]) -> str:
    cfg = run.get("config", {}) or {}
    baseline = cfg.get("baseline")
    if baseline:
        return str(baseline)
    return "ours_with_reranker" if cfg.get("use_reranker", False) else "candidates_only"


def _metric_col(metric: str) -> str:
    return metric


def _read_per_user(run_id: str, metric: str) -> Dict[str, float]:
    p = Path("experiments") / f"{run_id}_per_user_metrics.csv"
    if not p.exists():
        p = Path("experiments") / f"{run_id}_per_user.csv"
    if not p.exists():
        return {}
    out: Dict[str, float] = {}
    with open(p, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            uid = str(row.get("user") or row.get("user_id") or "")
            if not uid:
                continue
            try:
                out[uid] = float(row.get(_metric_col(metric)))
            except Exception:
                continue
    return out


def _pairwise_user_vectors(run_a: Dict[str, Any], run_b: Dict[str, Any], metric: str) -> Tuple[List[float], List[float]]:
    ua = _read_per_user(str(run_a.get("run_id")), metric=metric)
    ub = _read_per_user(str(run_b.get("run_id")), metric=metric)
    users = sorted(set(ua.keys()) & set(ub.keys()))
    return [ua[u] for u in users], [ub[u] for u in users]


def run_tests(reference_model: str = "ours_with_reranker", metrics: List[str] = None):
    if metrics is None:
        metrics = ["hr@10", "ndcg@10", "mrr@10", "map@10"]
    runs = _load_runs()
    runs_by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in runs:
        runs_by_model[_model_key(r)].append(r)
    if reference_model not in runs_by_model:
        raise SystemExit(f"Reference model '{reference_model}' not found in runs.jsonl")

    comparisons: List[Dict[str, Any]] = []
    all_pvals_boot = []
    all_pvals_perm = []

    for model, model_runs in runs_by_model.items():
        if model == reference_model:
            continue
        # Pair by exact experimental design keys except model identity.
        ref_by_key = {}
        for rr in runs_by_model[reference_model]:
            cfg = rr.get("config", {}) or {}
            key = (
                cfg.get("dataset"),
                cfg.get("seed"),
                cfg.get("split_seed"),
                cfg.get("init_seed"),
                cfg.get("n_users"),
                cfg.get("topk"),
            )
            ref_by_key[key] = rr

        for mr in model_runs:
            cfgm = mr.get("config", {}) or {}
            key = (
                cfgm.get("dataset"),
                cfgm.get("seed"),
                cfgm.get("split_seed"),
                cfgm.get("init_seed"),
                cfgm.get("n_users"),
                cfgm.get("topk"),
            )
            rr = ref_by_key.get(key)
            if rr is None:
                continue
            for metric in metrics:
                a_vals, b_vals = _pairwise_user_vectors(rr, mr, metric=metric)
                if not a_vals or not b_vals:
                    continue
                a_mean = float(np.mean(a_vals))
                b_mean = float(np.mean(b_vals))
                a_lo, a_hi = bootstrap_ci(a_vals, n_boot=2000, alpha=0.05, seed=42)
                b_lo, b_hi = bootstrap_ci(b_vals, n_boot=2000, alpha=0.05, seed=42)
                boot = paired_bootstrap_test(a_vals, b_vals, n_boot=2000, alpha=0.05, seed=42)
                perm = paired_permutation_test(a_vals, b_vals, n_perm=5000, seed=42)
                row = {
                    "reference_model": reference_model,
                    "compared_model": model,
                    "dataset": cfgm.get("dataset"),
                    "seed": cfgm.get("seed"),
                    "split_seed": cfgm.get("split_seed"),
                    "init_seed": cfgm.get("init_seed"),
                    "metric": metric,
                    "ref_mean": a_mean,
                    "ref_ci95_low": a_lo,
                    "ref_ci95_high": a_hi,
                    "cmp_mean": b_mean,
                    "cmp_ci95_low": b_lo,
                    "cmp_ci95_high": b_hi,
                    "delta_mean": boot["mean_diff"],
                    "ci95_delta_low": boot["ci95_low"],
                    "ci95_delta_high": boot["ci95_high"],
                    "p_value": boot["p_value"],
                    "p_value_permutation": perm["p_value"],
                    "n_users": boot["n_users"],
                    "table_ref": f"{a_mean:.3f} [{a_lo:.3f}, {a_hi:.3f}]",
                    "table_delta": f"{boot['mean_diff']:+.3f} [{boot['ci95_low']:+.3f}, {boot['ci95_high']:+.3f}], p={boot['p_value']:.4f}",
                }
                comparisons.append(row)
                all_pvals_boot.append(float(boot["p_value"]))
                all_pvals_perm.append(float(perm["p_value"]))

    if not comparisons:
        raise SystemExit("No comparable run pairs with per-user metrics found.")

    boot_holm = adjust_pvalues(all_pvals_boot, method="holm")
    boot_bh = adjust_pvalues(all_pvals_boot, method="bh")
    perm_holm = adjust_pvalues(all_pvals_perm, method="holm")
    perm_bh = adjust_pvalues(all_pvals_perm, method="bh")

    for i, row in enumerate(comparisons):
        row["p_value_holm"] = boot_holm[i]
        row["p_value_bh"] = boot_bh[i]
        row["p_perm_holm"] = perm_holm[i]
        row["p_perm_bh"] = perm_bh[i]

    out_json = OUTPUT_DIR / "stat_test_results.json"
    out_csv = OUTPUT_DIR / "stat_test_results.csv"
    out_md = OUTPUT_DIR / "stat_test_table.md"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"reference_model": reference_model, "results": comparisons}, f, ensure_ascii=False, indent=2)

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparisons[0].keys()))
        writer.writeheader()
        writer.writerows(comparisons)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("| metric | reference | compared | point estimate | delta |\n")
        f.write("|---|---|---|---|---|\n")
        for r in comparisons:
            f.write(
                f"| {r['metric']} | {r['reference_model']} | {r['compared_model']} | "
                f"{r['table_ref']} | {r['table_delta']} |\n"
            )
    print(f"Saved: {out_json}")
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_md}")


def main():
    print("Running centralized statistical tests (paired bootstrap + permutation + corrections)...")
    run_tests()


if __name__ == "__main__":
    main()
