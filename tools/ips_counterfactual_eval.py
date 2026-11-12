
import json
import math
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple
import numpy as np

RUNS_JSONL = Path("experiments") / "runs.jsonl"
RESULTS_DIR = Path("results")
GT_PATH = Path("experiments") / "ground_truth.json"
OUT_DIR = Path("experiments") / "counterfactual_evaluation"
OUT_JSON = OUT_DIR / "ips_snips_results.json"
OUT_MD = OUT_DIR / "ips_snips_report.md"
K = 10


def _position_propensity(k: int = K) -> List[float]:
    p = [1.0 / math.log2(1 + r) for r in range(1, k + 1)]
    s = sum(p)
    return [x / s for x in p]


def load_gt() -> Dict[str, List[str]]:
    if not GT_PATH.exists():
        return {}
    with open(GT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("data", data) if isinstance(data, dict) else {}


def load_run_results(run_id: str) -> Dict[str, List[Dict]]:
    p = RESULTS_DIR / f"{run_id}.json"
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("results", data)


def get_config_key(run: Dict[str, Any]) -> str:
    c = run.get("config", {})
    baseline = c.get("baseline")
    use_reranker = c.get("use_reranker", False)
    pool_size = c.get("candidate_pool_size", 1000)
    if baseline:
        return f"{baseline}_no_reranker"
    return f"ours_with_reranker_pool{pool_size}" if use_reranker else f"candidates_only_pool{pool_size}"


def hr_at_k_ips_snips_per_user(
    rec_ids: List[str],
    gt_set: set,
    propensities: List[float],
    k: int = K,
) -> Tuple[float, float, float, float]:
    ips_hr = 0.0
    inv_p_sum = 0.0
    ips_dcg = 0.0
    for r, iid in enumerate(rec_ids[:k]):
        if r >= len(propensities):
            break
        p_r = propensities[r]
        rel = 1.0 if iid in gt_set else 0.0
        inv_p = 1.0 / max(p_r, 1e-9)
        ips_hr += rel * inv_p
        inv_p_sum += inv_p
        gain = rel / math.log2(r + 2)
        ips_dcg += gain * inv_p
    snips_hr = ips_hr / inv_p_sum if inv_p_sum > 0 else 0.0
    snips_dcg = ips_dcg / inv_p_sum if inv_p_sum > 0 else 0.0
    return ips_hr, inv_p_sum, ips_dcg, inv_p_sum


def ndcg_at_k_binary(rec_ids: List[str], gt_set: set, k: int = K) -> float:
    """
    Binary nDCG@k for IPS/SNIPS (no need to normalise by IDCG here if
    all methods are compared consistently; keep DCG as-is).
    """
    dcg = 0.0
    for r, iid in enumerate(rec_ids[:k]):
        rel = 1.0 if iid in gt_set else 0.0
        dcg += rel / math.log2(r + 2)
    return dcg


def build_synthetic_log_from_runs(
    runs: List[Dict],
    gt: Dict[str, List[str]],
    propensities: List[float],
) -> Dict[str, Dict[str, Any]]:
    by_config: Dict[str, List[Tuple[str, float, float, float, float]]] = defaultdict(list)
    for run in runs:
        run_id = run.get("run_id", "")
        results = load_run_results(run_id)
        if not results:
            continue
        key = get_config_key(run)
        for uid, rec_list in results.items():
            rec_ids = [str(x.get("item_id", x)) for x in (rec_list or []) if x.get("item_id") is not None]
            gt_set = set(str(x) for x in gt.get(uid, []))
            if not rec_ids:
                continue
            ips_hr, denom_hr, ips_dcg, denom_dcg = hr_at_k_ips_snips_per_user(rec_ids, gt_set, propensities, K)
            by_config[key].append((uid, ips_hr, denom_hr, ips_dcg, denom_dcg))
    return by_config


def compute_ips_snips(
    by_config: Dict[str, List[Tuple[str, float, float, float, float]]],
    n_bootstrap: int = 500,
) -> Dict[str, Dict[str, Any]]:

    out = {}
    for key, rows in by_config.items():
        if not rows:
            continue
        n = len(rows)
        ips_hr_vals = [r[1] for r in rows]
        denom_hr = [r[2] for r in rows]
        ips_dcg_vals = [r[3] for r in rows]
        denom_dcg = [r[4] for r in rows]

        ips_hr_mean = np.mean(ips_hr_vals)
   
        total_ips_hr = sum(ips_hr_vals)
        total_denom_hr = sum(denom_hr)
        snips_hr_mean = total_ips_hr / total_denom_hr if total_denom_hr > 0 else 0.0
        ips_dcg_mean = np.mean(ips_dcg_vals)
        total_ips_dcg = sum(ips_dcg_vals)
        total_denom_dcg = sum(denom_dcg)
        snips_dcg_mean = total_ips_dcg / total_denom_dcg if total_denom_dcg > 0 else 0.0

        rng = np.random.default_rng(42)
        ips_hr_boot, snips_hr_boot, ips_dcg_boot, snips_dcg_boot = [], [], [], []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            b_ips_hr = np.mean([ips_hr_vals[i] for i in idx])
            b_denom_hr = sum(denom_hr[i] for i in idx)
            b_ips_hr_sum = sum(ips_hr_vals[i] for i in idx)
            snips_hr_boot.append(b_ips_hr_sum / b_denom_hr if b_denom_hr > 0 else 0.0)
            ips_hr_boot.append(b_ips_hr)
            b_ips_dcg = np.mean([ips_dcg_vals[i] for i in idx])
            b_denom_dcg = sum(denom_dcg[i] for i in idx)
            b_ips_dcg_sum = sum(ips_dcg_vals[i] for i in idx)
            snips_dcg_boot.append(b_ips_dcg_sum / b_denom_dcg if b_denom_dcg > 0 else 0.0)
            ips_dcg_boot.append(b_ips_dcg)
        out[key] = {
            "n_users": n,
            "IPS_HR@10": {"mean": float(ips_hr_mean), "ci_95_lower": float(np.percentile(ips_hr_boot, 2.5)), "ci_95_upper": float(np.percentile(ips_hr_boot, 97.5))},
            "SNIPS_HR@10": {"mean": float(snips_hr_mean), "ci_95_lower": float(np.percentile(snips_hr_boot, 2.5)), "ci_95_upper": float(np.percentile(snips_hr_boot, 97.5))},
            "IPS_nDCG@10": {"mean": float(ips_dcg_mean), "ci_95_lower": float(np.percentile(ips_dcg_boot, 2.5)), "ci_95_upper": float(np.percentile(ips_dcg_boot, 97.5))},
            "SNIPS_nDCG@10": {"mean": float(snips_dcg_mean), "ci_95_lower": float(np.percentile(snips_dcg_boot, 2.5)), "ci_95_upper": float(np.percentile(snips_dcg_boot, 97.5))},
        }
    return out


def run_from_logs_csv(logs_path: Path) -> Dict[str, Dict[str, Any]]:
    import csv
    rows_by_var: Dict[str, List[Tuple[float, float]]] = defaultdict(list)  # (clicked/p, 1/p) per impression
    with open(logs_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clicked = float(row.get("clicked", 0))
            prop = float(row.get("propensity", 1.0))
            var = row.get("model_variant", "default")
            if prop <= 0:
                continue
            rows_by_var[var].append((clicked / prop, 1.0 / prop))
    out = {}
    for var, pairs in rows_by_var.items():
        if not pairs:
            continue
        w = [p[0] for p in pairs]
        n = len(pairs)
        ips_mean = np.mean(w)
        snips_mean = sum(p[0] for p in pairs) / sum(p[1] for p in pairs) if sum(p[1] for p in pairs) > 0 else 0.0
        out[var] = {
            "n_impressions": n,
            "IPS_HR@10": {"mean": float(ips_mean)},  # simplified: treat as CTR-style
            "SNIPS_HR@10": {"mean": float(snips_mean)},
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="IPS/SNIPS counterfactual evaluation")
    parser.add_argument("--logs", type=str, default=None, help="Optional logs.csv with user_id, item_id, clicked, propensity [, model_variant]")
    parser.add_argument("--bootstrap", type=int, default=500, help="Bootstrap samples for CI")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    propensities = _position_propensity(K)
    if args.logs and Path(args.logs).exists():
        results = run_from_logs_csv(Path(args.logs))
        print("Loaded logs from", args.logs)
    else:
        gt = load_gt()
        if not gt:
            print("No ground_truth.json found. Run create_splits first.")
            return
        runs = []
        if RUNS_JSONL.exists():
            with open(RUNS_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            runs.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        if not runs:
            print("No runs in runs.jsonl. Run experiments first.")
            return
        seen_config = set()
        config_aggregated: Dict[str, List[Tuple]] = defaultdict(list)
        for run in runs:
            key = get_config_key(run)
            if key in seen_config:
                continue
            seen_config.add(key)
            res = load_run_results(run.get("run_id", ""))
            if not res:
                continue
            for uid, rec_list in res.items():
                rec_ids = [str(x.get("item_id", x)) for x in (rec_list or []) if x.get("item_id") is not None]
                gt_set = set(str(x) for x in gt.get(uid, []))
                if not rec_ids:
                    continue
                ips_hr, d_hr, ips_dcg, d_dcg = hr_at_k_ips_snips_per_user(rec_ids, gt_set, propensities, K)
                config_aggregated[key].append((uid, ips_hr, d_hr, ips_dcg, d_dcg))
        results = compute_ips_snips(dict(config_aggregated), n_bootstrap=args.bootstrap)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"propensity": "position_bias_1_log2_rank", "K": K, "results": results}, f, indent=2)
    lines = ["# IPS/SNIPS Counterfactual Evaluation", "", "Estimates of HR@10 and nDCG@10 with inverse propensity weighting (position bias).", ""]
    for key, data in results.items():
        lines.append(f"## {key}")
        lines.append(f"- n_users: {data.get('n_users', data.get('n_impressions', 'N/A'))}")
        for metric in ["IPS_HR@10", "SNIPS_HR@10", "IPS_nDCG@10", "SNIPS_nDCG@10"]:
            if metric not in data:
                continue
            m = data[metric]
            mean = m.get("mean", 0)
            ci_lo, ci_hi = m.get("ci_95_lower"), m.get("ci_95_upper")
            if ci_lo is not None and ci_hi is not None:
                lines.append(f"- **{metric}**: {mean:.4f} [95% CI: {ci_lo:.4f}, {ci_hi:.4f}]")
            else:
                lines.append(f"- **{metric}**: {mean:.4f}")
        lines.append("")
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"IPS/SNIPS results saved to {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
