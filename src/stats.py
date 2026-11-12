from typing import Dict, List, Any

import numpy as np


def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05, seed: int = 42):
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return None, None
    rng = np.random.default_rng(seed)
    n = arr.size
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(np.mean(arr[idx]))
    lo = float(np.percentile(boots, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(boots, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def paired_bootstrap_test(a, b, n_boot: int = 2000, alpha: float = 0.05, seed: int = 42):
    a_arr = np.asarray(list(a), dtype=float)
    b_arr = np.asarray(list(b), dtype=float)
    n = min(a_arr.size, b_arr.size)
    if n == 0:
        return {"mean_diff": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "p_one_sided": 1.0, "n_users": 0}
    diffs = a_arr[:n] - b_arr[:n]
    mean_diff = float(np.mean(diffs))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(np.mean(diffs[idx]))
    ci_low = float(np.percentile(boots, 100.0 * (alpha / 2.0)))
    ci_high = float(np.percentile(boots, 100.0 * (1.0 - alpha / 2.0)))
    p_one_sided = float(np.mean(boots <= 0.0))
    return {
        "mean_diff": mean_diff,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "p_one_sided": p_one_sided,
        "p_value": float(2.0 * min(p_one_sided, 1.0 - p_one_sided)),
        "n_users": int(n),
    }


def paired_permutation_test(a, b, n_perm: int = 5000, seed: int = 42):
    a_arr = np.asarray(list(a), dtype=float)
    b_arr = np.asarray(list(b), dtype=float)
    n = min(a_arr.size, b_arr.size)
    if n == 0:
        return {"mean_diff": 0.0, "p_value": 1.0, "n_users": 0}
    diffs = a_arr[:n] - b_arr[:n]
    observed = float(np.mean(diffs))
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice(np.array([1.0, -1.0], dtype=float), size=n)
        perm_mean = float(np.mean(diffs * signs))
        if abs(perm_mean) >= abs(observed):
            count += 1
    p_val = float((count + 1) / (n_perm + 1))
    return {"mean_diff": observed, "p_value": p_val, "n_users": int(n)}


def adjust_pvalues(p_values, method: str = "holm"):
    p = np.asarray(list(p_values), dtype=float)
    m = p.size
    if m == 0:
        return []
    method = (method or "holm").lower()
    if method in {"bh", "fdr_bh", "benjamini-hochberg", "benjamini_hochberg"}:
        order = np.argsort(p)
        ranks = np.arange(1, m + 1, dtype=float)
        q = p[order] * m / ranks
        q = np.minimum.accumulate(q[::-1])[::-1]
        out = np.empty(m, dtype=float)
        out[order] = np.clip(q, 0.0, 1.0)
        return out.tolist()
    # Holm-Bonferroni default
    order = np.argsort(p)
    out = np.empty(m, dtype=float)
    prev = 0.0
    for i, idx in enumerate(order):
        adj = (m - i) * p[idx]
        adj = max(prev, adj)
        prev = adj
        out[idx] = min(1.0, adj)
    return out.tolist()


def aggregate_user_metrics_with_ci(df_per_user: List[Dict[str, Any]], metric_cols: List[str] = None, n_boot: int = 2000, alpha: float = 0.05, seed: int = 42) -> Dict[str, Dict[str, float]]:
    if not df_per_user:
        return {}
    if metric_cols is None:
        metric_cols = []
        for k in df_per_user[0].keys():
            if k in {"user", "user_id", "rec_ids", "hits", "hit_bool"}:
                continue
            v = df_per_user[0].get(k)
            if isinstance(v, (float, int)):
                metric_cols.append(k)
    out: Dict[str, Dict[str, float]] = {}
    for m in metric_cols:
        vals = []
        for row in df_per_user:
            try:
                vals.append(float(row.get(m)))
            except Exception:
                continue
        if not vals:
            out[m] = {"mean": None, "std": None, "ci95_low": None, "ci95_high": None, "n_users": 0}
            continue
        lo, hi = bootstrap_ci(vals, n_boot=n_boot, alpha=alpha, seed=seed)
        out[m] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "ci95_low": lo,
            "ci95_high": hi,
            "n_users": len(vals),
        }
    return out
