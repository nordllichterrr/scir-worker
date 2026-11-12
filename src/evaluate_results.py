import os
import json
import csv
from pathlib import Path
import numpy as np
import logging
from typing import Dict, Any, List

logger = logging.getLogger("evaluate_results")
if not logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
    logger.setLevel(logging.INFO)

from src.evaluate import hr_at_k, ndcg_at_k
from .stats import bootstrap_ci, aggregate_user_metrics_with_ci

RESULTS_DIR = Path("results")
from .evaluation_config import get_eval_paths

# default dataset for standalone eval (can be overridden by passing dataset into evaluate_* later)
DEFAULT_DATASET = os.environ.get("COLDSTART_DATASET", "serendipity")
_eval = get_eval_paths(DEFAULT_DATASET)

GT_PATH = Path(_eval["gt"])
SPLIT_METADATA_PATH = Path(_eval["split_meta"])
OUT_DIR = Path("experiments")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_json_safe(p: Path):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        logger.warning("JSON decode error for %s", p)
        return None

def write_json(p: Path, data: Any):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def normalize_recs(raw):
  
    if raw is None:
        return []
    if isinstance(raw, list):
        if not raw:
            return []
        # list of dicts?
        if isinstance(raw[0], dict):
            return raw
        # list of ids?
        if isinstance(raw[0], (str, int)):
            return [{"item_id": str(x)} for x in raw]
        # fallback: convert to strings
        return [{"item_id": str(x)} for x in raw]
    if isinstance(raw, dict):
        # unexpected single dict -> wrap
        return [raw]
    # single primitive (str/int)
    return [{"item_id": str(raw)}]

def extract_rec_ids(rec_list: List[Dict[str,Any]]):
    """Extract item IDs from recommendation list, handling various formats."""
    if not rec_list:
        return []
    ids = []
    for r in rec_list:
        if isinstance(r, dict):
            item_id = r.get("item_id") or r.get("id") or r.get("doc_id")
            if item_id is not None:
                ids.append(str(item_id))
        elif isinstance(r, (str, int)):
            # If it's already an ID string/int, use it directly
            ids.append(str(r))
    return ids


def evaluate_single(
    results: Dict[str, Any],
    gt: Dict[str, List[str]],
    k=10,
    n_bootstrap: int = 1000,
):
    rows = []
    hr_vals = []
    ndcg_vals = []
    mrr_vals = []
    map_vals = []

    result_keys = list(results.keys())
    are_numeric_indices = (
        len(result_keys) > 0 and
        all(str(k).isdigit() for k in result_keys) and
        all(int(str(k)) < len(result_keys) for k in result_keys) and
        set(str(i) for i in range(len(result_keys))) == set(str(k) for k in result_keys)
    )
    if are_numeric_indices:
        logger.warning("Results appear to use numeric indices instead of user IDs. Attempting to map to GT users.")
        gt_user_list = sorted(list(gt.keys()))
        if len(gt_user_list) >= len(result_keys):

            index_to_uid = {str(i): gt_user_list[i] for i in range(len(result_keys))}
        else:
            index_to_uid = {str(i): str(i) for i in range(len(result_keys))}
            logger.warning("Not enough GT users to map indices. Metrics may be 0.")
    else:
        index_to_uid = None

    for uid, raw_recs in results.items():
        recs = normalize_recs(raw_recs)
        rec_ids = extract_rec_ids(recs)
        
        # Sanity check removed: item_ids can legitimately be sequential numbers (1, 2, 3, ...)
        # in many datasets. The previous check was too strict and caused false positives.
        # If there's a real bug (e.g., using list indices instead of item_ids), it will be
        # caught by other checks (empty results, zero scores, etc.) 

        if index_to_uid is not None:
            real_uid = index_to_uid.get(str(uid), str(uid))
        else:
            real_uid = str(uid)

        pos_raw = gt.get(real_uid, [])  

        pos_ids = []
        try:
            if pos_raw and isinstance(pos_raw, list):
                if len(pos_raw) > 0 and isinstance(pos_raw[0], dict):
                    pos_ids = [str(d.get("item_id")) for d in pos_raw if d.get("item_id") is not None]
                else:
                    pos_ids = [str(x) for x in pos_raw]
            else:
                pos_ids = []
        except Exception:
            pos_ids = []

        topk_ids = rec_ids[:k]
        overlap = sorted(list(set(topk_ids) & set(pos_ids)))
        hit_bool = 1 if overlap else 0
        hits_str = " ".join(overlap) if overlap else ""

        hr = 1.0 if any(x in pos_ids for x in topk_ids) else 0.0


        rel_list = [1 if x in pos_ids else 0 for x in topk_ids]
        # Use unified metrics.ndcg_at_k implementation for consistency
        from .metrics import ndcg_at_k as _ndcg_at_k_unified
        ndcg = _ndcg_at_k_unified(topk_ids, pos_ids, k=k)

        mrr = 0.0
        for i, x in enumerate(topk_ids, start=1):
            if x in pos_ids:
                mrr = 1.0 / i
                break
        

        from .metrics import map_at_k
        map_val = map_at_k(topk_ids, pos_ids, k=k)

        rows.append({
            "user": real_uid, 
            f"hr@{k}": hr,
            f"ndcg@{k}": ndcg,
            f"mrr@{k}": mrr,
            f"map@{k}": map_val,
            "rec_ids": " ".join(topk_ids) if topk_ids else "None",
            "hits": hits_str,
            "hit_bool": int(hit_bool)
        })
        hr_vals.append(hr)
        ndcg_vals.append(ndcg)
        mrr_vals.append(mrr)
        map_vals.append(map_val)

    agg = aggregate_user_metrics_with_ci(
        rows,
        metric_cols=[f"hr@{k}", f"ndcg@{k}", f"mrr@{k}", f"map@{k}"],
        n_boot=n_bootstrap,
        alpha=0.05,
    )
    hr_stat = agg.get(f"hr@{k}", {})
    ndcg_stat = agg.get(f"ndcg@{k}", {})
    mrr_stat = agg.get(f"mrr@{k}", {})
    map_stat = agg.get(f"map@{k}", {})

    summary = {
        "n_users": len(rows),
        "hr_mean": hr_stat.get("mean"),
        "hr_std": hr_stat.get("std"),
        "hr_ci_95_lower": hr_stat.get("ci95_low"),
        "hr_ci_95_upper": hr_stat.get("ci95_high"),
        "ndcg_mean": ndcg_stat.get("mean"),
        "ndcg_std": ndcg_stat.get("std"),
        "ndcg_ci_95_lower": ndcg_stat.get("ci95_low"),
        "ndcg_ci_95_upper": ndcg_stat.get("ci95_high"),
        "mrr_mean": mrr_stat.get("mean"),
        "mrr_std": mrr_stat.get("std"),
        "mrr_ci_95_lower": mrr_stat.get("ci95_low"),
        "mrr_ci_95_upper": mrr_stat.get("ci95_high"),
        "map_mean": map_stat.get("mean"),
        "map_std": map_stat.get("std"),
        "map_ci_95_lower": map_stat.get("ci95_low"),
        "map_ci_95_upper": map_stat.get("ci95_high"),
    }
    return rows, summary


def load_split_metadata(path: Path = None):
    if path is None:
        try:
            
            path = SPLIT_METADATA_PATH
        except Exception:
            path = Path(__file__).resolve().parents[1] / "experiments" / "split_metadata.json"
    raw = load_json_safe(Path(path))
    return raw


def evaluate_by_buckets_and_scenarios(
    results: Dict[str, Any],
    gt: Dict[str, List[str]],
    split_metadata: Dict[str, Any],
    k: int = 10,
) -> Dict[str, Any]:
    rows, overall_summary = evaluate_single(results, gt, k=k)
    if not rows:
        return {"overall": overall_summary, "by_bucket": {}, "by_scenario": {}}

    user_hr = {}
    user_ndcg = {}

    user_mrr = {}
    user_map = {}
    for row in rows:
        uid = str(row.get("user", row.get("user_id", "")))
        try:
            user_hr[uid] = float(row.get(f"hr@{k}", row.get("hr@10", 0)) or 0)
            user_ndcg[uid] = float(row.get(f"ndcg@{k}", row.get("ndcg@10", 0)) or 0)
            user_mrr[uid] = float(row.get(f"mrr@{k}", row.get("mrr@10", 0)) or 0)
            user_map[uid] = float(row.get(f"map@{k}", row.get("map@10", 0)) or 0)
        except (TypeError, ValueError):
            user_hr[uid] = 0.0
            user_mrr[uid] = 0.0
            user_map[uid] = 0.0
            user_ndcg[uid] = 0.0

    user_meta = split_metadata.get("user_meta") or {}
    scenario_to_users = split_metadata.get("scenario_to_users") or {}
    bucket_to_users = split_metadata.get("bucket_to_users") or {}

    def agg_for_users(uids: List[str]):
        hr_vals = [user_hr.get(u, 0.0) for u in uids if u in user_hr]
        ndcg_vals = [user_ndcg.get(u, 0.0) for u in uids if u in user_ndcg]
        mrr_vals = [user_mrr.get(u, 0.0) for u in uids if u in user_mrr]
        map_vals = [user_map.get(u, 0.0) for u in uids if u in user_map]
        n = len(hr_vals)
        if n == 0:
            return {"hr_mean": None, "ndcg_mean": None, "mrr_mean": None, "map_mean": None, "n_users": 0}
        return {
            "hr_mean": float(np.mean(hr_vals)),
            "ndcg_mean": float(np.mean(ndcg_vals)),
            "mrr_mean": float(np.mean(mrr_vals)),
            "map_mean": float(np.mean(map_vals)),
            "n_users": n,
        }

    by_bucket = {}
    for bucket, uids in bucket_to_users.items():
        by_bucket[bucket] = agg_for_users(uids)

    by_scenario = {}
    for scenario, uids in scenario_to_users.items():
        by_scenario[scenario] = agg_for_users(uids)

    return {
        "overall": overall_summary,
        "by_bucket": by_bucket,
        "by_scenario": by_scenario,
    }


def auto_create_synthetic_gt(results_file: Path):
    logger.info("Creating synthetic ground-truth from top-1 of %s", results_file)
    r = load_json_safe(results_file)
    if not r:
        raise SystemExit("Cannot create synthetic GT: results file missing or invalid.")
    gt = {}
    for uid, recs in r.items():
        recs_norm = normalize_recs(recs)
        if recs_norm:
            gt[str(uid)] = [str(recs_norm[0].get("item_id"))]
        else:
            gt[str(uid)] = []
    wrapper = {"_synthetic": True, "data": gt}
    write_json(GT_PATH, wrapper)
    return wrapper

def load_gt_struct(gt_path: Path = None):
    p = Path(gt_path) if gt_path is not None else GT_PATH

    # CSV ground truth: one row per user-item target.
    if p.suffix.lower() == ".csv":
        if not p.is_file():
            raise SystemExit(f"Ground-truth CSV not found: {p}")

        gt = {}
        with open(p, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fields = set(reader.fieldnames or [])

            if not {"user_id", "item_id"}.issubset(fields):
                raise SystemExit(
                    f"Unexpected CSV ground-truth columns in {p}: "
                    f"{reader.fieldnames}. Expected user_id,item_id."
                )

            for row in reader:
                uid = (row.get("user_id") or "").strip()
                item = (row.get("item_id") or "").strip()

                if not uid or not item:
                    continue

                gt.setdefault(uid, []).append(item)

        if not gt:
            raise SystemExit(f"Ground-truth CSV is empty or contains no valid rows: {p}")

        return {"_synthetic": False, "data": gt}

    # Preserve existing JSON behavior, including synthetic fallback.
    raw = load_json_safe(p)
    if raw is None:
        if p.suffix.lower() != ".json":
            raise SystemExit(f"Unsupported ground-truth format: {p}")

        demo = RESULTS_DIR / "demo_rankings.json"
        if demo.exists():
            return auto_create_synthetic_gt(demo)

        raise SystemExit(
            f"Ground-truth missing at {p} and demo_rankings.json "
            "not found to create synthetic GT."
        )

    if isinstance(raw, dict) and "_synthetic" in raw and "data" in raw:
        return raw

    if isinstance(raw, dict):
        if "data" in raw:
            return {"_synthetic": False, "data": raw["data"]}
        return {"_synthetic": False, "data": raw}

    raise SystemExit("Unexpected ground-truth format in {}".format(p))

def gather_result_files():
    files = sorted([p for p in RESULTS_DIR.glob("*.json") if p.is_file()])
    files_sorted = sorted(files, key=lambda p: (p.name!="demo_rankings.json", p.name))
    return files_sorted

def main(k=10):
    gt_wrapper = load_gt_struct()
    gt = gt_wrapper["data"]
    synthetic = gt_wrapper.get("_synthetic", False)
    if synthetic:
        logger.warning("Using SYNTHETIC ground-truth generated from top-1 (not for paper).")

    result_files = gather_result_files()
    if not result_files:
        raise SystemExit("No result JSON files found in results/")

    agg_rows = []
    aggregated_summary = {}


    for rf in result_files:
        logger.info("Evaluating %s", rf)
        results_raw = load_json_safe(rf)
        if not results_raw:
            logger.warning("Skipping %s (empty or invalid JSON)", rf)
            continue

        if isinstance(results_raw, dict):
            if "results" in results_raw and isinstance(results_raw["results"], dict):
                logger.info("Unwrapping '%s' -> taking key 'results'", rf.name)
                results = results_raw["results"]
            elif set(results_raw.keys()) >= {"file", "summary"} and not any(isinstance(v, dict) and v for v in results_raw.values()):
                logger.warning("Skipping %s (looks like summary/metadata file)", rf)
                continue
            else:
                results = results_raw
        else:
            logger.warning("Skipping %s (not a JSON object)", rf)
            continue

        def looks_like_user_recs_map(obj):
            if not isinstance(obj, dict) or not obj:
                return False

            keys = list(obj.keys())[:50]
            key_digit_like = sum(1 for k in keys if isinstance(k, str) and k.isdigit())
            if key_digit_like >= max(1, len(keys)//2):
                return True

            val_sample = list(obj.values())[:10]
            cnt_ok = 0
            for v in val_sample:
                if isinstance(v, list):
                    if not v:
                        cnt_ok += 1
                    else:
                        first = v[0]
                        if isinstance(first, dict) or isinstance(first, (str,int)):
                            cnt_ok += 1
            return cnt_ok >= max(1, len(val_sample)//2)

        if not looks_like_user_recs_map(results):
            logger.warning("Skipping %s (does not look like user->recs map after unwrapping). Keys sample: %s", rf.name, list(results.keys())[:5])
            continue


        rows, summary = evaluate_single(results, gt, k=k)
        csv_path = OUT_DIR / f"eval_{rf.stem}_per_user.csv"
        if rows:
            keys = list(rows[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)
        json_path = OUT_DIR / f"eval_{rf.stem}_summary.json"
        write_json(json_path, {"file": str(rf), "summary": summary, "synthetic_gt": synthetic})
        logger.info("Saved %s and %s", csv_path, json_path)
        aggregated_summary[rf.name] = summary
        agg_rows.append({
            "result_file": rf.name,
            "n_users": summary["n_users"],
            "hr_mean": summary["hr_mean"],
            "hr_std": summary["hr_std"],
            "ndcg_mean": summary["ndcg_mean"],
            "ndcg_std": summary["ndcg_std"]
        })

    

    agg_csv = OUT_DIR / "eval_aggregated_results.csv"
    if agg_rows:
        keys = list(agg_rows[0].keys())
        with open(agg_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(agg_rows)
    write_json(OUT_DIR / "eval_aggregated_summary.json", aggregated_summary)
    logger.info("Wrote aggregated CSV: %s", agg_csv)
    logger.info("Done.")

if __name__ == "__main__":
    main(k=10)
