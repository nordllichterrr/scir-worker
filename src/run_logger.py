import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import Counter
import numpy as np
from .utils import logger

RUNS_LOG_PATH = Path("experiments") / "runs.jsonl"


def compute_diagnostics(
    results: Dict[str, List[Dict]], 
    gt: Dict[str, List[str]],
    candidate_pools: Optional[Dict[str, List[str]]] = None,
    rerank_times: Optional[Dict[str, float]] = None,
    retrieval_times: Optional[Dict[str, float]] = None,
    run_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    top1_items = []
    topk_lengths = []
    for uid, recs in results.items():
        if recs and len(recs) > 0:
            top1_items.append(str(recs[0].get("item_id")))
        topk_lengths.append(len(recs) if recs else 0)
    
    unique_top1 = len(set(top1_items))
    top1_counter = Counter(top1_items)
    most_common_top1 = dict(top1_counter.most_common(1)) if top1_counter else {}
    top1_counts = [[item_id, count] for item_id, count in top1_counter.most_common(10)]
    
    length_counter = Counter(topk_lengths)
    topk_length_distribution = dict(length_counter)
    
    recall_at_50 = None
    recall_at_200 = None
    recall_at_1000 = None
    
    if candidate_pools is not None:
        total_relevant_items = 0
        total_found_50 = 0
        total_found_200 = 0
        total_found_1000 = 0
        
        for uid in results.keys():
            gt_items = gt.get(uid, [])
            if not gt_items:
                continue
            candidates = candidate_pools.get(uid, [])
            if not candidates:
                continue
            
          
            gt_ids = set(str(x) for x in gt_items)
            total_relevant = len(gt_ids)
            
            if total_relevant == 0:
                continue
            
           
            cand_list_50 = [str(x) for x in candidates[:50]]
            cand_list_200 = [str(x) for x in candidates[:200]]
            cand_list_1000 = [str(x) for x in candidates[:1000]]
            
            relevant_in_50 = len(gt_ids & set(cand_list_50))
            relevant_in_200 = len(gt_ids & set(cand_list_200))
            relevant_in_1000 = len(gt_ids & set(cand_list_1000))
            
            
            total_relevant_items += total_relevant
            total_found_50 += relevant_in_50
            total_found_200 += relevant_in_200
            total_found_1000 += relevant_in_1000
        
        if total_relevant_items > 0:
            recall_at_50 = total_found_50 / total_relevant_items
            recall_at_200 = total_found_200 / total_relevant_items
            recall_at_1000 = total_found_1000 / total_relevant_items
        else:
            recall_at_50 = None
            recall_at_200 = None
            recall_at_1000 = None
    
    unique_recommended_items_topk_total = len(set(str(rec.get("item_id")) for recs in results.values() for rec in (recs or [])))
    all_rec_lists = [[str(rec.get("item_id")) for rec in (recs or []) if rec.get("item_id")] for recs in results.values()]
    from .metrics import (
        coverage,
        catalog_coverage_at_k,
        user_coverage,
        mean_popularity_rank,
        mean_self_information_novelty,
        exposure_gini,
        exposure_entropy,
        top_p_share,
        long_tail_coverage,
        avg_log_popularity,
    )
    topk_k = 10

    n_users = len(results)
    n_empty = sum(1 for L in topk_lengths if L == 0)
    share_empty_topk = (n_empty / n_users) if n_users > 0 else None

    diagnostics = {
        "recall@50": recall_at_50,
        "recall@200": recall_at_200,
        "recall@1000": recall_at_1000,
        "unique_top1": unique_top1,
        "most_common_top1": most_common_top1,
        "top1_counts": top1_counts,
        "topk_length_distribution": topk_length_distribution,
        "unique_recommended_items_topk_total": unique_recommended_items_topk_total,
        "n_users": n_users,
        "share_users_empty_topk": share_empty_topk,
        "user_coverage": user_coverage(all_rec_lists, min_recs=1),
    }
    
    exposure_counter = Counter()
    for recs in results.values():
        for r in (recs or [])[:topk_k]:
            iid = r.get("item_id")
            if iid is not None:
                exposure_counter[str(iid)] += 1
    if exposure_counter:
        counts = list(exposure_counter.values())
        diagnostics["exposure_gini"] = exposure_gini(counts)
        diagnostics["exposure_entropy"] = exposure_entropy(counts)
        diagnostics["top10_share"] = top_p_share(counts, p=10)
    if run_meta:
        diagnostics["catalog_size"] = run_meta.get("catalog_size")
        catalog_size = run_meta.get("catalog_size")
        item_pop_rank = run_meta.get("item_pop_rank")
        if catalog_size and catalog_size > 0:
            diagnostics["coverage"] = coverage(all_rec_lists, catalog_size)
            diagnostics["catalog_coverage_at_10"] = catalog_coverage_at_k(all_rec_lists, k=10, catalog_size=catalog_size)
            diagnostics["catalog_coverage_at_50"] = catalog_coverage_at_k(all_rec_lists, k=50, catalog_size=catalog_size)
            diagnostics["catalog_coverage_at_100"] = catalog_coverage_at_k(all_rec_lists, k=100, catalog_size=catalog_size)
        if item_pop_rank:
            diagnostics["mean_popularity_rank"] = mean_popularity_rank(all_rec_lists, item_pop_rank)
            diagnostics["long_tail_coverage"] = long_tail_coverage(all_rec_lists, item_pop_rank, tail_threshold_percentile=80)

        item_pop_count = run_meta.get("item_pop_count")
        if item_pop_count:
            avg_log_pop = avg_log_popularity(all_rec_lists, item_pop_count)
            if avg_log_pop is not None:
                diagnostics["avg_log_popularity"] = avg_log_pop
            try:
                from .rerank_two_head import novelty_from_pop_rank
                item_novelty = novelty_from_pop_rank(item_pop_rank, catalog_size=catalog_size)
                ser_scores = []
                for uid, recs in results.items():
                    top_ids = [str(rec.get("item_id")) for rec in (recs or []) if rec.get("item_id")] [:topk_k]
                    if not top_ids:
                        continue
                    gt_ids = set(str(x) for x in gt.get(uid, []))
                    if not gt_ids:
                        continue
                    vals = [item_novelty.get(iid, 0.0) for iid in top_ids if iid in gt_ids]
                    if vals:
                        ser_scores.append(float(np.mean(vals)))
                diagnostics["serendipity@10"] = float(np.mean(ser_scores)) if ser_scores else None
            except Exception:
                pass

        item_pop_count = run_meta.get("item_pop_count")
        novelty_val = None
        if item_pop_count is not None and isinstance(item_pop_count, dict):
            total_count = float(sum(item_pop_count.values()))
            if total_count > 0:
                novelty_val = mean_self_information_novelty(all_rec_lists, item_pop_count, total_count)

        if novelty_val is None or novelty_val == 0.0:
            if exposure_counter:
                exp_total = float(sum(exposure_counter.values()))
                if exp_total > 0:
                    novelty_val = mean_self_information_novelty(all_rec_lists, dict(exposure_counter), exp_total)
        if novelty_val is not None:
            diagnostics["mean_self_information_novelty"] = novelty_val
        diagnostics["candidates_requested"] = run_meta.get("candidates_requested")
        diagnostics["candidates_after_filters"] = run_meta.get("candidates_after_filters")
        diagnostics["final_topk"] = run_meta.get("final_topk")
        diagnostics["retrieval_method"] = run_meta.get("retrieval_method")
        if run_meta.get("ann_k") is not None:
            diagnostics["ann_k"] = run_meta.get("ann_k")
        if run_meta.get("bm25_k") is not None:
            diagnostics["bm25_k"] = run_meta.get("bm25_k")
        if run_meta.get("popularity_k") is not None:
            diagnostics["popularity_k"] = run_meta.get("popularity_k")
        if run_meta.get("overlap") is not None:
            diagnostics["overlap"] = run_meta.get("overlap")
        if run_meta.get("union_size") is not None:
            diagnostics["union_size"] = run_meta.get("union_size")

        if run_meta.get("leakage_sanity") is not None:
            diagnostics["leakage_sanity"] = run_meta.get("leakage_sanity")
    else:
        # If run_meta doesn't provide pool size, derive it from candidate_pools dict.
        if candidate_pools and isinstance(candidate_pools, dict):
            pool_sizes = [len(v) for v in candidate_pools.values() if v is not None]
            if pool_sizes:
                diagnostics["candidate_pool_size_mean"] = float(np.mean(pool_sizes))
                diagnostics["candidate_pool_size_p50"] = int(np.median(pool_sizes))
                diagnostics["candidate_pool_size_min"] = int(np.min(pool_sizes))
                diagnostics["candidate_pool_size_max"] = int(np.max(pool_sizes))
                # keep old field name, but make it meaningful:
                diagnostics["candidate_pool_size"] = int(np.median(pool_sizes))
            else:
                diagnostics["candidate_pool_size"] = None
        else:
            diagnostics["candidate_pool_size"] = None
    

    sanity_sample = []
    for i, uid in enumerate(list(results.keys())[:5]):
        gt_items = set(str(x) for x in gt.get(uid, []))
        pool_set = set(candidate_pools.get(uid, [])) if candidate_pools else set()
        top10_ids = set(str(rec.get("item_id")) for rec in (results.get(uid) or []) if rec.get("item_id"))
        sanity_sample.append({
            "user_id": uid,
            "gt_size": len(gt_items),
            "gt_intersection_pool": len(gt_items & pool_set),
            "gt_intersection_top10": len(gt_items & top10_ids),
        })
    diagnostics["sanity_sample"] = sanity_sample
    
    if rerank_times:
        diagnostics["rerank_time_mean"] = rerank_times.get("mean")
        diagnostics["rerank_time_std"] = rerank_times.get("std")
        if rerank_times.get("p50") is not None:
            diagnostics["rerank_time_p50"] = rerank_times.get("p50")
        if rerank_times.get("p95") is not None:
            diagnostics["rerank_time_p95"] = rerank_times.get("p95")
    if retrieval_times:
        diagnostics["retrieval_time_mean"] = retrieval_times.get("mean")
        diagnostics["retrieval_time_std"] = retrieval_times.get("std")
        if retrieval_times.get("p50") is not None:
            diagnostics["retrieval_time_p50"] = retrieval_times.get("p50")
        if retrieval_times.get("p95") is not None:
            diagnostics["retrieval_time_p95"] = retrieval_times.get("p95")

    total_time = (run_meta or {}).get("total_time_per_user")
    if total_time and total_time.get("mean") and total_time["mean"] > 0:
        diagnostics["total_time_per_user_mean"] = total_time.get("mean")
        diagnostics["total_time_per_user_p50"] = total_time.get("p50")
        diagnostics["total_time_per_user_p95"] = total_time.get("p95")
        diagnostics["throughput_users_per_sec"] = 1.0 / total_time["mean"]
    return diagnostics


def log_run(
    run_id: str,
    config: Dict[str, Any],
    metrics: Dict[str, Dict[str, float]],
    metrics_pre: Optional[Dict[str, Dict[str, float]]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
    diagnostics_pre: Optional[Dict[str, Any]] = None,
    files: Optional[Dict[str, str]] = None,
    results: Optional[Dict[str, List[Dict]]] = None,
    gt: Optional[Dict[str, List[str]]] = None,
    candidate_pools: Optional[Dict[str, List[str]]] = None,
    rerank_times: Optional[Dict[str, float]] = None,
    retrieval_times: Optional[Dict[str, float]] = None,
    run_meta: Optional[Dict[str, Any]] = None,
    runs_log_path: Optional[Path] = None,
) -> None:
    out_path = Path(runs_log_path) if runs_log_path is not None else RUNS_LOG_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    if diagnostics is None and results is not None and gt is not None:
        diagnostics = compute_diagnostics(
            results, gt, candidate_pools, rerank_times, retrieval_times, run_meta
        )
    
    record = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "config": config,
        "metrics": metrics,
        "metrics_pre": metrics_pre or {},
        "diagnostics": diagnostics or {},
        "diagnostics_pre": diagnostics_pre or {},
        "files": files or {}
    }
    
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    metric_bits = []
    for m_name, m_val in (metrics or {}).items():
        if not isinstance(m_val, dict):
            continue
        mean = m_val.get("mean")
        lo = m_val.get("ci95_low")
        hi = m_val.get("ci95_high")
        n_users = m_val.get("n_users")
        if mean is None:
            continue
        if lo is not None and hi is not None:
            metric_bits.append(f"{m_name}={mean:.4f} [{lo:.4f}, {hi:.4f}] (n={n_users})")
        else:
            metric_bits.append(f"{m_name}={mean:.4f} (n={n_users})")
    if metric_bits:
        logger.info("Metrics with CI: %s", " | ".join(metric_bits))

    logger.info("Logged run %s to %s", run_id, out_path)


def load_runs(path: Optional[Path] = None) -> list:
    p = Path(path) if path is not None else RUNS_LOG_PATH
    if not p.exists():
        return []
    
    runs = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    runs.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning("Failed to parse line in runs.jsonl: %s", e)
                    continue
    
    return runs


def get_runs_by_config(config_filter: Dict[str, Any], path: Optional[Path] = None) -> list:
    all_runs = load_runs(path=path)
    matching = []
    
    for run in all_runs:
        match = True
        config = run.get("config", {})
        
        for key_path, value in config_filter.items():
            keys = key_path.split(".")
            current = config
            for key in keys:
                if not isinstance(current, dict) or key not in current:
                    match = False
                    break
                current = current[key]
            
            if match and current != value:
                match = False
            
            if not match:
                break
        
        if match:
            matching.append(run)
    
    return matching
