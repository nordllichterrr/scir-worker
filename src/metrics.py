
import numpy as np
from collections import Counter
from typing import List, Dict
import math

def hr_at_k(recommended: List[str], ground_truth: List[str], k=10):
    rec = recommended[:k]
    return 1.0 if any(r in ground_truth for r in rec) else 0.0

def ndcg_at_k(recommended: List[str], ground_truth: List[str], k=10):
    """
    Binary nDCG@k:
      - rel = 1 if item in ground_truth else 0
      - DCG = sum_{i=1..k} rel_i / log2(i+1)
      - IDCG = DCG of ideal ranking with all relevant items first
    """
    if not recommended or not ground_truth or k <= 0:
        return 0.0
    gt_set = set(str(x) for x in ground_truth)
    rec = [str(x) for x in recommended[:k]]

    dcg = 0.0
    for i, r in enumerate(rec, start=1):
        if r in gt_set:
            dcg += 1.0 / math.log2(i + 1)

    rel = min(len(gt_set), k)
    if rel <= 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, rel + 1))
    return dcg / idcg if idcg > 0 else 0.0

def mrr_at_k(recommended: List[str], ground_truth: List[str], k=10):

    rec = recommended[:k]
    for i, r in enumerate(rec, start=1):
        if r in ground_truth:
            return 1.0 / i
    return 0.0

def map_at_k(recommended: List[str], ground_truth: List[str], k=10):
 
    if not ground_truth:
        return 0.0
    
    gt_set = set(ground_truth)
    rec = recommended[:k]
    
    if not rec:
        return 0.0
    

    num_relevant = sum(1 for r in rec if r in gt_set)
    
    if num_relevant == 0:
        return 0.0

    precision_sum = 0.0
    relevant_found = 0
    
    for i, r in enumerate(rec, start=1):
        if r in gt_set:
            relevant_found += 1
            precision_at_i = relevant_found / i
            precision_sum += precision_at_i
    
    ap = precision_sum / len(gt_set)
    return ap

def coverage(all_recommended: List[List[str]], catalog_size: int):
    unique = set([it for lst in all_recommended for it in lst])
    return len(unique) / float(catalog_size)

def catalog_coverage_list(all_recommended: List[List[str]]):
    unique = set([it for lst in all_recommended for it in lst])
    return unique

def catalog_coverage_at_k(all_recommended: List[List[str]], k: int, catalog_size: int) -> float:

    unique = set()
    for lst in all_recommended:
        for iid in lst[:k]:
            unique.add(str(iid))
    return len(unique) / float(catalog_size) if catalog_size > 0 else 0.0

def user_coverage(all_recommended: List[List[str]], min_recs: int = 1) -> float:

    if not all_recommended:
        return 0.0
    n_with = sum(1 for lst in all_recommended if len(lst) >= min_recs)
    return n_with / len(all_recommended)

def mean_popularity_rank(novelty_inputs: List[List[str]], item_pop_rank: Dict[str,int]):

    vals = []
    for lst in novelty_inputs:
        vals.extend([item_pop_rank.get(i, len(item_pop_rank)+1) for i in lst])
    return float(np.mean(vals)) if vals else None

def mean_self_information_novelty(
    all_recommended: List[List[str]],
    item_pop_count: Dict[str, float],
    total_count: float,
) -> float:

    if total_count <= 0 or not item_pop_count:
        return 0.0
    vals = []
    for lst in all_recommended:
        for iid in lst:
            p = float(item_pop_count.get(str(iid), 0) or 0) / total_count
            p = max(p, 1.0 / total_count)  
            vals.append(-math.log2(p))
    return float(np.mean(vals)) if vals else 0.0

def intra_list_diversity(all_recommended: List[List[str]], embeddings: Dict[str, np.ndarray]):
    import numpy as np
    out = []
    for lst in all_recommended:
        vecs = [embeddings.get(i) for i in lst if embeddings.get(i) is not None]
        if len(vecs) < 2:
            out.append(0.0)
            continue
        mat = np.stack(vecs)

        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        matn = mat / (norms + 1e-9)
        sims = matn @ matn.T
        n = len(vecs)
        dists = []
        for i in range(n):
            for j in range(i+1, n):
                dists.append(1.0 - float(sims[i,j]))
        out.append(float(np.mean(dists)) if dists else 0.0)
    return float(np.mean(out))

def compute_item_popularity(interactions: List[tuple]):
    c = Counter([it for _, it in interactions])
    sorted_items = [it for it, _ in c.most_common()]
    rank = {it: i+1 for i, it in enumerate(sorted_items)}
    return rank


def exposure_gini(item_counts: List[float]) -> float:

    if not item_counts or sum(item_counts) == 0:
        return 0.0
    x = np.array(sorted(item_counts), dtype=float)
    n = len(x)
    cum = np.cumsum(x)
    return float((2 * np.sum((np.arange(1, n + 1) * x)) / (n * cum[-1])) - (n + 1) / n)

def exposure_entropy(item_counts: List[float]) -> float:
    if not item_counts:
        return 0.0
    total = sum(item_counts)
    if total <= 0:
        return 0.0
    probs = [c / total for c in item_counts if c > 0]
    return float(-sum(p * math.log2(p) for p in probs))

def top_p_share(item_counts: List[float], p: int = 10) -> float:
    if not item_counts or p <= 0:
        return 0.0
    sorted_counts = sorted(item_counts, reverse=True)
    total = sum(sorted_counts)
    if total <= 0:
        return 0.0
    top_sum = sum(sorted_counts[:p])
    return float(top_sum / total)

def long_tail_coverage(
    all_recommended: List[List[str]],
    item_pop_rank: Dict[str, int],
    tail_threshold_percentile: int = 80
) -> float:

    if not all_recommended or not item_pop_rank:
        return 0.0
    

    all_ranks = sorted(item_pop_rank.values())
    if not all_ranks:
        return 0.0
    
    threshold_idx = int(len(all_ranks) * tail_threshold_percentile / 100.0)
    threshold_rank = all_ranks[min(threshold_idx, len(all_ranks) - 1)]

    tail_count = 0
    total_count = 0
    
    for rec_list in all_recommended:
        for item_id in rec_list:
            total_count += 1
            rank = item_pop_rank.get(str(item_id))
            if rank is not None and rank >= threshold_rank:
                tail_count += 1
    
    return float(tail_count / total_count) if total_count > 0 else 0.0

def avg_log_popularity(
    all_recommended: List[List[str]],
    item_pop_count: Dict[str, float]
) -> float:

    if not all_recommended or not item_pop_count:
        return None
    
    log_pops = []
    for rec_list in all_recommended:
        for item_id in rec_list:
            pop = item_pop_count.get(str(item_id), 0.0)
            log_pops.append(math.log2(1.0 + float(pop)))
    
    return float(np.mean(log_pops)) if log_pops else None
