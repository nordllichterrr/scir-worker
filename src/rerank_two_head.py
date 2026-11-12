
from __future__ import annotations

from typing import Dict, List, Any, Optional

import numpy as np


def novelty_from_pop_rank(
    item_pop_rank: Dict[str, int],
    catalog_size: Optional[int] = None,
) -> Dict[str, float]:

    if not item_pop_rank:
        return {}
    max_rank = max(item_pop_rank.values()) if item_pop_rank else 1
    if catalog_size is not None and catalog_size > max_rank:
        max_rank = catalog_size
    out = {}
    for iid, rank in item_pop_rank.items():
    
        nov = (max_rank - rank + 1) / max_rank if max_rank else 0.0
        out[str(iid)] = max(0.0, min(1.0, nov))
    return out


def combine_relevance_novelty(
    scored_list: List[Dict[str, Any]],
    item_novelty: Dict[str, float],
    alpha: float,
) -> List[Dict[str, Any]]:

    if not scored_list or alpha < 0 or alpha > 1:
        return scored_list
    ids = [x["item_id"] for x in scored_list]
    rel_scores = np.array([float(x["score"]) for x in scored_list])
    r_min, r_max = rel_scores.min(), rel_scores.max()
    if r_max > r_min:
        rel_norm = (rel_scores - r_min) / (r_max - r_min)
    else:
        rel_norm = np.ones_like(rel_scores)
    nov_scores = np.array([item_novelty.get(str(iid), 0.0) for iid in ids])

    nov_scores = np.where(nov_scores > 0, nov_scores, 0.5)
    combined = alpha * rel_norm + (1.0 - alpha) * nov_scores
    out = []
    for i, x in enumerate(scored_list):
        rec = dict(x)
        rec["score"] = float(combined[i])
        rec["relevance"] = float(rel_scores[i])
        rec["novelty"] = float(nov_scores[i])
        out.append(rec)
    out.sort(key=lambda r: r["score"], reverse=True)
    return out


def select_pareto_balanced(
    scored_list: List[Dict[str, Any]],
    topk: int,
    relevance_key: str = "relevance",
    novelty_key: str = "novelty",
) -> List[Dict[str, Any]]:
 
    if not scored_list or topk <= 0:
        return scored_list[:topk]
    rels = np.array([float(x.get(relevance_key, x.get("score", 0))) for x in scored_list])
    novs = np.array([float(x.get(novelty_key, 0)) for x in scored_list])
    r_min, r_max = rels.min(), rels.max()
    n_min, n_max = novs.min(), novs.max()
    if r_max > r_min:
        rels = (rels - r_min) / (r_max - r_min)
    if n_max > n_min:
        novs = (novs - n_min) / (n_max - n_min)
    balanced = np.minimum(rels, novs)
    tie = rels + novs
    order = np.lexsort((-tie, -balanced))
    return [scored_list[i] for i in order[:topk]]
