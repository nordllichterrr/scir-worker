
from __future__ import annotations

import math
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

from .utils import logger


def _scored_to_list(scored: List[Dict]) -> List[Tuple[str, float]]:
 
    out = []
    for x in scored:
        iid = x.get("item_id")
        s = x.get("score")
        if iid is not None and s is not None:
            out.append((str(iid), float(s)))
    return out


def _list_to_scored(ordered: List[Tuple[str, float]], reason: str = "diversify") -> List[Dict]:

    return [{"item_id": iid, "score": score, "reason": reason} for iid, score in ordered]


def apply_popularity_penalty(
    scored: List[Tuple[str, float]],
    item_popularity: Dict[str, float],
    alpha: float,
    pop_key: str = "pop",
) -> List[Tuple[str, float]]:
    if alpha <= 0 or not item_popularity:
        return list(scored)
    out = []
    for iid, s in scored:
        pop = float(item_popularity.get(iid, 0) or 0)
        penalty = alpha * math.log1p(max(0, pop))
        out.append((iid, s - penalty))
    return out


def apply_exposure_penalty(
    scored: List[Tuple[str, float]],
    exposure_map: Dict[str, float],
    beta: float,
) -> List[Tuple[str, float]]:
    if beta <= 0 or not exposure_map:
        return list(scored)
    out = []
    for iid, s in scored:
        exp = float(exposure_map.get(iid, 0) or 0)
        out.append((iid, s - beta * exp))
    return out


def apply_mmr(
    scored: List[Tuple[str, float]],
    topk: int,
    item_embeddings: np.ndarray,
    id2idx: Dict[str, int],
    lambda_mmr: float,
) -> List[Tuple[str, float]]:

    if lambda_mmr <= 0 or topk <= 0 or not scored:
        return scored[:topk]
    ids = [x[0] for x in scored]
    rel = {x[0]: x[1] for x in scored}

    vecs = []
    valid_ids = []
    for iid in ids:
        idx = id2idx.get(iid)
        if idx is None:
            continue
        vecs.append(item_embeddings[int(idx)])
        valid_ids.append(iid)
    if len(valid_ids) < 2:
        return scored[:topk]
    mat = np.stack(vecs, axis=0).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms
    id_to_pos = {iid: i for i, iid in enumerate(valid_ids)}
    selected: List[str] = []
    selected_pos: List[int] = []
    remaining = set(valid_ids)

    for _ in range(min(topk, len(remaining))):
        best_id = None
        best_mmr = -np.inf
        for iid in remaining:
            pos = id_to_pos[iid]
            rel_score = rel.get(iid, 0.0)
            sim_to_sel = 0.0
            if selected_pos:
                sims = mat[pos] @ mat[selected_pos].T
                sim_to_sel = float(np.max(sims))
            mmr = rel_score - lambda_mmr * sim_to_sel
            if mmr > best_mmr:
                best_mmr = mmr
                best_id = iid
        if best_id is None:
            break
        selected.append(best_id)
        selected_pos.append(id_to_pos[best_id])
        remaining.discard(best_id)

    if len(selected) < topk:
        rest = [(iid, rel[iid]) for iid in ids if iid not in selected]
        rest.sort(key=lambda x: x[1], reverse=True)
        for iid, s in rest:
            if len(selected) >= topk:
                break
            selected.append(iid)

    return [(iid, rel[iid]) for iid in selected[:topk]]


def _get_categories(item_meta: Dict, category_key: str) -> List[str]:
    raw = item_meta.get(category_key) or item_meta.get("genres") or item_meta.get("format_tags") or ""
    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw if x]
    s = str(raw).strip()
    if not s:
        return ["__unknown__"]
    return [x.strip() for x in s.replace("|", ",").split(",") if x.strip()]


def apply_xquad(
    scored: List[Tuple[str, float]],
    topk: int,
    items_meta: Dict[str, Dict[str, Any]],
    category_key: str = "genres",
    lambda_diversify: float = 0.5,
) -> List[Tuple[str, float]]:

    if lambda_diversify <= 0 or topk <= 0 or not scored:
        return scored[:topk]
    rel = {x[0]: x[1] for x in scored}
    id_to_cats = {}
    all_cats = set()
    for iid, _ in scored:
        meta = items_meta.get(iid) or items_meta.get(str(iid)) or {}
        cats = _get_categories(meta, category_key)
        id_to_cats[iid] = cats
        all_cats.update(cats)
    if not all_cats:
        return scored[:topk]

    p_intent = 1.0 / len(all_cats)
    selected: List[str] = []
    remaining = {x[0] for x in scored}
    cat_counts: Dict[str, int] = {c: 0 for c in all_cats}

    for _ in range(min(topk, len(scored))):
        best_id = None
        best_score = -np.inf
        for iid in remaining:
            r = rel.get(iid, 0.0)
            rel_score = rel.get(iid, 0.0)
            n_sel = len(selected)
            novelty = 0.0
            for c in id_to_cats.get(iid, ["__unknown__"]):
                p_sel_given_intent = cat_counts[c] / n_sel if n_sel else 0.0
                novelty += p_intent * (1.0 - p_sel_given_intent)
            combo = (1.0 - lambda_diversify) * rel_score + lambda_diversify * novelty
            if combo > best_score:
                best_score = combo
                best_id = iid
        if best_id is None:
            break
        selected.append(best_id)
        remaining.discard(best_id)
        for c in id_to_cats.get(best_id, ["__unknown__"]):
            cat_counts[c] = cat_counts.get(c, 0) + 1

    return [(iid, rel[iid]) for iid in selected]


def _item_pop_rank_from_meta(items_meta: Dict[str, Dict], pop_key: str = "pop") -> Dict[str, int]:

    pairs = []
    for iid, meta in items_meta.items():
        pop = float(meta.get(pop_key, 0) or 0)
        pairs.append((iid, pop))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return {iid: r for r, (iid, _) in enumerate(pairs, start=1)}


def apply_fairness_head_mid_tail(
    scored: List[Tuple[str, float]],
    topk: int,
    item_pop_rank: Dict[str, int],
    head_n: int,
    mid_n: int,
    tail_n: int,
    total_head: int,
    total_mid: int,
    total_tail: int,
) -> List[Tuple[str, float]]:
  
    if topk <= 0 or not scored:
        return scored[:topk]
    rel = {x[0]: x[1] for x in scored}
    head_ids = [iid for iid in rel if 1 <= item_pop_rank.get(iid, 999999) <= total_head]
    mid_ids = [iid for iid in rel if total_head < item_pop_rank.get(iid, 999999) <= total_head + total_mid]
    tail_ids = [iid for iid in rel if item_pop_rank.get(iid, 999999) > total_head + total_mid]
    head_ids.sort(key=lambda i: rel[i], reverse=True)
    mid_ids.sort(key=lambda i: rel[i], reverse=True)
    tail_ids.sort(key=lambda i: rel[i], reverse=True)
    result: List[Tuple[str, float]] = []
    h, m, t = 0, 0, 0
    for _ in range(topk):
        if h < head_n and h < len(head_ids):
            result.append((head_ids[h], rel[head_ids[h]]))
            h += 1
        elif m < mid_n and m < len(mid_ids):
            result.append((mid_ids[m], rel[mid_ids[m]]))
            m += 1
        elif t < tail_n and t < len(tail_ids):
            result.append((tail_ids[t], rel[tail_ids[t]]))
            t += 1
        else:
    
            if h < len(head_ids):
                result.append((head_ids[h], rel[head_ids[h]]))
                h += 1
            elif m < len(mid_ids):
                result.append((mid_ids[m], rel[mid_ids[m]]))
                m += 1
            elif t < len(tail_ids):
                result.append((tail_ids[t], rel[tail_ids[t]]))
                t += 1
            else:
                break
    return result


def diversify(
    scored_list: List[Dict[str, Any]],
    topk: int,
    *,
    popularity_penalty_alpha: float = 0.0,
    exposure_beta: float = 0.0,
    item_popularity: Optional[Dict[str, float]] = None,
    exposure_map: Optional[Dict[str, float]] = None,
    mmr_lambda: float = 0.0,
    item_embeddings: Optional[np.ndarray] = None,
    id2idx: Optional[Dict[str, int]] = None,
    xquad_lambda: float = 0.0,
    items_meta: Optional[Dict[str, Dict[str, Any]]] = None,
    category_key: str = "genres",
    fairness: Optional[Dict[str, int]] = None,
    item_pop_rank: Optional[Dict[str, int]] = None,
    log_stats: bool = True,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    
    from .utils import logger
    
    stats = {
        "penalty_applied": False,
        "penalty_type": None,
        "penalty_coefficient": None,
        "max_penalty_value": None,
        "mean_penalty_value": None,
        "reordered": False,
        "jaccard_at_10": None,
        "percent_reordered": None,
    }
    
    if not scored_list:
        return [], stats
    

    original_top10 = [x["item_id"] for x in scored_list[:10]]
    lst = _scored_to_list(scored_list)
    original_scores = {iid: score for iid, score in lst}


    penalties_applied = []
    if popularity_penalty_alpha > 0 and item_popularity:
        penalty_values = []
        for iid, score in lst:
            pop = float(item_popularity.get(iid, 0) or 0)
            penalty = popularity_penalty_alpha * math.log1p(max(0, pop))
            penalty_values.append(penalty)
        if penalty_values:
            stats["penalty_applied"] = True
            stats["penalty_type"] = "popularity"
            stats["penalty_coefficient"] = popularity_penalty_alpha
            stats["max_penalty_value"] = float(max(penalty_values))
            stats["mean_penalty_value"] = float(np.mean(penalty_values))
            if log_stats:
                logger.debug("Popularity penalty: alpha=%.3f, max=%.3f, mean=%.3f", 
                           popularity_penalty_alpha, stats["max_penalty_value"], stats["mean_penalty_value"])
        lst = apply_popularity_penalty(lst, item_popularity, popularity_penalty_alpha)
        penalties_applied.append("popularity")
    
    if exposure_beta > 0 and exposure_map:
        penalty_values = []
        for iid, score in lst:
            exp = float(exposure_map.get(iid, 0) or 0)
            penalty = exposure_beta * exp
            penalty_values.append(penalty)
        if penalty_values:
            stats["penalty_applied"] = True
            stats["penalty_type"] = "exposure" if not stats["penalty_type"] else "both"
            if stats["penalty_coefficient"] is None:
                stats["penalty_coefficient"] = exposure_beta
            else:
                stats["penalty_coefficient"] = f"{stats['penalty_coefficient']}+{exposure_beta}"
            stats["max_penalty_value"] = float(max(penalty_values))
            stats["mean_penalty_value"] = float(np.mean(penalty_values))
            if log_stats:
                logger.debug("Exposure penalty: beta=%.3f, max=%.3f, mean=%.3f", 
                           exposure_beta, stats["max_penalty_value"], stats["mean_penalty_value"])
        lst = apply_exposure_penalty(lst, exposure_map, exposure_beta)
        penalties_applied.append("exposure")

  
    lst_before_sort = lst[:topk]  
    lst.sort(key=lambda x: x[1], reverse=True)


    if mmr_lambda > 0 and item_embeddings is not None and id2idx:
        lst_before_mmr = lst[:topk]
        lst = apply_mmr(lst, topk, item_embeddings, id2idx, mmr_lambda)
        stats["reordered"] = True
    elif xquad_lambda > 0 and items_meta:
        lst_before_xquad = lst[:topk]
        lst = apply_xquad(lst, topk, items_meta, category_key=category_key, lambda_diversify=xquad_lambda)
        stats["reordered"] = True
    else:
        lst = lst[:topk]
    
  
    final_top10 = [iid for iid, _ in lst[:10]]
    if original_top10:
        intersection = len(set(original_top10) & set(final_top10))
        union = len(set(original_top10) | set(final_top10))
        stats["jaccard_at_10"] = intersection / union if union > 0 else 0.0
        stats["percent_reordered"] = (1.0 - stats["jaccard_at_10"]) * 100.0
        if log_stats and (stats["penalty_applied"] or stats["reordered"]):
            logger.debug("Debias stats: Jaccard@10=%.3f, %%reordered=%.1f%%, penalty_applied=%s", 
                        stats["jaccard_at_10"], stats["percent_reordered"], stats["penalty_applied"])

 
    if fairness and item_pop_rank is not None:
        head_n = fairness.get("head", 0)
        mid_n = fairness.get("mid", 0)
        tail_n = fairness.get("tail", 0)
        total = len(item_pop_rank)
        t1 = max(1, total // 3)
        t2 = max(1, 2 * total // 3)
        total_head, total_mid = t1, t2 - t1
        total_tail = total - t2
        lst = apply_fairness_head_mid_tail(
            lst, topk, item_pop_rank,
            head_n, mid_n, tail_n,
            total_head, total_mid, total_tail,
        )


    out = _list_to_scored(lst[:topk], reason="diversified")

    # IMPORTANT: For MMR/xQuAD/fairness, the list order is a greedy selection order,
    # so relevance scores are not guaranteed to be monotonic.
    # Keep original relevance in a separate field and set a rank-based monotonic score.
    if stats.get("reordered") or fairness:
        n = len(out)
        for rank, rec in enumerate(out):
            rec["relevance"] = float(rec.get("score", 0.0))
            rec["score"] = float(n - rank)  # strictly non-increasing, preserves ranking order

    return out, stats
