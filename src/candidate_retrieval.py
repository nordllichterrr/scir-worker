import hashlib
from .vector_index import search_index
from .embeddings import load_embeddings
from .config import CANDIDATE_POOL, EMBED_MODEL
from .utils import logger, get_embedder
from .bm25 import BM25Index
import numpy as np
import os
from .vark_simulator import preference_match_score


_query_encoder = None
_bm25_index = None
_bm25_item_ids = None


def _stable_int_hash(s: str) -> int:
    """Deterministic int hash (unlike Python hash(), which is salted per process)."""
    s = "" if s is None else str(s)
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)

def _get_query_encoder():
    global _query_encoder
    if _query_encoder is None:
    
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        logger.info("Loading SentenceTransformer model for queries: %s", EMBED_MODEL)
        try:
            _query_encoder = get_embedder(EMBED_MODEL)
        except Exception as e:
    
            logger.error(
                "Failed to load query encoder '%s': %s. Falling back to BM25/popularity only.",
                EMBED_MODEL,
                e,
            )
            _query_encoder = None
    return _query_encoder


def _get_bm25_index(items_list: list) -> tuple:

    global _bm25_index, _bm25_item_ids
    if _bm25_index is not None and _bm25_item_ids is not None:
        return _bm25_index, _bm25_item_ids

    texts = []
    ids = []
    for it in items_list:
        item_id = str(it.get("item_id"))
        title = str(it.get("title", "") or "")
        genres = str(it.get("genres", "") or "")
        format_tags = it.get("format_tags") or it.get("tags") or ""
        category = it.get("category") or it.get("categories") or ""
        desc = it.get("description") or it.get("text") or ""

        # нормализуем списки в строки
        if isinstance(format_tags, (list, tuple, set)):
            format_tags = ", ".join(str(x) for x in format_tags)
        if isinstance(category, (list, tuple, set)):
            category = ", ".join(str(x) for x in category)
        if isinstance(desc, (list, tuple, set)):
            desc = " ".join(str(x) for x in desc)

        text = f"{title} {genres} {format_tags} {category} {desc}".strip()
        texts.append(text)
        ids.append(item_id)

    logger.info("Building BM25 index over %d items (title + genres)...", len(texts))
    _bm25_index = BM25Index(texts)
    _bm25_item_ids = ids
    return _bm25_index, _bm25_item_ids


def _get_popularity_candidates(items_list: list, top_m: int, pop_key: str = "pop") -> list:
    sorted_items = sorted(
        items_list,
        key=lambda x: float(x.get(pop_key, 0) or 0),
        reverse=True,
    )
    return [str(it.get("item_id")) for it in sorted_items[:top_m]]


def get_candidates_for_user(
    user_profile: dict,
    items_list: list,
    faiss_index=None,
    id2idx: dict = None,
    embeddings: np.ndarray = None,
    graph=None,
    pool_size: int = CANDIDATE_POOL,
    retrieval_mode: str = "ann",
    ann_overfetch_factor: int = 4,
    bm25_overfetch_factor: int = 4,
    hybrid_union_max: int = None,
    preference_prior_mode: str = "prior_plus_context",
):
 
    model = _get_query_encoder()
    qtext = user_profile.get("text_profile") or user_profile.get("goal") or ""
    if len(qtext.strip()) == 0:
        user_id = user_profile.get("user_id", "")
        vark = user_profile.get("vark", "visual")
        qtext = f"user_vark:{vark} user_id:{user_id}"


    m_ann = max(pool_size * ann_overfetch_factor, pool_size + 500)
    m_bm25 = max(pool_size * bm25_overfetch_factor, pool_size + 500)
    m_pop = pool_size  

    ann_candidates = []
    ann_scores_dict = {}  # item_id -> similarity score
    if faiss_index is not None and id2idx is not None and embeddings is not None:
        top_k = m_ann
        precomputed = user_profile.get("query_vector")
        if precomputed is not None:
            qvec = np.asarray(precomputed, dtype=np.float32)
            if qvec.ndim == 1:
                qvec = qvec.reshape(1, -1)
        else:
            model = _get_query_encoder()
     
            if model is None:
                logger.warning(
                    "Query encoder unavailable; skipping ANN retrieval and using BM25/popularity only."
                )
                qvec = None
            else:
                qvec = model.encode([qtext], convert_to_numpy=True)
        if qvec is not None:
            D, I = search_index(faiss_index, qvec, top_k=top_k)
            idx2id = {v: k for k, v in id2idx.items()}
            for i, idx in enumerate(I[0]):
                item_id = idx2id.get(str(idx)) or idx2id.get(int(idx)) or None
                if item_id:
                    ann_candidates.append(str(item_id))
                    # Store similarity score (higher is better)
                    if i < len(D[0]):
                        ann_scores_dict[str(item_id)] = float(D[0][i])
                if len(ann_candidates) >= top_k:
                    break


    bm25_candidates = []
    bm25_scores_dict = {}  # item_id -> BM25 score
    if retrieval_mode in ("bm25", "hybrid"):
        bm25_index, bm25_item_ids = _get_bm25_index(items_list)
        top_k = m_bm25
        idxs, scores = bm25_index.search(qtext, top_k=top_k)
        for i, idx in enumerate(idxs):
            if 0 <= idx < len(bm25_item_ids):
                item_id = str(bm25_item_ids[idx])
                bm25_candidates.append(item_id)
                if i < len(scores):
                    bm25_scores_dict[item_id] = float(scores[i])
            if len(bm25_candidates) >= top_k:
                break

    popularity_candidates = []
    pop_rank_dict = {}  # item_id -> popularity rank (lower rank = more popular)
    if retrieval_mode == "hybrid":
        popularity_candidates = _get_popularity_candidates(items_list, top_m=m_pop)
        # Create rank dict (lower rank = higher popularity)
        for rank, item_id in enumerate(popularity_candidates):
            pop_rank_dict[str(item_id)] = rank

    retrieval_meta = {
        "retrieval_mode": retrieval_mode,
        "ann_k": len(ann_candidates) if ann_candidates else None,
        "bm25_k": len(bm25_candidates) if bm25_candidates else None,
        "popularity_k": len(popularity_candidates) if popularity_candidates else None,
        "overlap": None,
    }
    if retrieval_mode == "bm25":
        candidates = bm25_candidates
    elif retrieval_mode == "hybrid":

        ann_set = set(ann_candidates)
        bm25_set = set(bm25_candidates)
        pop_set = set(popularity_candidates)
        retrieval_meta["overlap"] = len(ann_set & bm25_set)  
        seen = set()
        candidates = []
        candidate_scores = {}  # item_id -> combined score for sorting
        
        # Pre-compute normalization factors
        bm25_max = max(bm25_scores_dict.values()) if bm25_scores_dict else 1.0
        bm25_max = max(bm25_max, 1e-6)  # Avoid division by zero
        
        # Collect all unique candidates with their scores
        for rank, iid in enumerate(ann_candidates):
            if iid not in seen:
                seen.add(iid)
                # ANN score is cosine similarity (0-1 range after normalization), higher is better
                ann_score = ann_scores_dict.get(iid, 0.0)
                ann_norm = max(0.0, min(1.0, ann_score))  # Clamp to [0, 1]
                candidate_scores[iid] = {"ann": ann_norm, "ann_rank": rank}
        
        for rank, iid in enumerate(bm25_candidates):
            if iid not in seen:
                seen.add(iid)
                bm25_score = bm25_scores_dict.get(iid, 0.0)
                bm25_norm = bm25_score / bm25_max
                candidate_scores[iid] = {"bm25": bm25_norm, "bm25_rank": rank}
            else:
                # Update existing candidate with BM25 info
                if iid in candidate_scores:
                    bm25_score = bm25_scores_dict.get(iid, 0.0)
                    bm25_norm = bm25_score / bm25_max
                    candidate_scores[iid]["bm25"] = bm25_norm
                    candidate_scores[iid]["bm25_rank"] = rank
        
        for rank, iid in enumerate(popularity_candidates):
            if iid not in seen:
                seen.add(iid)
                # Popularity rank (lower = more popular, normalize to 0-1)
                pop_rank = pop_rank_dict.get(iid, len(popularity_candidates))
                pop_norm = 1.0 / (1.0 + pop_rank)  # Higher score for more popular
                candidate_scores[iid] = {"pop": pop_norm, "pop_rank": pop_rank}
            else:
                # Update existing candidate with popularity info
                if iid in candidate_scores:
                    pop_rank = pop_rank_dict.get(iid, len(popularity_candidates))
                    pop_norm = 1.0 / (1.0 + pop_rank)
                    candidate_scores[iid]["pop"] = pop_norm
                    candidate_scores[iid]["pop_rank"] = pop_rank
        
        # Reciprocal Rank Fusion (RRF): score = sum 1/(k + rank) over sources.
        # RRF preserves user-specific orderings (ANN/BM25 differ per user) and avoids
        # popularity dominating when relevance scores are similar.
        RRF_K = 60  # standard constant

        def compute_rrf_score(item_id):
            scores = candidate_scores.get(item_id, {})
            rrf = 0.0
            ann_rank = scores.get("ann_rank")
            if ann_rank is not None:
                rrf += 1.0 / (RRF_K + ann_rank)
            bm25_rank = scores.get("bm25_rank")
            if bm25_rank is not None:
                rrf += 1.0 / (RRF_K + bm25_rank)
            pop_rank = scores.get("pop_rank")
            if pop_rank is not None:
                rrf += 0.5 / (RRF_K + pop_rank)  # downweight popularity so user signals dominate
            if preference_prior_mode != "no_prior":
                meta = next((it for it in items_list if str(it.get("item_id")) == str(item_id)), {})
                pref_bonus = preference_match_score(
                    user_profile,
                    meta,
                    use_context=(preference_prior_mode == "prior_plus_context"),
                )
                rrf += 0.1 * float(pref_bonus)
            return rrf

        # Sort by RRF (higher = better). Tie-break by hash(user_id, item_id) so different
        # users get different orderings when scores are equal (reduces top-1 collapse).
        # For cold-start users with similar profiles, add user-specific randomization.
        user_id = user_profile.get("user_id", "")
        user_hash = hash(str(user_id)) % 10000  # User-specific offset for randomization
        
        def sort_key(item):
            iid, score = item
            # Multi-level tie-breaking: hash(user_id+item_id), then item_id hash, then user offset
            tie1 = hash((str(user_id), str(iid))) % (2 ** 32)
            tie2 = hash(str(iid)) % (2 ** 16)
            # Add small user-specific offset to break ties for cold-start users
            user_offset = (user_hash % 100) / 100000.0  # Very small offset (0.0000-0.0001)
            return (score + user_offset, tie1, tie2)
        candidates_with_scores = [(iid, compute_rrf_score(iid)) for iid in seen]
        candidates_with_scores.sort(key=sort_key, reverse=True)
        candidates = [iid for iid, _ in candidates_with_scores]

        # --- Anti-collapse fallback ---
        # If ANN is unavailable AND BM25 has essentially no signal (all ~0),
        # rotate popularity candidates by user_id so top-1 isn't identical for everyone.
        bm25_max_raw = max(bm25_scores_dict.values()) if bm25_scores_dict else 0.0
        if (not ann_candidates) and (bm25_max_raw < 1e-8) and popularity_candidates:
            offset = _stable_int_hash(user_profile.get("user_id", "")) % len(popularity_candidates)
            rotated_pop = popularity_candidates[offset:] + popularity_candidates[:offset]
            pop_set = set(rotated_pop)
            candidates = rotated_pop + [iid for iid in candidates if iid not in pop_set]
        # --- end anti-collapse fallback ---
        
        max_union = hybrid_union_max if hybrid_union_max is not None else 3 * pool_size
        if len(candidates) > max_union:
            candidates = candidates[:max_union]
        retrieval_meta["union_size"] = len(candidates)
    else:
        candidates = ann_candidates[:pool_size]

    if graph is not None and candidates and len(candidates) < 2 * pool_size:
        tokens = set(qtext.lower().split())
        for node, data in graph.nodes(data=True):
            if data.get("type") == "entity":
                name = data.get("name", "").lower()
                if any(t in name for t in tokens):
                    for nbr in graph.neighbors(node):
                        if nbr.startswith("item:"):
                            iid = nbr.split("item:")[1]
                            if iid not in candidates:
                                candidates.append(iid)
                                if len(candidates) >= 2 * pool_size:
                                    break
            if len(candidates) >= 2 * pool_size:
                break

    if retrieval_mode != "hybrid" and len(candidates) < pool_size:
        for it in items_list:
            iid = str(it["item_id"])
            if iid not in candidates:
                candidates.append(iid)
            if len(candidates) >= pool_size:
                break

    out = candidates if retrieval_mode == "hybrid" else candidates[:pool_size]
    logger.info(
        "Candidate pool size for user (%s mode): %d",
        retrieval_mode,
        len(out),
    )
    return out, retrieval_meta
