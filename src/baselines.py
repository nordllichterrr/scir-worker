
import random
import numpy as np
from typing import List, Dict, Any, Set, Optional
from collections import Counter
from .utils import logger
from .embeddings import get_embedder, load_embeddings
from .config import EMBED_MODEL, EMBEDDING_MAP, EMBEDDINGS_NPY


def random_baseline(items_list: List[Dict], k: int = 10, seed: int = None) -> List[Dict]:
    """
    Uniform random top-k from the catalog. If ``seed`` is fixed for all users,
    every user gets the same recommendation list — pass a per-user sub-seed
    (e.g. ``utils.per_user_seed(global_seed, user_id)``) when evaluating many users.
    """
    if len(items_list) < k:
        k = len(items_list)

    if seed is not None:
        rng = random.Random(seed)
        selected = rng.sample(items_list, k)
    else:
        selected = random.sample(items_list, k)

    return [{"item_id": str(it.get("item_id")), "score": 0.0, "method": "random"} for it in selected]


def popularity_baseline(items_list: List[Dict], k: int = 10, pop_key: str = "pop") -> List[Dict]:

    sorted_items = sorted(items_list, key=lambda x: float(x.get(pop_key, 0)), reverse=True)
    top_k = sorted_items[:k]
    
    return [
        {
            "item_id": str(it.get("item_id")),
            "score": float(it.get(pop_key, 0)),
            "method": "popularity"
        }
        for it in top_k
    ]


def embedding_cosine_baseline(
    user_profile: Dict[str, Any],
    items_list: List[Dict],
    embeddings: np.ndarray,
    id2idx: Dict[str, int],
    k: int = 10,
    pool_size: int = 1000
) -> List[Dict]:
 
    qtext = user_profile.get("text_profile") or user_profile.get("goal") or ""
    if len(qtext.strip()) == 0:
        user_id = user_profile.get("user_id", "")
        vark = user_profile.get("vark", "visual")
        qtext = f"user_vark:{vark} user_id:{user_id}"
    

    model = get_embedder(EMBED_MODEL)
    user_vec = model.encode([qtext], convert_to_numpy=True)[0]  
    
    similarities = []
    for item in items_list[:pool_size]: 
        item_id = str(item.get("item_id"))
        idx = id2idx.get(item_id)
        
        if idx is None or idx >= len(embeddings):
            continue
        
        item_vec = embeddings[idx]
        
    
        dot_product = np.dot(user_vec, item_vec)
        norm_user = np.linalg.norm(user_vec)
        norm_item = np.linalg.norm(item_vec)
        
        if norm_user > 0 and norm_item > 0:
            cosine_sim = dot_product / (norm_user * norm_item)
        else:
            cosine_sim = 0.0
        
        similarities.append({
            "item_id": item_id,
            "score": float(cosine_sim),
            "method": "embedding_cosine"
        })
    

    similarities.sort(key=lambda x: x["score"], reverse=True)
    return similarities[:k]


def oracle_upper_bound_baseline(
    items_list: List[Dict],
    gt_set: Set[str],
    k: int = 10
) -> List[Dict]:
    
    gt_items = []
    non_gt_items = []
    
    for item in items_list:
        item_id = str(item.get("item_id"))
        if item_id in gt_set:
            gt_items.append({
                "item_id": item_id,
                "score": 1.0,
                "method": "oracle_upper_bound"
            })
        else:
            non_gt_items.append({
                "item_id": item_id,
                "score": 0.0,
                "method": "oracle_upper_bound"
            })
    

    result = gt_items + non_gt_items
    return result[:k]


def random_in_candidate_pool_baseline(
    candidate_pool: List[str],
    k: int = 10,
    seed: int = None
) -> List[Dict]:
    """Same as ``random_baseline``: use a per-user seed when ``seed`` is not None."""
    if len(candidate_pool) < k:
        k = len(candidate_pool)
    
    if seed is not None:
        rng = random.Random(seed)
        selected_ids = rng.sample(candidate_pool, k)
    else:
        selected_ids = random.sample(candidate_pool, k)
    
    return [
        {
            "item_id": str(item_id),
            "score": 0.0,
            "method": "random_in_candidate_pool"
        }
        for item_id in selected_ids
    ]


def get_baseline_recommendations(
    baseline_type: str,
    user_profile: Dict[str, Any],
    items_list: List[Dict],
    k: int = 10,
    embeddings: np.ndarray = None,
    id2idx: Dict[str, int] = None,
    seed: int = None,
    gt_set: Optional[Set[str]] = None,
    candidate_pool: Optional[List[str]] = None
) -> List[Dict]:
   
    if baseline_type == "random":
        return random_baseline(items_list, k=k, seed=seed)
    elif baseline_type == "popularity":
        return popularity_baseline(items_list, k=k)
    elif baseline_type == "embedding_cosine":
        if embeddings is None or id2idx is None:
            raise ValueError("embeddings and id2idx required for embedding_cosine baseline")
        return embedding_cosine_baseline(user_profile, items_list, embeddings, id2idx, k=k)
    elif baseline_type == "oracle_upper_bound":
        if gt_set is None:
            raise ValueError("gt_set required for oracle_upper_bound baseline")
        return oracle_upper_bound_baseline(items_list, gt_set, k=k)
    elif baseline_type == "random_in_candidate_pool":
        if candidate_pool is None:
            raise ValueError("candidate_pool required for random_in_candidate_pool baseline")
        return random_in_candidate_pool_baseline(candidate_pool, k=k, seed=seed)
    elif baseline_type in ("itemknn", "ease", "mf"):
        raise ValueError(
            f"Use run_experiment with config.baseline='{baseline_type}' for strong baselines (they need training data and model)."
        )
    else:
        raise ValueError(
            f"Unknown baseline type: {baseline_type}. Use 'random', 'popularity', 'embedding_cosine', 'oracle_upper_bound', 'random_in_candidate_pool', 'itemknn', 'ease', or 'mf'."
        )
