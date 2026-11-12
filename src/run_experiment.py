
import csv
import random
import time
from .evaluation_config import get_eval_paths, normalize_dataset_key
from pathlib import Path
from .utils import logger, set_seed, save_json, per_user_seed
from .llm_enrich import LLMEnricher, load_items_from_csv
from .embeddings import build_embeddings, load_embeddings
from .vector_index import build_faiss_index, load_faiss_index
from .graph_builder import build_graph_from_items, add_similarity_edges
from .candidate_retrieval import get_candidates_for_user
from .baselines import get_baseline_recommendations
from .baselines_strong import get_strong_baseline_model, strong_baseline_recommend
from .run_logger import log_run
import json
from .vark_simulator import build_user_profile_from_minimal
import numpy as np
from .config import HF_MODEL_NAME, HF_DEVICE
from .sanity_checks import sanity_check_rankings
from .config import PROCESSED_DIR, EMBED_MODEL, EMBEDDINGS_NPY, EMBEDDING_MAP, FAISS_INDEX_PATH, RESULTS_DIR

from .rerank_llm import CrossReranker


def run_experiment(
    n_users: int = None,
    seed: int = 42,
    split_seed: int = 42,
    init_seed: int = 42,
    config: dict = None,
    dataset: str = "serendipity",
) -> dict:
    """
    Run full experiment pipeline.
    
    Args:
        n_users: Number of users to evaluate (default: from evaluation_config, 500)
        seed: Random seed
        config: Experiment configuration dict with keys:
            - baseline: str, one of 'random', 'popularity', 'embedding_cosine', 'itemknn', 'ease', 'mf', or None for full pipeline.
                Strong baselines (itemknn, ease, mf) require training CSV; EASE/MF are trained on first run and cached under data/processed/baselines/.
            - use_reranker: bool, whether to use cross-encoder reranker
            - use_profile_enrichment: bool, whether to enrich user profile (future)
            - use_bias_penalty: bool, whether to apply bias penalty (future)
            - reranker_model: str, model name for reranker
            - topk: int, number of final recommendations
            - candidate_pool_size: int, size of ANN / BM25 candidate pool (per-source M in hybrid)
            - retrieval_mode: 'ann' (default), 'bm25', or 'hybrid' (union ANN ∪ BM25 ∪ popularity for recall)
            - hybrid_union_max: int or None; max union size in hybrid (default 3*candidate_pool_size)
            - use_viewed_items_profile: bool; if True and training.csv exists, build text_profile from last k viewed items for reranker
            - viewed_items_k: int; number of recent train items to use for profile (default 5)
            - profile_type: str; 'last_k_items' (default), 'summary', or 'centroid'. For ablation: last_k = long text; summary = short "User likes: ..."; centroid = ANN query = mean of train item embeddings.
            - diversify_config: optional dict for anti-bias/diversification at rerank:
                - popularity_penalty_alpha: score -= alpha*log(1+pop)
                - exposure_beta: score -= beta*exposure (use with exposure_map)
                - mmr_lambda: MMR relevance − λ·sim_to_selected (needs item_embeddings, id2idx)
                - xquad_lambda: intent-aware diversification over category_key (e.g. genres)
                - category_key: "genres" or "format_tags"
                - fairness: { "head": n, "mid": n, "tail": n } slot quotas by popularity rank
            - exposure_map: optional dict item_id -> cumulative exposure for exposure_beta
            - two_head_config: optional dict for relevance+novelty multi-objective:
                - alpha: weight for relevance in [0,1] (1 = only relevance, 0 = only novelty)
                - mode: "scalarize" (default) or "pareto_balanced" (min(rel,nov) ranking)
                - catalog_size: optional; default from items
            (Novelty from popularity rank: tail = high novelty; coverage/mean_popularity_rank in diagnostics.)
            
    Returns:
        Dict with results: {user_id: [rec_dicts]}
    """
    if config is None:
        config = {
            "baseline": None,  # None = full pipeline
            "use_reranker": True,
            "use_profile_enrichment": False,
            "use_bias_penalty": False,
            "use_viewed_items_profile": True,
            "viewed_items_k": 5,
            "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "topk": 10,
            "candidate_pool_size": 1000,
            "retrieval_mode": "ann",
            "preference_prior_mode": "prior_plus_context",
            "dataset": dataset,
        }
    else:

        config = {
            **{
                "baseline": None,
                "use_reranker": True,
                "use_profile_enrichment": False,
                "use_bias_penalty": False,
                "use_viewed_items_profile": True,
                "viewed_items_k": 5,
                "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "topk": 10,
                "candidate_pool_size": 1000,
                "retrieval_mode": "ann",
                "preference_prior_mode": "prior_plus_context",
                "dataset": dataset,
            },
            **config,
        }
    run_meta = {}
    from .evaluation_config import N_TEST_USERS, get_eval_paths
    if n_users is None:
        n_users = N_TEST_USERS

    set_seed(seed)
    

    dataset_key = normalize_dataset_key(config.get("dataset", dataset))
    # Preprocessing is intentionally excluded from experiment execution.
    # --- Dataset-specific eval files (train/gt/split_meta) ---
    # 
    paths = get_eval_paths(dataset_key)

    # If user didn't pass explicit paths, use dataset defaults
    if not config.get("train_interactions_path"):
        config["train_interactions_path"] = str(paths["train"])
    if not config.get("ground_truth_path"):
        config["ground_truth_path"] = str(paths["gt"])
    if not config.get("split_metadata_path"):
        config["split_metadata_path"] = str(paths["split_meta"])

    suffix = dataset_key

    items_csv = PROCESSED_DIR / f"items_{suffix}.csv"

    # dataset-specific cache files (so serendipity and taobao don't overwrite each other)
    from .config import EMBEDDINGS_DIR, INDEX_DIR

    embeddings_npy = EMBEDDINGS_DIR / f"item_embeddings_{suffix}.npy"
    embedding_map  = EMBEDDINGS_DIR / f"id2idx_{suffix}.json"
    faiss_index_path = INDEX_DIR / f"items_{suffix}.faiss"
    if not items_csv.exists():
        logger.error("Items csv not found at %s. Run preprocess properly.", items_csv)
        return {}
    items = load_items_from_csv(items_csv)

    # Normalize item_id to str to avoid int/str mismatches downstream
    for it in items:
        if isinstance(it, dict) and "item_id" in it and it["item_id"] is not None:
            it["item_id"] = str(it["item_id"])
    
    enr = LLMEnricher(backend="heuristic")
    enriched = enr.enrich_items_list(items)

    # Use canonical dataset-specific paths defined above.
    emb, id2idx = build_embeddings(
        enriched,
        out_npy=embeddings_npy,
        map_path=embedding_map,
    )
    index = build_faiss_index(
        emb,
        index_path=faiss_index_path,
    )
    

    G = None
    if config.get("use_graph", True):
        G = build_graph_from_items(enriched)
        G = add_similarity_edges(G, enriched, emb, id2idx, top_k=5)
    
    reranker = None
    if config.get("use_reranker") and config.get("baseline") is None:
        two_head_cfg = config.get("two_head_config")
        if two_head_cfg is not None and isinstance(two_head_cfg, dict):
            two_head_cfg = dict(two_head_cfg)
            if two_head_cfg.get("catalog_size") is None:
                two_head_cfg["catalog_size"] = len(enriched)
        reranker = CrossReranker(
            model_name=config.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            items_meta=enriched,
            device=config.get("reranker_device", "cpu"),
            item_embeddings=emb,
            id2idx=id2idx,
            diversify_config=config.get("diversify_config"),
            exposure_map=config.get("exposure_map"),
            two_head_config=two_head_cfg,
            use_fp16=config.get("reranker_fp16", False),
            max_length=config.get("reranker_max_length"), 
            two_stage_config=config.get("two_stage_rerank_config"), 
        )
    
    # Dataset-specific eval file paths (gt/train/meta)
    eval_paths = get_eval_paths(dataset_key)

    gt_path = Path(config.get("ground_truth_path") or eval_paths["gt"])
    split_meta_path = Path(config.get("split_metadata_path") or eval_paths["split_meta"])
    split_train_path = Path(config.get("train_interactions_path") or eval_paths["train"])
    from .evaluate_results import load_gt_struct

    gt_wrapper = load_gt_struct(gt_path)
    gt_data = gt_wrapper["data"]

    # Normalize GT to str ids (keys=user_id, values=item_id)
    norm_gt = {}
    for uid, vals in (gt_data or {}).items():
        uid_s = str(uid)
        if isinstance(vals, dict) and "items" in vals:
            vals = vals["items"]
        if vals is None:
            norm_gt[uid_s] = []
        else:
            norm_gt[uid_s] = [str(x) for x in vals]
    gt_data = norm_gt
    user_pool = sorted(list(gt_data.keys()))

    # ---- Paper protocol: optional filtering of evaluation users ----
    eval_scenario = (config.get("eval_scenario") or "").strip().lower()
    eval_bucket = (config.get("eval_bucket") or "").strip()
    eval_min_train = config.get("eval_min_train_interactions")
    eval_max_train = config.get("eval_max_train_interactions")

    if eval_scenario or eval_bucket or eval_min_train is not None or eval_max_train is not None:
        try:
            split_meta = json.load(open(split_meta_path, "r", encoding="utf-8"))
        except Exception:
            split_meta = {}

        scenario_to_users = split_meta.get("scenario_to_users") or {}
        bucket_to_users = split_meta.get("bucket_to_users") or {}
        user_meta = split_meta.get("user_meta") or {}

        if eval_scenario and eval_scenario != "all":
            allowed = set(str(u) for u in scenario_to_users.get(eval_scenario, []))
            user_pool = [u for u in user_pool if u in allowed]

        if eval_bucket:
            allowed = set(str(u) for u in bucket_to_users.get(eval_bucket, []))
            user_pool = [u for u in user_pool if u in allowed]

        if eval_min_train is not None or eval_max_train is not None:
            lo = int(eval_min_train) if eval_min_train is not None else None
            hi = int(eval_max_train) if eval_max_train is not None else None
            filtered = []
            for u in user_pool:
                n_tr = int((user_meta.get(str(u)) or {}).get("n_train_interactions", 0))
                if lo is not None and n_tr < lo:
                    continue
                if hi is not None and n_tr > hi:
                    continue
                filtered.append(u)
            user_pool = filtered

        user_pool = sorted(user_pool)
    # ---- end paper protocol filtering ----

    
    if len(user_pool) < n_users:
        logger.warning("Requested %d users but GT has only %d users; reducing n_users.", n_users, len(user_pool))
        n_users = len(user_pool)
    
    selected_users = random.sample(user_pool, n_users)


    leakage_gt_profile_overlap = 0
    leakage_gt_profile_overlap_examples = []
    gt_sizes = []
    gt_hit_in_candidates = 0
    gt_full_covered_in_candidates = 0
    gt_recall_frac_in_candidates = []

    train_items_by_user = {}
    if config.get("use_viewed_items_profile", True):

        override_train = config.get("train_interactions_path") if config else None
        if override_train:
            train_path = Path(override_train)
        else:
            # split_train_path = Path("experiments") / "training_interactions.csv"
            if split_train_path.exists():
                train_path = split_train_path
            else:
                try:
                    from .evaluation_config import TRAINING_CSV_PATH
                    train_path = TRAINING_CSV_PATH
                except Exception:
                    train_path = Path("data/serendipity-sac2018/training.csv")
        if train_path.exists():
            with open(train_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                u_col = None
                i_col = None
                for row in reader:
                    if not row:
                        continue
                    if u_col is None:
                        keys = [k for k in row.keys() if k]
                        u_col = next((k for k in keys if "user" in k.lower() and "id" in k.lower()), keys[0] if keys else None)
                        i_col = next((k for k in keys if "item" in k.lower() or "movie" in k.lower()), keys[1] if len(keys) > 1 else None)
                    if u_col and i_col:
                        uid, iid = str(row.get(u_col, "")), str(row.get(i_col, ""))
                        if uid and iid:
                            train_items_by_user.setdefault(uid, []).append(iid)
            logger.info("Loaded training interactions for %d users (for viewed-items profile).", len(train_items_by_user))
    viewed_k = config.get("viewed_items_k", 5)
    enriched_by_id = {str(it["item_id"]): it for it in enriched} if enriched else {}


    strong_baseline_model = None
    if config.get("baseline") in ("itemknn", "ease", "mf"):
        override_train = config.get("train_interactions_path") if config else None
        if override_train:
            train_path = Path(override_train)
        else:
            # split_train_path = Path("experiments") / "training_interactions.csv"
            if split_train_path.exists():
                train_path = split_train_path
            else:
                try:
                    from .evaluation_config import TRAINING_CSV_PATH
                    train_path = TRAINING_CSV_PATH
                except Exception:
                    train_path = Path("data/serendipity-sac2018/training.csv")
        if not train_path.exists():
            train_path = Path("data/serendipity-sac2018/training.csv")
        if not train_path.exists():
            raise FileNotFoundError(f"Training data for strong baseline '{config.get('baseline')}' not found at {train_path}")

        if not train_items_by_user:
            with open(train_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                u_col = i_col = None
                for row in reader:
                    if not row:
                        continue
                    if u_col is None:
                        keys = [k for k in (reader.fieldnames or []) if k]
                        u_col = next((k for k in keys if "user" in k.lower() and "id" in k.lower()), keys[0] if keys else None)
                        i_col = next((k for k in keys if "item" in k.lower() or "movie" in k.lower()), keys[1] if len(keys) > 1 else None)
                    if u_col and i_col:
                        uid, iid = str(row.get(u_col, "")), str(row.get(i_col, ""))
                        if uid and iid:
                            train_items_by_user.setdefault(uid, []).append(iid)
            logger.info("Loaded training for strong baselines: %d users.", len(train_items_by_user))

        item_ids_catalog = [str(it["item_id"]) for it in enriched]

        # Isolate baseline caches by dataset and split/fold.
        # A user-provided baselines_cache_dir is treated as the cache root.
        baseline_cache_root = Path(
            config["baselines_cache_dir"]
        ) if config.get("baselines_cache_dir") else PROCESSED_DIR / "baselines"

        split_type = str(config.get("split_type") or "standard")
        if split_type not in {"standard", "user_kfold", "item_kfold", "both_kfold"}:
            raise ValueError(f"Unsupported split_type for baseline cache: {split_type}")

        baseline_cache_dir = (
            baseline_cache_root
            / dataset_key
            / split_type
        )

        fold_index = config.get("fold_index")
        if split_type != "standard":
            if fold_index is None:
                raise ValueError(
                    f"fold_index is required for split_type={split_type}"
                )
            baseline_cache_dir = baseline_cache_dir / f"fold_{int(fold_index)}"

        strong_baseline_model = get_strong_baseline_model(
            config["baseline"],
            training_path=train_path,
            item_ids_catalog=item_ids_catalog,
            embeddings=emb if config["baseline"] == "itemknn" else None,
            id2idx=id2idx if config["baseline"] == "itemknn" else None,
            init_seed=int(init_seed),
            cache_dir=baseline_cache_dir,
        )
    

    results = {}
    results_pre = {}
    candidate_pools = {}
    candidate_pools_post = {}
    reranker_scores_all = {}
    retrieval_metas = []
    user_times = []
    retrieval_times = []

    topk = int(config.get("topk", 10) or 10)

    pool_k = int(config.get("candidate_pool_size", 1000) or 1000)
    pool_k = max(pool_k, 1000, topk)

    catalog_size = len(enriched)
    
    for user_real_id in selected_users:
        uid = str(user_real_id)
        

        all_train = train_items_by_user.get(uid, []) if train_items_by_user else []
        fewshot_cap = config.get("fewshot_train_cap")

        if fewshot_cap is not None:
            cap = int(fewshot_cap)
            if cap <= 0:
                viewed_ids = []
            else:
                viewed_ids = all_train[-cap:]
        else:
            viewed_ids = all_train[-viewed_k:] if all_train else []



        gt_items = gt_data.get(uid, [])
        gt_set = set(str(x) for x in (gt_items or []))
        gt_sizes.append(len(gt_set))
        if viewed_ids and gt_set:
            overlap = set(str(x) for x in viewed_ids) & gt_set
            if overlap:
                leakage_gt_profile_overlap += 1
                if len(leakage_gt_profile_overlap_examples) < 3:
                    leakage_gt_profile_overlap_examples.append(
                        {
                            "user_id": uid,
                            "overlap_items": list(overlap)[:10],
                            "viewed_k": int(viewed_k),
                            "gt_size": int(len(gt_set)),
                        }
                    )
        info = {
            "user_id": uid,
            "goal": "",  
            "time_of_day": config.get("session_time_of_day", "evening"),
            "session_len": config.get("session_len", 15),
            "device": config.get("device_type", "desktop"),
            "viewed_item_ids": viewed_ids,
            "items_meta": enriched,
        }
        profile = build_user_profile_from_minimal(
            info,
            prior_mode=config.get("preference_prior_mode", "prior_plus_context"),
        )
        profile_type = config.get("profile_type", "last_k_items")
        if profile_type == "summary" and viewed_ids and enriched_by_id:
            titles = [enriched_by_id.get(str(iid), {}).get("title", "") for iid in viewed_ids]
            titles = [t.strip() for t in titles if t and t.strip()]
            if titles:
                profile["text_profile"] = "User likes: " + "; ".join(titles)
        if profile_type == "centroid" and viewed_ids and emb is not None and id2idx is not None:
            vecs = []
            for iid in viewed_ids:
                idx = id2idx.get(str(iid))
                if idx is not None and idx < len(emb):
                    vecs.append(emb[int(idx)])
            if vecs:
                centroid = np.mean(vecs, axis=0).astype(np.float32)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid = centroid / norm
                profile["query_vector"] = centroid


        if config.get("baseline") is not None:
            baseline_type = config["baseline"]
            
            if baseline_type in ("random_in_candidate_pool", "content_bm25", "two_tower"):
                retrieval_mode_baseline = (
                    "bm25" if baseline_type == "content_bm25"
                    else "ann" if baseline_type == "two_tower"
                    else config.get("retrieval_mode", "bm25")
                )
                t0 = time.time()
                candidates, retrieval_meta = get_candidates_for_user(
                    profile, enriched, index, id2idx, emb,
                    graph=G,
                    pool_size=config.get("candidate_pool_size", 1000),
                    retrieval_mode=retrieval_mode_baseline,
                    hybrid_union_max=config.get("hybrid_union_max"),
                )
                retrieval_times.append(time.time() - t0)
                if not candidates:
                    logger.warning("No candidates for user %s; using fallback.", uid)
                    candidates = [str(it["item_id"]) for it in enriched[:50]]
                candidate_pools[uid] = candidates
                
                if baseline_type == "random_in_candidate_pool":
                    recs = get_baseline_recommendations(
                        baseline_type=baseline_type,
                        user_profile=profile,
                        items_list=enriched,
                        k=topk,
                        seed=per_user_seed(seed, uid),
                        candidate_pool=candidates
                    )
                else:

                    recs = [{"item_id": c, "score": 0.0, "method": baseline_type} for c in candidates[:topk]]
                results[uid] = recs
                results_pre[uid] = recs
                candidate_pools[uid] = list(candidates[:pool_k])
                
                cand_set = set(str(x) for x in (candidates or []))
                if gt_set and cand_set:
                    inter = gt_set & cand_set
                    if inter:
                        gt_hit_in_candidates += 1
                    if len(inter) == len(gt_set):
                        gt_full_covered_in_candidates += 1
                    gt_recall_frac_in_candidates.append(len(inter) / max(1, len(gt_set)))
                continue
            

            if baseline_type in ("itemknn", "ease", "mf") and strong_baseline_model is not None:
                user_train = train_items_by_user.get(uid, [])
                item_ids_catalog = [str(it["item_id"]) for it in enriched]

                def _popularity_fallback_for_user():
                    sorted_items = sorted(
                        enriched, key=lambda x: float(x.get("pop", 0) or 0), reverse=True
                    )
                    user_hash = hash(str(uid)) % min(100, max(1, len(sorted_items) // 2))
                    selected_items = []
                    seen = set()
                    for i in range(len(sorted_items)):
                        idx = (i + user_hash) % len(sorted_items)
                        item = sorted_items[idx]
                        item_id = str(item["item_id"])
                        if item_id not in seen:
                            selected_items.append(item)
                            seen.add(item_id)
                        if len(selected_items) >= topk:
                            break
                    if len(selected_items) < topk:
                        for item in sorted_items:
                            item_id = str(item["item_id"])
                            if item_id not in seen:
                                selected_items.append(item)
                                seen.add(item_id)
                            if len(selected_items) >= topk:
                                break
                    return [
                        {"item_id": str(it["item_id"]), "score": float(it.get("pop", 0) or 0), "reason": "popularity_fallback"}
                        for it in selected_items[:topk]
                    ], [str(it["item_id"]) for it in sorted_items[:pool_k]]

            
                if not user_train:
                    fallback_recs, fallback_pool = _popularity_fallback_for_user()
                    results[uid] = fallback_recs
                    candidate_pools[uid] = fallback_pool
                else:
                    full_ranking = strong_baseline_recommend(
                        baseline_type,
                        strong_baseline_model,
                        user_id=uid,
                        user_train_items=user_train,
                        item_ids_candidate=item_ids_catalog,
                        k=pool_k,
                    )
                    if not full_ranking:
                        fallback_recs, fallback_pool = _popularity_fallback_for_user()
                        results[uid] = fallback_recs
                        candidate_pools[uid] = fallback_pool
                    else:
                        results[uid] = full_ranking[:topk]
                        candidate_pools[uid] = [str(r["item_id"]) for r in full_ranking[:pool_k]]
                results_pre[uid] = results[uid]
                candidate_pools_post[uid] = candidate_pools[uid]
            else:
    
                if baseline_type == "oracle_upper_bound":
                    recs = get_baseline_recommendations(
                        baseline_type=baseline_type,
                        user_profile=profile,
                        items_list=enriched,
                        k=topk,
                        gt_set=gt_set
                    )
                    results[uid] = recs
                elif baseline_type == "popularity":
                    sorted_items = sorted(enriched, key=lambda x: float(x.get("pop", 0) or 0), reverse=True)
                    user_hash = hash(str(uid)) % min(100, max(1, len(sorted_items) // 2))
                    selected_items = []
                    seen = set()
                    for i in range(len(sorted_items)):
                        idx = (i + user_hash) % len(sorted_items)
                        item = sorted_items[idx]
                        item_id = str(item["item_id"])
                        if item_id not in seen:
                            selected_items.append(item)
                            seen.add(item_id)
                        if len(selected_items) >= topk:
                            break
                    if len(selected_items) < topk:
                        for item in sorted_items:
                            item_id = str(item["item_id"])
                            if item_id not in seen:
                                selected_items.append(item)
                                seen.add(item_id)
                            if len(selected_items) >= topk:
                                break
                    recs = [
                        {"item_id": str(it["item_id"]), "score": float(it.get("pop", 0) or 0), "method": "popularity"}
                        for it in selected_items[:topk]
                    ]
                    results[uid] = recs
                    candidate_pools[uid] = [str(it["item_id"]) for it in sorted_items[:pool_k]]
                else:
                    recs = get_baseline_recommendations(
                        baseline_type=baseline_type,
                        user_profile=profile,
                        items_list=enriched,
                        k=topk,
                        embeddings=emb if baseline_type == "embedding_cosine" else None,
                        id2idx=id2idx if baseline_type == "embedding_cosine" else None,
                        seed=per_user_seed(seed, uid) if baseline_type == "random" else seed,
                    )
                    results[uid] = recs
                
                if baseline_type == "random":
                    user_rng = random.Random(per_user_seed(seed, uid))
                    shuffled_items = enriched.copy()
                    user_rng.shuffle(shuffled_items)
                    candidate_pools[uid] = [str(it["item_id"]) for it in shuffled_items[:pool_k]]
                elif baseline_type == "popularity":
                    pass  # already set above
                elif baseline_type == "embedding_cosine":
                    qtext = profile.get("text_profile") or profile.get("goal") or ""
                    if len(qtext.strip()) == 0:
                        qtext = "user_vark:" + profile.get("vark", "visual")
                    from .embeddings import get_embedder
                    from .config import EMBED_MODEL
                    model = get_embedder(EMBED_MODEL)
                    user_vec = model.encode([qtext], convert_to_numpy=True)[0]
                    similarities = []
                    for item in enriched:
                        item_id = str(item.get("item_id"))
                        idx = id2idx.get(item_id)
                        if idx is None or idx >= len(emb):
                            continue
                        item_vec = emb[idx]
                        dot_product = np.dot(user_vec, item_vec)
                        norm_user = np.linalg.norm(user_vec)
                        norm_item = np.linalg.norm(item_vec)
                        if norm_user > 0 and norm_item > 0:
                            cosine_sim = dot_product / (norm_user * norm_item)
                        else:
                            cosine_sim = 0.0
                        similarities.append((item_id, float(cosine_sim)))
                    similarities.sort(key=lambda x: x[1], reverse=True)
                    candidate_pools[uid] = [item_id for item_id, _ in similarities[:pool_k]]
                elif baseline_type == "oracle_upper_bound":
                    gt_items = [str(it["item_id"]) for it in enriched if str(it.get("item_id")) in gt_set]
                    non_gt_items = [str(it["item_id"]) for it in enriched if str(it.get("item_id")) not in gt_set]
                    candidate_pools[uid] = (gt_items + non_gt_items)[:pool_k]
                else:
                    candidate_pools[uid] = [str(it["item_id"]) for it in enriched[:pool_k]]
                if not results[uid]:
                    logger.warning(
                        "Baseline %s produced empty top-k for user %s; using popularity fallback.",
                        baseline_type,
                        uid,
                    )
                    sorted_items = sorted(
                        enriched, key=lambda x: float(x.get("pop", 0) or 0), reverse=True
                    )
                    fallback = [
                        {"item_id": str(it["item_id"]), "score": float(it.get("pop", 0) or 0), "reason": "popularity_fallback"}
                        for it in sorted_items[:topk]
                    ]
                    results[uid] = fallback
                    candidate_pools[uid] = [str(it["item_id"]) for it in sorted_items[:pool_k]]
                    candidate_pools_post[uid] = candidate_pools[uid]
                results_pre[uid] = results[uid]
                candidate_pools_post[uid] = candidate_pools[uid]

            cand_set = set(str(x) for x in (candidate_pools.get(uid) or []))
            if gt_set and cand_set:
                inter = gt_set & cand_set
                if inter:
                    gt_hit_in_candidates += 1
                if len(inter) == len(gt_set):
                    gt_full_covered_in_candidates += 1
                gt_recall_frac_in_candidates.append(len(inter) / max(1, len(gt_set)))
            continue

        t0 = time.time()
        candidates, retrieval_meta = get_candidates_for_user(
            profile, enriched, index, id2idx, emb,
            graph=G,
            pool_size=config.get("candidate_pool_size", 1000),
            retrieval_mode=config.get("retrieval_mode", "ann"),
            hybrid_union_max=config.get("hybrid_union_max"), 
            preference_prior_mode=config.get("preference_prior_mode", "prior_plus_context"),
        )
        retrieval_times.append(time.time() - t0)
        
        if not candidates:
            logger.warning("No candidates for user %s; using fallback.", uid)
            candidates = [str(it["item_id"]) for it in enriched[:50]]
            retrieval_meta = retrieval_meta or {
                "retrieval_mode": config.get("retrieval_mode", "ann"),
                "ann_k": None,
                "bm25_k": None,
                "overlap": None,
            }
        

        if config.get("filter_already_seen") and viewed_ids:
            viewed_set = set(str(x) for x in viewed_ids)
            n_before = len(candidates)
            candidates = [c for c in candidates if c not in viewed_set]
            if n_before > len(candidates):
                logger.debug("User %s: filtered %d already-seen items from candidates.", uid, n_before - len(candidates))
        
        candidate_pools[uid] = candidates

        cand_set = set(str(x) for x in (candidates or []))
        if gt_set and cand_set:
            inter = gt_set & cand_set
            if inter:
                gt_hit_in_candidates += 1
            if len(inter) == len(gt_set):
                gt_full_covered_in_candidates += 1
            gt_recall_frac_in_candidates.append(len(inter) / max(1, len(gt_set)))
        if not retrieval_metas:
            retrieval_metas.append(retrieval_meta)
        
        reranker_scores = {}
        
        if reranker is not None:
            rerank_pool_size = config.get("rerank_pool_size", len(candidates))
            candidates_to_rerank = candidates[:rerank_pool_size]
            results_pre[uid] = [{"item_id": str(x)} for x in candidates_to_rerank[:topk]]
            
            t1 = time.time()
            reranked_full = reranker.rerank(profile, candidates_to_rerank, topk=len(candidates_to_rerank))

            # --- FIX: enforce deterministic descending-by-score order ---
            if reranked_full:
                reranked_full = sorted(
                    reranked_full,
                    key=lambda d: (
                        float(d.get("score")) if d.get("score") is not None else float("-inf"),
                        str(d.get("item_id", "")),
                    ),
                    reverse=True,
                )
            # --- end fix ---
            
            dt = time.time() - t1
            user_times.append(dt)
            
    
            debias_stats = getattr(reranker, "_last_debias_stats", {})
            if debias_stats:
                run_meta.setdefault("debias_stats", [])
                run_meta["debias_stats"].append({
                    "user_id": uid,
                    **debias_stats
                })
            
            # If you use diversification/pareto, score may be non-monotonic by design -> don't treat as ERROR.
            enforce_score_sortedness = not bool(config.get("diversify_config")) and not bool(config.get("two_head_config"))
            if enforce_score_sortedness and reranked_full and len(reranked_full) > 1:
                scores = [float(x.get("score")) for x in reranked_full if x.get("score") is not None]
                for i in range(len(scores) - 1):
                    if scores[i] + 1e-9 < scores[i + 1]:
                        logger.warning(
                            "Reranker results not sorted by score (sanity-check). User: %s, pos %d: %.6f < %.6f",
                            uid, i, scores[i], scores[i + 1]
                        )
                        break
                     
            
            for item in reranked_full:
                item_id = item.get("item_id")
                score = item.get("score")
                if item_id and score is not None:
                    reranker_scores[item_id] = float(score)

            for item in reranked_full:
                item_id = item.get("item_id")
                score = item.get("score")
                if item_id and score is not None:
                    stored_score = reranker_scores.get(str(item_id))
                    if stored_score is not None and abs(float(score) - float(stored_score)) > 1e-6:
                        logger.warning("ID-score mismatch after rerank! User: %s, Item: %s", uid, item_id)
            
            reranked = reranked_full[:topk]
            
         
            if reranked and len(reranked) > 1:
                scores = [item.get("score") for item in reranked if item.get("score") is not None]
                if len(scores) > 1:
                    for i in range(len(scores) - 1):
                        if scores[i] < scores[i+1]:
                            logger.error("Reranker results not sorted correctly after topk! User: %s, pos %d: %.4f < %.4f", 
                                       uid, i, scores[i], scores[i+1])

            if not reranked:
                logger.warning(
                    "Reranker produced empty top-k for user %s; applying popularity fallback.",
                    uid,
                )

                pop_source = []
                if candidates:
                
                    enriched_by_id = {str(it["item_id"]): it for it in enriched}
                    for cid in candidates:
                        meta = enriched_by_id.get(str(cid))
                        if meta is not None:
                            pop_source.append(meta)
                if not pop_source:
                    pop_source = enriched
                sorted_items = sorted(
                    pop_source, key=lambda x: float(x.get("pop", 0) or 0), reverse=True
                )
                reranked = [
                    {"item_id": str(it["item_id"]), "score": float(it.get("pop", 0) or 0), "reason": "popularity_fallback"}
                    for it in sorted_items[:topk]
                ]
            reranker_scores_all[uid] = reranker_scores
            reranked_ids = [str(x.get("item_id")) for x in (reranked_full or reranked) if x.get("item_id") is not None]
            candidate_pools_post[uid] = reranked_ids + candidates[rerank_pool_size:]
        else:
            reranked = candidates[:topk]
            results_pre[uid] = [{"item_id": str(x)} for x in candidates[:topk]]
            candidate_pools_post[uid] = list(candidates[:pool_k])

        norm = []
        if reranked is None:
            norm = []
        elif isinstance(reranked, list) and reranked:
            first = reranked[0]
            if isinstance(first, dict):
                for r in reranked:
                    mid = r.get("item_id") or r.get("id") or r.get("doc_id")
                    if mid is None:
                        continue
                    rec = {"item_id": str(mid)}
                    if "score" in r:
                        rec["score"] = r.get("score")
                    if "reason" in r:
                        rec["reason"] = r.get("reason")
                    norm.append(rec)
            elif isinstance(first, (str, int)):
                norm = [{"item_id": str(x)} for x in reranked]
        else:
            norm = [{"item_id": str(reranked)}] if reranked else []
        
        if reranker is not None and norm and len(norm) > 1:
            scores = [r.get("score") for r in norm if r.get("score") is not None]
            if len(scores) > 1:
                for i in range(len(scores) - 1):
                    if scores[i] < scores[i+1]:
                        logger.error("Final results not sorted correctly! User: %s, pos %d: %.4f < %.4f", 
                                   uid, i, scores[i], scores[i+1])
        
        if norm:
            ids = [r.get("item_id") for r in norm if r.get("item_id")]
            if len(ids) != len(set(ids)):
                logger.warning("Duplicate IDs found in final results! User: %s, total: %d, unique: %d", 
                             uid, len(ids), len(set(ids)))
        
        results[uid] = norm
    
    if user_times:
        user_times_arr = np.array(user_times)
        logger.info("Rerank time per user: mean=%.3fs std=%.3fs", float(user_times_arr.mean()), float(user_times_arr.std()))
    if retrieval_times:
        rt_arr = np.array(retrieval_times)
        logger.info("Retrieval time per user: mean=%.3fs std=%.3fs", float(rt_arr.mean()), float(rt_arr.std()))
    
    if user_times:
        rerank_times_meta = {
            "mean": float(np.mean(user_times)),
            "std": float(np.std(user_times)),
            "p50": float(np.percentile(user_times, 50)),
            "p95": float(np.percentile(user_times, 95)),
        }
    else:
        rerank_times_meta = None
    if retrieval_times:
        retrieval_times_meta = {
            "mean": float(np.mean(retrieval_times)),
            "std": float(np.std(retrieval_times)),
            "p50": float(np.percentile(retrieval_times, 50)),
            "p95": float(np.percentile(retrieval_times, 95)),
        }
    else:
        retrieval_times_meta = None


    total_time_per_user_meta = None
    if retrieval_times and user_times and len(retrieval_times) == len(user_times):
        total_per_user = np.array(retrieval_times) + np.array(user_times)
        total_time_per_user_meta = {
            "mean": float(np.mean(total_per_user)),
            "std": float(np.std(total_per_user)),
            "p50": float(np.percentile(total_per_user, 50)),
            "p95": float(np.percentile(total_per_user, 95)),
        }
        logger.info("Total time per user (retrieval+rerank): mean=%.3fs p50=%.3fs p95=%.3fs",
                    total_time_per_user_meta["mean"], total_time_per_user_meta["p50"], total_time_per_user_meta["p95"])
    elif retrieval_times and not user_times:
        total_per_user = np.array(retrieval_times)
        total_time_per_user_meta = {
            "mean": float(np.mean(total_per_user)),
            "std": float(np.std(total_per_user)),
            "p50": float(np.percentile(total_per_user, 50)),
            "p95": float(np.percentile(total_per_user, 95)),
        }
    

    pool_sizes = [len(candidate_pools[uid]) for uid in candidate_pools] if candidate_pools else []
    candidates_requested = config.get("candidate_pool_size", 1000)
    if config.get("baseline") is not None:
        retrieval_method = config["baseline"]  
        candidates_requested = catalog_size  
    else:
        retrieval_method = (retrieval_metas[0]["retrieval_mode"] if retrieval_metas else config.get("retrieval_mode", "ann"))
    

    from .rerank_diversify import _item_pop_rank_from_meta
    items_by_id = {str(it.get("item_id")): it for it in enriched}
    item_pop_rank = _item_pop_rank_from_meta(items_by_id, pop_key="pop") if items_by_id else {}
    item_pop_count = {str(it.get("item_id")): float(it.get("pop", 0) or 0) for it in enriched} if enriched else {}

    run_meta = {
        "split_seed": int(split_seed),
        "init_seed": int(init_seed),
        "catalog_size": catalog_size,
        "candidates_requested": candidates_requested,
        "candidates_after_filters": {
            "min": min(pool_sizes) if pool_sizes else None,
            "median": int(np.median(pool_sizes)) if pool_sizes else None,
            "max": max(pool_sizes) if pool_sizes else None,
        },
        "final_topk": topk,
        "retrieval_method": retrieval_method,
        "item_pop_rank": item_pop_rank,
        "item_pop_count": item_pop_count,
    }

    try:
        gt_sizes_arr = np.array(gt_sizes, dtype=np.int32) if gt_sizes else np.array([], dtype=np.int32)
        rec_arr = np.array(gt_recall_frac_in_candidates, dtype=np.float32) if gt_recall_frac_in_candidates else np.array([], dtype=np.float32)
        run_meta["leakage_sanity"] = {
            "n_users_eval": int(n_users),

            "share_users_gt_intersects_profile": float(leakage_gt_profile_overlap / max(1, n_users)),
            "gt_profile_overlap_examples": leakage_gt_profile_overlap_examples,

            "gt_size_mean": float(gt_sizes_arr.mean()) if gt_sizes_arr.size else None,
            "gt_size_p50": float(np.percentile(gt_sizes_arr, 50)) if gt_sizes_arr.size else None,
            "gt_size_p75": float(np.percentile(gt_sizes_arr, 75)) if gt_sizes_arr.size else None,
            "gt_size_p90": float(np.percentile(gt_sizes_arr, 90)) if gt_sizes_arr.size else None,
            "gt_size_p95": float(np.percentile(gt_sizes_arr, 95)) if gt_sizes_arr.size else None,
            "share_users_any_gt_in_candidates": float(gt_hit_in_candidates / max(1, n_users)),
            "share_users_gt_fully_covered_in_candidates": float(gt_full_covered_in_candidates / max(1, n_users)),
            "gt_recall_frac_in_candidates_mean": float(rec_arr.mean()) if rec_arr.size else None,
            "gt_recall_frac_in_candidates_p50": float(np.percentile(rec_arr, 50)) if rec_arr.size else None,
            "gt_recall_frac_in_candidates_p90": float(np.percentile(rec_arr, 90)) if rec_arr.size else None,
        }
    except Exception:
        pass
    if retrieval_metas and retrieval_metas[0].get("retrieval_mode") == "hybrid":
        run_meta["ann_k"] = retrieval_metas[0].get("ann_k")
        run_meta["bm25_k"] = retrieval_metas[0].get("bm25_k")
        run_meta["popularity_k"] = retrieval_metas[0].get("popularity_k")
        run_meta["overlap"] = retrieval_metas[0].get("overlap")
        run_meta["union_size"] = retrieval_metas[0].get("union_size")
    if total_time_per_user_meta is not None:
        run_meta["total_time_per_user"] = total_time_per_user_meta

    results_meta = {
        "results": results,           
        "results_pre": results_pre,   
        "candidate_pools": candidate_pools,  
        "candidate_pools_post": candidate_pools_post, 
        "rerank_times": rerank_times_meta,
        "retrieval_times": retrieval_times_meta,
        "reranker_scores": reranker_scores_all,
        "run_meta": run_meta,
    }
    
    return results_meta


def run_with_logging(
    run_id: str,
    n_users: int = None,
    seed: int = 42,
    split_seed: int = 42,
    init_seed: int = 42,
    config: dict = None,
    dataset: str = "serendipity",
) -> dict:
   
    results_meta = run_experiment(
        n_users=n_users,
        seed=seed,
        split_seed=split_seed,
        init_seed=init_seed,
        config=config,
        dataset=dataset,
    )
    
    if isinstance(results_meta, dict) and "results" in results_meta:
        results = results_meta["results"]
        results_pre = results_meta.get("results_pre") or {}
        candidate_pools = results_meta.get("candidate_pools", {})
        candidate_pools_post = results_meta.get("candidate_pools_post") or candidate_pools
        rerank_times = results_meta.get("rerank_times")
        retrieval_times = results_meta.get("retrieval_times")
        reranker_scores_all = results_meta.get("reranker_scores", {})
        run_meta = results_meta.get("run_meta")
    else:
        results = results_meta
        results_pre = {}
        candidate_pools = {}
        candidate_pools_post = {}
        rerank_times = None
        retrieval_times = None
        reranker_scores_all = {}
        run_meta = None
    
  
    from .evaluate_results import evaluate_single, load_gt_struct, load_split_metadata, evaluate_by_buckets_and_scenarios
    
    from .evaluation_config import get_eval_paths

    cfg = config or {}
    dataset_key = normalize_dataset_key(cfg.get("dataset") or dataset)
    paths = get_eval_paths(dataset_key)

    gt_path = Path(cfg.get("ground_truth_path") or paths["gt"])
    split_meta_path = Path(cfg.get("split_metadata_path") or paths["split_meta"])

    gt_wrapper = load_gt_struct(gt_path=gt_path)
    gt = gt_wrapper["data"]
    
    try:
        from .evaluation_config import N_BOOTSTRAP
    except ImportError:
        N_BOOTSTRAP = 1000

    rows, summary = evaluate_single(
        results, gt,
        k=config.get("topk", 10) if config else 10,
        n_bootstrap=N_BOOTSTRAP,
    )

    metrics = {
        "hr@10": {
            "mean": summary.get("hr_mean", 0.0),
            "std": summary.get("hr_std", 0.0),
            "ci95_low": summary.get("hr_ci_95_lower"),
            "ci95_high": summary.get("hr_ci_95_upper"),
            "n_users": summary.get("n_users", 0),
        },
        "ndcg@10": {
            "mean": summary.get("ndcg_mean", 0.0),
            "std": summary.get("ndcg_std", 0.0),
            "ci95_low": summary.get("ndcg_ci_95_lower"),
            "ci95_high": summary.get("ndcg_ci_95_upper"),
            "n_users": summary.get("n_users", 0),
        },
        "mrr@10": {
            "mean": summary.get("mrr_mean", 0.0),
            "std": summary.get("mrr_std", 0.0),
            "ci95_low": summary.get("mrr_ci_95_lower"),
            "ci95_high": summary.get("mrr_ci_95_upper"),
            "n_users": summary.get("n_users", 0),
        },
        "map@10": {
            "mean": summary.get("map_mean", 0.0),
            "std": summary.get("map_std", 0.0),
            "ci95_low": summary.get("map_ci_95_lower"),
            "ci95_high": summary.get("map_ci_95_upper"),
            "n_users": summary.get("n_users", 0),
        },
    }


    metrics_pre = None
    if results_pre:
        _, summary_pre = evaluate_single(
            results_pre, gt,
            k=config.get("topk", 10) if config else 10,
            n_bootstrap=N_BOOTSTRAP,
        )
        metrics_pre = {
            "hr@10": {
                "mean": summary_pre.get("hr_mean", 0.0),
                "std": summary_pre.get("hr_std", 0.0),
                "ci95_low": summary_pre.get("hr_ci_95_lower"),
                "ci95_high": summary_pre.get("hr_ci_95_upper"),
                "n_users": summary_pre.get("n_users", 0),
            },
            "ndcg@10": {
                "mean": summary_pre.get("ndcg_mean", 0.0),
                "std": summary_pre.get("ndcg_std", 0.0),
                "ci95_low": summary_pre.get("ndcg_ci_95_lower"),
                "ci95_high": summary_pre.get("ndcg_ci_95_upper"),
                "n_users": summary_pre.get("n_users", 0),
            },
            "mrr@10": {
                "mean": summary_pre.get("mrr_mean", 0.0),
                "std": summary_pre.get("mrr_std", 0.0),
                "ci95_low": summary_pre.get("mrr_ci_95_lower"),
                "ci95_high": summary_pre.get("mrr_ci_95_upper"),
                "n_users": summary_pre.get("n_users", 0),
            },
            "map@10": {
                "mean": summary_pre.get("map_mean", 0.0),
                "std": summary_pre.get("map_std", 0.0),
                "ci95_low": summary_pre.get("map_ci_95_lower"),
                "ci95_high": summary_pre.get("map_ci_95_upper"),
                "n_users": summary_pre.get("n_users", 0),
            },
        }

    split_meta = load_split_metadata(path=split_meta_path)
    if split_meta:
        bucket_scenario = evaluate_by_buckets_and_scenarios(
            results, gt, split_meta, k=config.get("topk", 10) if config else 10
        )
        metrics["by_bucket"] = bucket_scenario.get("by_bucket", {})
        metrics["by_scenario"] = bucket_scenario.get("by_scenario", {})
    
    from .run_logger import compute_diagnostics
    diagnostics_post = compute_diagnostics(
        results, gt, candidate_pools_post, rerank_times, retrieval_times, run_meta
    )
    diagnostics_pre = None
    if results_pre:
        diagnostics_pre = compute_diagnostics(
            results_pre, gt, candidate_pools, None, retrieval_times, run_meta
        )

    from .utils import logger
    
    sanity_failures = []
    

    retrieval_mode = config.get("retrieval_mode", "ann") if config else "ann"
    if retrieval_mode == "hybrid" and results:
        top1_items = []
        for uid, recs in results.items():
            if recs and len(recs) > 0:
                top1_items.append(str(recs[0].get("item_id", "")))
        unique_top1 = len(set(top1_items))
        total_users = len(top1_items)
        if total_users > 0:
            share_unique = unique_top1 / total_users
            # With reranker we expect diverse top-1; without reranker (candidates only)
            # allow lower diversity (e.g. cold users with similar profiles).
            threshold = 0.15 if not config.get("use_reranker", True) else 0.5
            if share_unique < threshold:
                msg = (
                    f"SANITY CHECK WARNING: Hybrid retrieval produces identical recommendations. "
                    f"Only {unique_top1}/{total_users} unique top-1 items (share={share_unique:.2%}). "
                    f"Threshold={threshold:.0%}. This suggests retrieval collapse or bug."
                )
                logger.warning(msg)
                sanity_failures.append(msg)
    
    if metrics_pre and metrics and config and config.get("use_reranker", True):
        reranker_degradation_threshold = 0.10  
        for metric_name in ["ndcg@10", "hr@10"]:
            pre_mean = metrics_pre.get(metric_name, {}).get("mean", 0.0)
            post_mean = metrics.get(metric_name, {}).get("mean", 0.0)
            if pre_mean > 0:
                relative_change = (post_mean - pre_mean) / pre_mean
                if relative_change < -reranker_degradation_threshold:
                    msg = (
                        f"SANITY CHECK WARNING: Reranker degrades {metric_name} by {abs(relative_change):.1%} "
                        f"(pre={pre_mean:.4f}, post={post_mean:.4f}). "
                        f"Degradation exceeds threshold of {reranker_degradation_threshold:.0%}. "
                        f"This suggests reranker bug or misconfiguration."
                    )
                    # Log as warning and continue; treat reranker as ablation.
                    logger.warning(msg)
                    sanity_failures.append(msg)
    

    if candidate_pools and gt:
        n_users_with_gt_in_candidates = 0
        total_users_checked = 0
        for uid, candidates in candidate_pools.items():
            if uid in gt and gt[uid]:
                total_users_checked += 1
                gt_set = set(str(x) for x in gt[uid])
                cand_set = set(str(x) for x in candidates)
                if gt_set & cand_set:  
                    n_users_with_gt_in_candidates += 1
        
        if total_users_checked > 0:
            share_with_gt = n_users_with_gt_in_candidates / total_users_checked

            if share_with_gt < 0.10:
                msg = (
                    f"SANITY CHECK WARNING: Very few users have GT items in candidates. "
                    f"Only {n_users_with_gt_in_candidates}/{total_users_checked} users ({share_with_gt:.1%}) "
                    f"have any GT item in their candidate pool. "
                    f"This suggests retrieval failure or GT/candidate mismatch."
                )
                logger.warning(msg)
                sanity_failures.append(msg)
    

    if sanity_failures:
        warning_summary = "\n".join([f"  [{i+1}] {msg}" for i, msg in enumerate(sanity_failures)])
        logger.warning(
            f"EXPERIMENT SANITY CHECKS WARNINGS ({len(sanity_failures)} warning(s)):\n{warning_summary}\n"
            f"Run ID: {run_id}\n"
            f"Continuing execution despite warnings. Please investigate these issues."
        )

    results_dir = Path(config.get("results_dir")) if config and config.get("results_dir") else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    per_user_metrics_path = (
        Path(config["per_user_metrics_path"])
        if config and config.get("per_user_metrics_path")
        else Path("experiments") / f"{run_id}_per_user_metrics.csv"
    )
    per_user_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    per_user_legacy_path = per_user_metrics_path.with_name(
        f"{run_id}_per_user.csv"
    )


    runs_log_path = config.get("runs_log_path") if config else None
    log_run(
        run_id=run_id,
        config={
            "seed": seed,
            "split_seed": int(split_seed),
            "init_seed": int(init_seed),
            "n_users": n_users,
            "topk": config.get("topk", 10) if config else 10,
            "baseline": config.get("baseline") if config else None,
            "use_reranker": config.get("use_reranker", True) if config else True,
            "reranker_model": config.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2") if config else "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "reranker_ablation": config.get("reranker_ablation") if config else None,
            "candidate_pool_size": config.get("candidate_pool_size", 1000) if config else 1000,
            "retrieval_mode": config.get("retrieval_mode", "ann") if config else "ann",
            "dataset": dataset,
            "two_head_config": config.get("two_head_config") if config else None,
            "profile_type": config.get("profile_type") if config else None,
            "preference_prior_mode": config.get("preference_prior_mode") if config else None,
            "diversify_config": config.get("diversify_config") if config else None,
            "robustness": config.get("robustness") if config else None,
            "train_interactions_path": config.get("train_interactions_path") if config else None,
            "ground_truth_path": str(gt_path) if gt_path else None,
            "split_metadata_path": str(split_meta_path) if split_meta_path else None,
        },
        metrics=metrics,
        metrics_pre=metrics_pre,
        diagnostics=diagnostics_post,
        diagnostics_pre=diagnostics_pre,
        results=results,
        gt=gt,
        candidate_pools=candidate_pools_post,
        rerank_times=rerank_times,
        retrieval_times=retrieval_times,
        run_meta=run_meta,
        files={
            "raw_results": str((results_dir / f"{run_id}.json").as_posix()),
            "per_user_metrics_csv": str(per_user_metrics_path.as_posix()),
        },
        runs_log_path=Path(runs_log_path) if runs_log_path else None,
    )
    
  
    save_pool_k = int(config.get("candidate_pool_size", 1000) if config else 1000)
    save_pool_k = max(save_pool_k, 1000, int(config.get("topk", 10) if config else 10))

    compact_candidate_pools = {
        uid: (pool[:save_pool_k] if isinstance(pool, list) else pool)
        for uid, pool in (candidate_pools or {}).items()
    }

    compact_candidate_pools_post = {
        uid: (pool[:save_pool_k] if isinstance(pool, list) else pool)
        for uid, pool in (candidate_pools_post or {}).items()
    }

    results_to_save = {
        "results": results,
        "results_pre": results_pre,
        "candidate_pools": compact_candidate_pools,
        "candidate_pools_post": compact_candidate_pools_post,
        "rerank_times": rerank_times,
        "reranker_scores": reranker_scores_all,
        "ranked_list_contract": {
            "results": "final (post-rerank) top-k list used for HR/nDCG/MRR",
            "results_pre": "pre-rerank (retrieval-only) top-k list (if reranker used)",
            "candidate_pools": "pre-rerank ranked pool used for recall@K",
            "candidate_pools_post": "post-rerank ranked pool (reranked subset + rest), used for recall@K_post",
        },
    }

    save_json(results_dir / f"{run_id}.json", results_to_save)
    

   
    import csv
    csv_path = per_user_metrics_path
    csv_legacy_path = per_user_legacy_path
    if rows:
        keys = list(rows[0].keys())
        for out_path in (csv_path, csv_legacy_path):
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)
        logger.info("Saved per-user metrics CSV: %s", csv_path)
    
    return results



def small_demo_run(n_users=50, seed=42):
    """Deprecated: use run_experiment instead."""
    logger.warning("small_demo_run is deprecated. Use run_experiment() instead.")
    return run_experiment(n_users=n_users, seed=seed)


if __name__ == "__main__":

    run_with_logging(
        run_id="test_full_pipeline",
        n_users=50,
        seed=42,
        config={
            "baseline": None,
            "use_reranker": True,
            "topk": 10
        }
    )
