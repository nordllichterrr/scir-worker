
import os
from typing import Dict, List, Any, Optional
from sentence_transformers import CrossEncoder
import numpy as np
from pathlib import Path
import torch
from .utils import logger
from .rerank_diversify import diversify, _item_pop_rank_from_meta
from .rerank_two_head import novelty_from_pop_rank, combine_relevance_novelty, select_pareto_balanced
from .vark_simulator import preference_match_score


MAX_QUERY_CHARS = 300  
MAX_DOC_CHARS = 200  
QUERY_TEMPLATE = "User likes: {}"
DOC_TEMPLATE = "Recommend similar item: {}"


def format_reranker_query(user_text: str) -> str:
    if not user_text or not str(user_text).strip():
        return "User likes: (no preference)"
    s = str(user_text).strip()
    if len(s) > MAX_QUERY_CHARS:
        s = s[: MAX_QUERY_CHARS].rsplit(" ", 1)[0] + "…"
    return QUERY_TEMPLATE.format(s)


def format_reranker_doc(item_text: str) -> str:
   
    text = (item_text or "").strip() or "(no description)"
    if len(text) > MAX_DOC_CHARS:
        text = text[:MAX_DOC_CHARS].rsplit(" ", 1)[0] + "…"
    return DOC_TEMPLATE.format(text)


class CrossReranker:
 
    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        items_meta=None,
        device="cpu",
        item_embeddings: Optional[np.ndarray] = None,
        id2idx: Optional[Dict[str, int]] = None,
        diversify_config: Optional[Dict[str, Any]] = None,
        exposure_map: Optional[Dict[str, float]] = None,
        two_head_config: Optional[Dict[str, Any]] = None,
        use_fp16: bool = False,
        max_length: Optional[int] = None,
        two_stage_config: Optional[Dict[str, Any]] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.item_embeddings = item_embeddings
        self.id2idx = id2idx or {}
        self.diversify_config = diversify_config or {}
        self.exposure_map = exposure_map
        self.two_head_config = two_head_config or {}
        self.use_fp16 = use_fp16 and device != "cpu" 
        self.max_length = max_length  
        self.two_stage_config = two_stage_config  
        logger.info("Loading CrossEncoder: %s (device=%s, fp16=%s, max_length=%s)", 
                   model_name, device, self.use_fp16, self.max_length)

    
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


        try:
            self.model = CrossEncoder(model_name, device=device, max_length=self.max_length)
   
            if self.use_fp16 and device != "cpu":
                try:
                    if hasattr(self.model, "model") and hasattr(self.model.model, "half"):
                        self.model.model = self.model.model.half()
                        logger.info("Enabled FP16 for CrossEncoder")
                except Exception as e:
                    logger.warning("Failed to enable FP16: %s", e)
        except Exception as e:
            logger.error("Failed to load CrossEncoder model %s: %s", model_name, e)
            logger.error("Reranker will be disabled for this run (falling back to identity ranking).")
            self.model = None

        self.fast_model = None
        if self.two_stage_config and self.two_stage_config.get("fast_model"):
            try:
                fast_model_name = self.two_stage_config["fast_model"]
                logger.info("Loading fast reranker: %s", fast_model_name)
                self.fast_model = CrossEncoder(fast_model_name, device=device, max_length=self.max_length)
                if self.use_fp16 and device != "cpu":
                    try:
                        if hasattr(self.fast_model, "model") and hasattr(self.fast_model.model, "half"):
                            self.fast_model.model = self.fast_model.model.half()
                    except Exception:
                        pass
            except Exception as e:
                logger.warning("Failed to load fast reranker: %s", e)

        self.items: Dict[str, Dict[str, Any]] = {}
        for it in (items_meta or []):
            self.items[str(it.get("item_id"))] = it
    
        self._item_popularity = {iid: float(meta.get("pop", 0) or 0) for iid, meta in self.items.items()}
        self._item_pop_rank = _item_pop_rank_from_meta(self.items, pop_key="pop") if self.items else {}

    def _make_text_for_item(self, meta):
        title = meta.get("title","")
        desc = meta.get("description","") or meta.get("text","") or ""
        tags = meta.get("format_tags") or meta.get("genres") or ""
        if isinstance(tags, (list,tuple)):
            tags = ", ".join(tags)
        s = f"Title: {title}. Description: {desc}. Tags: {tags}"
        return s.strip()

    def rerank(self, user_profile: dict, candidate_ids: list, topk=10, batch_size=256):

        raw_query = user_profile.get("text_profile") or user_profile.get("goal") or ""
        # For cold-start users, ensure query is not empty and includes user_id for diversity
        if not raw_query or not raw_query.strip() or raw_query.strip().startswith("User profile (cold-start"):
            user_id = user_profile.get("user_id", "")
            vark = user_profile.get("vark", "visual")
            raw_query = f"user_id={user_id}, vark={vark}, exploring diverse recommendations"
        query = format_reranker_query(raw_query)
        

        if self.two_stage_config and self.fast_model and len(candidate_ids) > self.two_stage_config.get("fast_topk", 200):
            fast_topk = self.two_stage_config.get("fast_topk", 200)
            logger.debug("Two-stage rerank: fast model on %d candidates → top-%d", len(candidate_ids), fast_topk)
            
          
            pairs_fast = []
            ids_fast = []
            for cid in candidate_ids:
                meta = self.items.get(str(cid), {})
                item_text = self._make_text_for_item(meta)
                doc = format_reranker_doc(item_text)
                pairs_fast.append([query, doc])
                ids_fast.append(str(cid))
            
            with torch.no_grad():
                scores_fast = self.fast_model.predict(pairs_fast, show_progress_bar=False, batch_size=batch_size)
   
            scored_fast = list(zip(ids_fast, scores_fast))
            scored_fast.sort(key=lambda x: x[1], reverse=True)
            candidate_ids = [cid for cid, _ in scored_fast[:fast_topk]]
            logger.debug("Fast model selected %d candidates", len(candidate_ids))
        
        pairs = []
        ids = []
        for cid in candidate_ids:
            meta = self.items.get(str(cid), {})
            item_text = self._make_text_for_item(meta)
            doc = format_reranker_doc(item_text)
            pairs.append([query, doc])
            ids.append(str(cid))
        
    
        if len(pairs) == 0 or self.model is None:
            return []
        

        with torch.no_grad():
            scores = self.model.predict(pairs, show_progress_bar=False, batch_size=batch_size)
        scored = list(zip(ids, scores))

        scored_with_original_rank = [(id, score, orig_idx) for orig_idx, (id, score) in enumerate(scored)]
        prior_mode = (user_profile.get("preference_prior_mode") or "prior_plus_context").strip().lower()
        if prior_mode != "no_prior":
            boosted = []
            for cid, score, orig_idx in scored_with_original_rank:
                meta = self.items.get(str(cid), {})
                pref_bonus = preference_match_score(
                    user_profile,
                    meta,
                    use_context=(prior_mode == "prior_plus_context"),
                )
                boosted.append((cid, float(score) + 0.15 * float(pref_bonus), orig_idx))
            scored_with_original_rank = boosted
        scored_with_original_rank.sort(key=lambda x: (x[1], -x[2]), reverse=True) 
        scored = [(id, score) for id, score, _ in scored_with_original_rank]
        out = []
        for cid, s in scored:
            out.append({"item_id": cid, "score": float(s), "reason": "CrossEncoder rerank"})


        th = self.two_head_config
        if th and (th.get("alpha") is not None or th.get("mode") == "pareto_balanced"):
            catalog_size = th.get("catalog_size") or len(self.items)
            item_novelty = novelty_from_pop_rank(self._item_pop_rank, catalog_size=catalog_size)
            if th.get("mode") == "pareto_balanced":
                out = combine_relevance_novelty(out, item_novelty, alpha=0.5)
                out = select_pareto_balanced(out, topk, relevance_key="relevance", novelty_key="novelty")
            else:
                alpha = float(th.get("alpha", 0.5))
                out = combine_relevance_novelty(out, item_novelty, alpha=alpha)

            if not self.diversify_config or not any([
                self.diversify_config.get("popularity_penalty_alpha", 0) > 0,
                self.diversify_config.get("exposure_beta", 0) > 0,
                self.diversify_config.get("mmr_lambda", 0) > 0,
                self.diversify_config.get("xquad_lambda", 0) > 0,
                self.diversify_config.get("fairness"),
            ]):
                return out[:topk] if th.get("mode") != "pareto_balanced" else out



        cfg = self.diversify_config
        debias_stats = {}
        if cfg and any([
            cfg.get("popularity_penalty_alpha", 0) > 0,
            cfg.get("exposure_beta", 0) > 0,
            cfg.get("mmr_lambda", 0) > 0,
            cfg.get("xquad_lambda", 0) > 0,
            cfg.get("fairness"),
        ]):
            out, debias_stats = diversify(
                out,
                topk,
                popularity_penalty_alpha=float(cfg.get("popularity_penalty_alpha", 0)),
                exposure_beta=float(cfg.get("exposure_beta", 0)),
                item_popularity=self._item_popularity if cfg.get("popularity_penalty_alpha") else None,
                exposure_map=self.exposure_map if cfg.get("exposure_beta") else None,
                mmr_lambda=float(cfg.get("mmr_lambda", 0)),
                item_embeddings=self.item_embeddings,
                id2idx=self.id2idx if self.id2idx else None,
                xquad_lambda=float(cfg.get("xquad_lambda", 0)),
                items_meta=self.items if cfg.get("xquad_lambda") else None,
                category_key=str(cfg.get("category_key", "genres")),
                fairness=cfg.get("fairness"),
                item_pop_rank=self._item_pop_rank if cfg.get("fairness") else None,
                log_stats=True,
            )

            self._last_debias_stats = debias_stats
        else:
            out = out[:topk]
            self._last_debias_stats = {}
        return out
