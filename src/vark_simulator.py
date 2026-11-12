
from typing import Dict, List, Any, Optional
from .utils import logger

import hashlib
import random
import re
from collections import Counter

_TAG_VOCAB = None

def _stable_seed(s: str) -> int:
    s = s or ""
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)

def _get_tag_vocab(items_meta, max_vocab: int = 2000):
    """
    Собирает частотный словарь жанров/тегов из каталога.
    Кэшируется один раз на запуск, чтобы не сканировать каталог для каждого юзера.
    """
    global _TAG_VOCAB
    if _TAG_VOCAB is not None:
        return _TAG_VOCAB

    c = Counter()
    iters = items_meta if isinstance(items_meta, list) else list((items_meta or {}).values())

    for it in iters:
        for key in ("genres", "format_tags", "tags", "category", "categories"):
            tags = it.get(key)
            if not tags:
                continue

            if isinstance(tags, str):
                parts = [p.strip() for p in re.split(r"[|,/;]+", tags) if p.strip()]
            elif isinstance(tags, (list, tuple, set)):
                parts = [str(p).strip() for p in tags if str(p).strip()]
            else:
                parts = [str(tags).strip()]

            for p in parts:
                if p:
                    c[p.lower()] += 1

    _TAG_VOCAB = [t for t, _ in c.most_common(max_vocab)]
    return _TAG_VOCAB


VARK_TO_DESCRIPTION = {
    "visual": "User prefers visual learning; interested in diagrams, images, and visual content.",
    "auditory": "User prefers auditory learning; interested in podcasts, lectures, and spoken content.",
    "reading": "User prefers reading and text; interested in articles, books, and written material.",
    "kinesthetic": "User prefers learning by doing; interested in hands-on and interactive content.",
}

VARK_QUESTIONS = [

    "I prefer to learn with diagrams or images.",
    "I remember best when I hear information.",
    "I prefer reading instructions.",
    "I learn best by doing / practicing.",

] * 4  

def score_vark_from_answers(answers):

    from collections import Counter
    c = Counter(answers)
    if not c:
        return "visual"
    dom = c.most_common(1)[0][0]
    return dom

def simulate_vark_quiz_responses(seed=None):
    """
    Per-user pseudo-random V/A/R/K responses. Uses a local RNG seeded from
    `seed` (typically user_id) so outputs differ across users and do not
    mutate the global `random` module (which would break other baselines).
    """
    seed_key = f"vark_quiz:{seed!s}"
    rng = random.Random(_stable_seed(seed_key))
    choices = ['visual','auditory','reading','kinesthetic']
    answers = [rng.choice(choices) for _ in range(16)]
    dom = score_vark_from_answers(answers)
    c = Counter(answers)
    total = max(1, len(answers))
    scores = {
        "V": float(c.get("visual", 0) / total),
        "A": float(c.get("auditory", 0) / total),
        "R": float(c.get("reading", 0) / total),
        "K": float(c.get("kinesthetic", 0) / total),
    }
    return {"answers": answers, "dominant": dom, "scores": scores}

def estimate_session_context(time_of_day: str = None, session_length_min: int = 5, device: str = "desktop"):

    state = {"status": "neutral", "max_complexity": 4}
    tod = (time_of_day or "").lower()
    

    if session_length_min is None:
        session_length_min = 5 
    
    if session_length_min > 30 and tod in ["evening","night","23:00","22:00"]:
        state["status"] = "tired"
        state["max_complexity"] = 2
    elif session_length_min < 10:
        state["status"] = "fresh"
        state["max_complexity"] = 5
    else:
        state["status"] = "neutral"
        state["max_complexity"] = 4
    return state


def get_item_preference_features(meta: Dict) -> Dict[str, float]:
    title = str(meta.get("title", "") or "").lower()
    desc = str(meta.get("description", "") or meta.get("text", "") or "").lower()
    genres = str(meta.get("genres", "") or "").lower()
    tags = meta.get("format_tags") or meta.get("tags") or ""
    if isinstance(tags, (list, tuple, set)):
        tags = " ".join(str(x).lower() for x in tags)
    else:
        tags = str(tags).lower()
    blob = " ".join([title, desc, genres, tags])

    def has_any(words):
        return 1.0 if any(w in blob for w in words) else 0.0

    text_density = min(1.0, len(desc) / 600.0) if desc else 0.2
    visual_density = has_any(["video", "visual", "diagram", "image", "plot", "chart"])
    has_audio = has_any(["audio", "podcast", "speech", "lecture"])
    has_video = has_any(["video", "youtube", "screencast", "visual"])
    interactivity_level = has_any(["interactive", "exercise", "quiz", "lab", "practice"])
    hands_on_score = has_any(["hands-on", "project", "lab", "build", "practice"])
    difficulty = has_any(["advanced", "expert", "hard"]) * 1.0 + has_any(["beginner", "intro"]) * 0.2
    prerequisites = has_any(["prerequisite", "requires", "before you start"])
    return {
        "has_video": has_video,
        "has_audio": has_audio,
        "text_density": float(text_density),
        "interactivity_level": interactivity_level,
        "hands_on_score": hands_on_score,
        "visual_density": visual_density,
        "difficulty": float(min(1.0, difficulty if difficulty > 0 else 0.4)),
        "prerequisites": prerequisites,
    }


def preference_match_score(profile: Dict, item_meta: Dict, use_context: bool = True) -> float:
    pref = profile.get("preference_prior") or {}
    vark_scores = pref.get("vark_scores") or {"V": 0.25, "A": 0.25, "R": 0.25, "K": 0.25}
    f = get_item_preference_features(item_meta or {})
    # Map VARK components to item modality features.
    score = (
        float(vark_scores.get("V", 0.0)) * (0.6 * f["visual_density"] + 0.4 * f["has_video"])
        + float(vark_scores.get("A", 0.0)) * f["has_audio"]
        + float(vark_scores.get("R", 0.0)) * f["text_density"]
        + float(vark_scores.get("K", 0.0)) * (0.7 * f["interactivity_level"] + 0.3 * f["hands_on_score"])
    )
    if use_context:
        ctx = profile.get("session_context_features") or {}
        fatigue = float(ctx.get("fatigue_level", 0.3))
        # Penalize high-difficulty items in high-fatigue sessions.
        score -= 0.15 * fatigue * f["difficulty"]
    return float(score)

def build_text_profile_from_viewed_items(
    viewed_item_ids: List[str],
    items_meta: Any,
    max_items: int = 5,
    max_chars: int = 600,
) -> str:

    if not viewed_item_ids or not items_meta:
        return ""
    
    by_id = {}
    if isinstance(items_meta, list):
        for it in items_meta:
            by_id[str(it.get("item_id"))] = it
    else:
        by_id = {str(k): v for k, v in (items_meta or {}).items()}
    parts = []
    total_len = 0
    for iid in viewed_item_ids[:max_items]:
        if total_len >= max_chars:
            break
        it = by_id.get(str(iid))
        if not it:
            continue
        title = str(it.get("title", "")).strip()
        desc = (it.get("description") or it.get("text", "") or "")
        if isinstance(desc, list):
            desc = " ".join(str(x) for x in desc)
        desc = str(desc).strip()[:150]
        tags = it.get("format_tags") or it.get("genres", "")
        if isinstance(tags, (list, tuple)):
            tags = ", ".join(str(x) for x in tags)
        tags = str(tags).strip()[:80]
        s = title
        if desc:
            s += ". " + desc
        if tags:
            s += " [" + tags + "]"
        if s:
            parts.append(s)
            total_len += len(s) + 2
    if not parts:
        return ""
    return "Recently viewed interests: " + " | ".join(parts)


def build_user_profile_from_minimal(info: Dict, prior_mode: str = "prior_plus_context"):

    user_id = info.get("user_id", "")
    v = simulate_vark_quiz_responses(seed=user_id)
    state = estimate_session_context(
        time_of_day=info.get("time_of_day"),
        session_length_min=info.get("session_len", 5),
        device=info.get("device", "desktop"),
    )
    goal = info.get("goal", "")
    vark = v["dominant"]

    text_profile = info.get("text_profile", goal)
    if (text_profile and text_profile.strip()):
        pass  
    else:

        viewed_ids = info.get("viewed_item_ids") or []
        items_meta = info.get("items_meta")
        if viewed_ids and items_meta:
            text_profile = build_text_profile_from_viewed_items(
                viewed_ids, items_meta, max_items=5, max_chars=600
            )
  
        if not text_profile or not text_profile.strip():
            desc = VARK_TO_DESCRIPTION.get(vark, "User exploring diverse recommendations.")

            # Deterministic "onboarding" interests from catalog tags (NO GT leakage).
            sampled = []
            if items_meta:
                vocab = _get_tag_vocab(items_meta)
                if vocab:
                    rng = random.Random(_stable_seed(str(user_id)))
                    k = min(3, len(vocab))
                    sampled = rng.sample(vocab, k=k)

            if sampled:
                text_profile = f"Cold-start profile: {desc} Stated interests: {', '.join(sampled)}."
            else:
                text_profile = f"Cold-start profile: {desc}"

    prior_mode = (prior_mode or "prior_plus_context").strip().lower()
    if prior_mode not in {"no_prior", "prior_only", "prior_plus_context"}:
        prior_mode = "prior_plus_context"
    if prior_mode == "no_prior":
        preference_prior = {"vark_scores": {"V": 0.25, "A": 0.25, "R": 0.25, "K": 0.25}}
    else:
        preference_prior = {"vark_scores": v.get("scores", {"V": 0.25, "A": 0.25, "R": 0.25, "K": 0.25})}

    session_context_features = {
        "time_of_day": info.get("time_of_day"),
        "session_length_min": info.get("session_len", 5),
        "device_type": info.get("device", "desktop"),
        "fatigue_level": 0.8 if state["status"] == "tired" else (0.2 if state["status"] == "fresh" else 0.4),
    }
    if prior_mode == "prior_only":
        session_context_features = {"time_of_day": None, "session_length_min": None, "device_type": None, "fatigue_level": 0.4}
    if prior_mode == "no_prior":
        session_context_features = {"time_of_day": None, "session_length_min": None, "device_type": None, "fatigue_level": 0.4}

    profile = {
        "user_id": user_id,
        "goal": goal or "learn about topic X",
        "vark": vark,
        "vark_scores": v.get("scores", {"V": 0.25, "A": 0.25, "R": 0.25, "K": 0.25}),
        "vark_answers": v["answers"],
        "max_complexity": state["max_complexity"],
        "time_of_day": session_context_features["time_of_day"],
        "preference_prior_mode": prior_mode,
        "preference_prior": preference_prior,
        "session_context_features": session_context_features,
        "text_profile": text_profile,
    }
    return profile
