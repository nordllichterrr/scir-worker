
import hashlib
import json
import logging
from pathlib import Path
import random
from sentence_transformers import SentenceTransformer
import numpy as np

from .config import EMBED_MODEL
 

logger = logging.getLogger("coldstart")
if not logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
    logger.setLevel(logging.INFO)

def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def per_user_seed(global_seed: int, uid) -> int:
    """
    Deterministic per-user sub-seed (reproducible across runs/machines).
    Use for random baselines so each user gets a different sample while the
    experiment remains reproducible for a fixed global_seed.
    """
    uid_s = str(uid) if uid is not None else ""
    h = hashlib.md5(f"{int(global_seed)}:{uid_s}".encode("utf-8")).hexdigest()
    # Keep in positive 31-bit range for Random / legacy consumers
    return int(h[:8], 16) & 0x7FFFFFFF


_EMBEDDERS = {}  # model_name -> SentenceTransformer

def get_embedder(model_name: str = None):
    """
    Loads SentenceTransformer once per model name.
    This makes ablations/config sweeps reproducible and correct.
    """
    model_name = model_name or EMBED_MODEL
    model_name_norm = model_name if isinstance(model_name, str) else str(model_name)

    if model_name_norm in _EMBEDDERS:
        return _EMBEDDERS[model_name_norm]

    logger.info("Loading SentenceTransformer model '%s' (cached per name)...", model_name_norm)
    emb = SentenceTransformer(model_name_norm)
    _EMBEDDERS[model_name_norm] = emb
    return emb

def clear_embedder_cache(model_name: str = None):
    """Optional: free RAM if you loaded multiple models."""
    if model_name is None:
        _EMBEDDERS.clear()
    else:
        _EMBEDDERS.pop(model_name, None)