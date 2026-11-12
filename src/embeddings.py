import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .config import EMBED_MODEL, EMBEDDINGS_NPY, EMBEDDING_MAP
from .utils import logger, get_embedder as _shared_get_embedder


CACHE_SCHEMA = 2


def get_embedder(model_name: str = EMBED_MODEL):
    return _shared_get_embedder(model_name)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _atomic_npy(path: Path, array: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, array, allow_pickle=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


@contextmanager
def _cache_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _versioned_path(path: Path, fingerprint: str) -> Path:
    return path.with_name(f"{path.stem}.{fingerprint}{path.suffix}")


def _prepare_items(items):
    ids = []
    texts = []

    for item in items:
        ids.append(str(item.get("item_id")))
        text = item.get("description") or item.get("title") or item.get("text") or ""
        texts.append(str(text))

    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate item_id values are not supported in embeddings.")

    if any(not item_id or item_id == "None" for item_id in ids):
        raise ValueError("Empty or missing item_id encountered.")

    return ids, texts


def _fingerprint(ids, texts, model_name):
    payload = {
        "schema": CACHE_SCHEMA,
        "model_name": str(model_name),
        "item_ids": ids,
        "texts": texts,
    }
    raw = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _validate_cache(emb_path, map_path, meta_path, fingerprint, ids):
    if not all(p.is_file() for p in (emb_path, map_path, meta_path)):
        return None

    try:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        if meta.get("schema") != CACHE_SCHEMA:
            return None
        if meta.get("fingerprint") != fingerprint:
            return None
        if meta.get("embeddings_sha256") != _sha256_file(emb_path):
            return None
        if meta.get("map_sha256") != _sha256_file(map_path):
            return None

        emb = np.load(emb_path, allow_pickle=False)
        with map_path.open("r", encoding="utf-8") as f:
            id2idx = json.load(f)

        expected_map = {item_id: idx for idx, item_id in enumerate(ids)}

        if emb.ndim != 2 or emb.shape[0] != len(ids):
            return None
        if not np.issubdtype(emb.dtype, np.number):
            return None
        if not np.isfinite(emb).all():
            return None
        if id2idx != expected_map:
            return None
        if list(emb.shape) != meta.get("shape"):
            return None
        if str(emb.dtype) != meta.get("dtype"):
            return None

        return emb, id2idx
    except Exception as exc:
        logger.warning("Embeddings cache validation failed: %s", exc)
        return None


def build_embeddings(
    items: list,
    model_name=EMBED_MODEL,
    out_npy=EMBEDDINGS_NPY,
    map_path=EMBEDDING_MAP,
):
    out_npy = Path(out_npy)
    map_path = Path(map_path)

    ids, texts = _prepare_items(items)
    fingerprint = _fingerprint(ids, texts, model_name)

    versioned_emb = _versioned_path(out_npy, fingerprint)
    versioned_map = _versioned_path(map_path, fingerprint)
    meta_path = versioned_emb.with_suffix(".meta.json")
    lock_path = versioned_emb.with_suffix(".lock")

    with _cache_lock(lock_path):
        cached = _validate_cache(
            versioned_emb, versioned_map, meta_path, fingerprint, ids
        )
        if cached is not None:
            logger.info("Validated embeddings cache loaded: %s", versioned_emb)
            return cached

        model = get_embedder(model_name)
        logger.info(
            "Generating embeddings for %d items with %s", len(texts), model_name
        )
        emb = np.asarray(
            model.encode(
                texts, show_progress_bar=True, convert_to_numpy=True
            )
        )

        if emb.ndim != 2 or emb.shape[0] != len(ids):
            raise ValueError(
                f"Invalid embeddings shape {emb.shape}; expected ({len(ids)}, d)."
            )
        if not np.issubdtype(emb.dtype, np.number) or not np.isfinite(emb).all():
            raise ValueError("Embeddings contain invalid numeric values.")

        emb = np.asarray(emb, dtype=np.float32)
        id2idx = {item_id: idx for idx, item_id in enumerate(ids)}

        _atomic_npy(versioned_emb, emb)
        _atomic_json(versioned_map, id2idx)

        metadata = {
            "schema": CACHE_SCHEMA,
            "fingerprint": fingerprint,
            "model_name": str(model_name),
            "shape": list(emb.shape),
            "dtype": str(emb.dtype),
            "embeddings_sha256": _sha256_file(versioned_emb),
            "map_sha256": _sha256_file(versioned_map),
        }
        _atomic_json(meta_path, metadata)

        logger.info("Saved fingerprinted embeddings cache: %s", versioned_emb)
        return emb, id2idx


def load_embeddings(emb_path=EMBEDDINGS_NPY, map_path=EMBEDDING_MAP):
    emb_path = Path(emb_path)
    map_path = Path(map_path)

    emb = np.load(emb_path, allow_pickle=False)
    with map_path.open("r", encoding="utf-8") as f:
        id2idx = json.load(f)

    if emb.ndim != 2:
        raise ValueError(f"Invalid embeddings dimensions: {emb.shape}")
    if not np.issubdtype(emb.dtype, np.number) or not np.isfinite(emb).all():
        raise ValueError("Embeddings contain invalid numeric values.")
    if not isinstance(id2idx, dict) or len(id2idx) != emb.shape[0]:
        raise ValueError("Embeddings and ID map sizes do not match.")
    if sorted(id2idx.values()) != list(range(emb.shape[0])):
        raise ValueError("ID map indices are not unique and contiguous.")

    return emb, id2idx
