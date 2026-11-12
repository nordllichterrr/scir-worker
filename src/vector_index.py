import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import faiss
import numpy as np

from .config import FAISS_INDEX_PATH
from .utils import logger


INDEX_SCHEMA = 2


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, data):
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


@contextmanager
def _index_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _prepare_vectors(embeddings):
    xb = np.asarray(embeddings, dtype=np.float32)

    if xb.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {xb.shape}")
    if xb.shape[0] == 0 or xb.shape[1] == 0:
        raise ValueError("Cannot build FAISS index from empty embeddings.")
    if not np.isfinite(xb).all():
        raise ValueError("Embeddings contain NaN or infinite values.")

    xb = np.ascontiguousarray(xb)
    norms = np.linalg.norm(xb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.ascontiguousarray(xb / norms, dtype=np.float32)


def _index_fingerprint(xb):
    digest = hashlib.sha256()
    digest.update(f"schema={INDEX_SCHEMA}".encode())
    digest.update(str(xb.shape).encode())
    digest.update(str(xb.dtype).encode())
    digest.update(xb.tobytes(order="C"))
    return digest.hexdigest()


def _versioned_path(path: Path, fingerprint: str):
    return path.with_name(f"{path.stem}.{fingerprint}{path.suffix}")


def _load_valid_cache(index_path, meta_path, fingerprint, xb):
    if not index_path.is_file() or not meta_path.is_file():
        return None

    try:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        if meta.get("schema") != INDEX_SCHEMA:
            return None
        if meta.get("fingerprint") != fingerprint:
            return None
        if meta.get("index_sha256") != _sha256_file(index_path):
            return None
        if meta.get("ntotal") != xb.shape[0] or meta.get("dimension") != xb.shape[1]:
            return None

        index = faiss.read_index(str(index_path))
        if index.ntotal != xb.shape[0] or index.d != xb.shape[1]:
            return None

        return index
    except Exception as exc:
        logger.warning("FAISS cache validation failed: %s", exc)
        return None


def build_faiss_index(
    embeddings: np.ndarray,
    index_path: Path = FAISS_INDEX_PATH,
    force_rebuild: bool = False,
):
    index_path = Path(index_path)
    xb = _prepare_vectors(embeddings)
    fingerprint = _index_fingerprint(xb)

    versioned_index = _versioned_path(index_path, fingerprint)
    meta_path = versioned_index.with_suffix(".meta.json")
    lock_path = versioned_index.with_suffix(".lock")

    with _index_lock(lock_path):
        if not force_rebuild:
            cached = _load_valid_cache(
                versioned_index, meta_path, fingerprint, xb
            )
            if cached is not None:
                logger.info("Validated FAISS cache loaded: %s", versioned_index)
                return cached

        index = faiss.IndexFlatIP(xb.shape[1])
        index.add(xb)

        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{versioned_index.name}.",
            suffix=".tmp",
            dir=str(versioned_index.parent),
        )
        os.close(fd)

        try:
            faiss.write_index(index, tmp_name)

            with open(tmp_name, "rb") as f:
                os.fsync(f.fileno())

            # Atomic replacement: the previous index remains intact
            # until the complete temporary index is ready.
            os.replace(tmp_name, versioned_index)

            metadata = {
                "schema": INDEX_SCHEMA,
                "fingerprint": fingerprint,
                "ntotal": int(index.ntotal),
                "dimension": int(index.d),
                "index_sha256": _sha256_file(versioned_index),
            }
            _atomic_json(meta_path, metadata)

        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

        logger.info(
            "FAISS index built: %d vectors, %d dimensions, %s",
            index.ntotal, index.d, versioned_index,
        )
        return index


def load_faiss_index(index_path: Path = FAISS_INDEX_PATH):
    return faiss.read_index(str(index_path))


def search_index(index, query_vec: np.ndarray, top_k=50):
    q = np.asarray(query_vec, dtype=np.float32)

    if q.ndim != 2:
        raise ValueError(f"Expected 2D query vectors, got shape {q.shape}")
    if not np.isfinite(q).all():
        raise ValueError("Query vectors contain NaN or infinite values.")

    q = np.ascontiguousarray(q)
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    q = q / norms

    return index.search(q, top_k)
