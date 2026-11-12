import argparse
import csv
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

from src.rerank_llm import format_reranker_query, format_reranker_doc


ROOT = Path(__file__).resolve().parents[1]
DATA_SEREN = ROOT / "data" / "serendipity-sac2018" / "training.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
ITEMS_CSV = PROCESSED_DIR / "items_serendipity.csv"
FAISS_INDEX_PATH = ROOT / "data" / "index" / "items.faiss"
EMBEDDING_MAP_PATH = ROOT / "data" / "embeddings" / "id2idx.json"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def make_item_text(meta: Dict[str, Any]) -> str:

    title = meta.get("title", "")
    desc = meta.get("description", "") or meta.get("text", "") or ""
    tags = meta.get("format_tags") or meta.get("genres", "") or ""
    if isinstance(tags, (list, tuple)):
        tags = ", ".join(str(x) for x in tags)
    s = f"Title: {title}. Description: {desc}. Tags: {tags}"
    return s.strip()


def load_items_meta(path: Path) -> Dict[str, Dict[str, Any]]:
  
    if not path.exists():
        return {}
    by_id = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iid = row.get("item_id") or row.get("movieId") or row.get("movie_id")
            if not iid:
                continue
            by_id[str(iid)] = {
                "title": row.get("title", ""),
                "description": row.get("description", ""),
                "genres": row.get("genres", ""),
            }
    return by_id


def load_training_positives(path: Path, max_users: int = None) -> Dict[str, List[str]]:

    if not path.exists():
        return {}
    from collections import defaultdict
    user_items = defaultdict(list)
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        u_col = i_col = None
        for row in reader:
            if not row:
                continue
            if u_col is None:
                keys = [k for k in row.keys() if k]
                u_col = next((k for k in keys if "user" in k.lower() and "id" in k.lower()), keys[0] if keys else None)
                i_col = next((k for k in keys if "item" in k.lower() or "movie" in k.lower()), keys[1] if len(keys) > 1 else None)
            if u_col and i_col:
                uid = str(row.get(u_col, ""))
                iid = str(row.get(i_col, ""))
                if uid and iid:
                    user_items[uid].append(iid)
            if max_users and len(user_items) >= max_users:
                break
    return dict(user_items)


def get_hard_negatives(
    user_texts: Dict[str, str],
    user_positives: Dict[str, List[str]],
    items_meta: Dict[str, Dict[str, Any]],
    index_path: Path,
    id2idx_path: Path,
    query_encoder_name: str,
    top_k: int = 200,
    negs_per_user: int = 5,
) -> Dict[str, List[str]]:
    try:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return {}
    if not index_path.exists() or not id2idx_path.exists():
        return {}
    with open(id2idx_path, "r", encoding="utf-8") as f:
        id2idx = json.load(f)
    idx2id = {str(v): k for k, v in id2idx.items()}
    index = faiss.read_index(str(index_path))
    model = SentenceTransformer(query_encoder_name)
    def _norm(x):
        x = x.astype("float32")
        n = np.linalg.norm(x, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return x / n
    hard_negs = {}
    for uid, qtext in user_texts.items():
        pos_set = set(user_positives.get(uid, []))
        qvec = model.encode([qtext], convert_to_numpy=True)
        qvec = _norm(qvec)
        D, I = index.search(qvec, min(top_k, index.ntotal))
        negs = []
        for idx in I[0]:
            iid = idx2id.get(str(int(idx)))
            if iid and iid not in pos_set and iid in items_meta:
                negs.append(iid)
            if len(negs) >= negs_per_user:
                break
        hard_negs[uid] = negs
    return hard_negs


def build_training_examples(
    user_positives: Dict[str, List[str]],
    items_meta: Dict[str, Dict[str, Any]],
    user_texts: Dict[str, str],
    hard_negatives: Dict[str, List[str]] = None,
    max_positive_pairs: int = 50000,
    neg_per_pos_ratio: float = 1.0,
) -> List[InputExample]:
    import random
    examples = []
    for uid, pos_ids in user_positives.items():
        if not pos_ids:
            continue
        qtext = user_texts.get(uid)
        if not qtext:
            continue
        for iid in pos_ids:
            if len(examples) >= max_positive_pairs:
                break
            meta = items_meta.get(iid, {})
            item_text = make_item_text(meta)
            if not item_text.strip():
                continue
            query_side = format_reranker_query(qtext)
            doc_side = format_reranker_doc(item_text)
            examples.append(InputExample(texts=[query_side, doc_side], label=1.0))
        if len(examples) >= max_positive_pairs:
            break
    n_pos = len(examples)
    neg_source = hard_negatives if hard_negatives else {}
    all_iids = list(items_meta.keys())
    target_neg = int(n_pos * neg_per_pos_ratio)
    n_neg = 0
    for uid, pos_ids in user_positives.items():
        if n_neg >= target_neg:
            break
        qtext = user_texts.get(uid)
        if not qtext:
            continue
        pos_set = set(pos_ids)
        negs = neg_source.get(uid, [])
        if not negs:
            negs = [i for i in all_iids if i not in pos_set]
            random.shuffle(negs)
        for iid in negs[:max(1, int(neg_per_pos_ratio * len(pos_ids)) + 1)]:
            if n_neg >= target_neg:
                break
            meta = items_meta.get(iid, {})
            item_text = make_item_text(meta)
            if not item_text.strip():
                continue
            query_side = format_reranker_query(qtext)
            doc_side = format_reranker_doc(item_text)
            examples.append(InputExample(texts=[query_side, doc_side], label=0.0))
            n_neg += 1
    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune CrossEncoder on Serendipity with pairwise data and optional hard negatives."
    )
    parser.add_argument("--model-name", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2", help="Base CrossEncoder.")
    parser.add_argument("--output-dir", type=str, default="models/crossencoder_finetuned", help="Save fine-tuned model here.")
    parser.add_argument("--training-csv", type=Path, default=DATA_SEREN, help="Training interactions (user_id, item_id).")
    parser.add_argument("--items-csv", type=Path, default=ITEMS_CSV, help="Items metadata (item_id, title, genres, ...).")
    parser.add_argument("--use-hard-negatives", action="store_true", help="Use ANN retrieval for hard negatives (requires existing FAISS index).")
    parser.add_argument("--ann-top-k", type=int, default=200, help="ANN top-K to sample hard negs from.")
    parser.add_argument("--negs-per-user", type=int, default=5, help="Max hard negatives per user.")
    parser.add_argument("--max-samples", type=int, default=50000, help="Max (positive) training pairs.")
    parser.add_argument("--neg-per-pos-ratio", type=float, default=1.0, help="Negatives per positive (1.0 = balanced).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--scheduler", type=str, default="WarmupLinear", choices=["WarmupLinear", "WarmupCosine", "WarmupConstant"])
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd", "adafactor"])
    parser.add_argument("--sgd-momentum", type=float, default=0.9)
    parser.add_argument("--init-seed", type=int, default=42)
    parser.add_argument("--max-users", type=int, default=None, help="Cap users for hard neg retrieval (faster).")
    parser.add_argument("--dry-run", action="store_true", help="Build dataset and print train config without fitting.")
    args = parser.parse_args()

    random.seed(args.init_seed)
    np.random.seed(args.init_seed)
    torch.manual_seed(args.init_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.init_seed)

    items_meta = load_items_meta(args.items_csv)
    if not items_meta:
        raise SystemExit(f"No items loaded from {args.items_csv}. Run preprocess and ensure items CSV exists.")

    user_positives = load_training_positives(args.training_csv, max_users=args.max_users)
    if not user_positives:
        raise SystemExit(f"No training positives from {args.training_csv}.")

    user_texts = {}
    for uid, pos_ids in user_positives.items():
        titles = []
        for iid in pos_ids[:10]:
            t = (items_meta.get(iid) or {}).get("title", "")
            if t:
                titles.append(t)
        user_texts[uid] = "; ".join(titles) if titles else f"User {uid}"

    hard_negatives = {}
    if args.use_hard_negatives:
        print("Loading ANN index for hard negatives...")
        hard_negatives = get_hard_negatives(
            user_texts,
            user_positives,
            items_meta,
            index_path=FAISS_INDEX_PATH,
            id2idx_path=EMBEDDING_MAP_PATH,
            query_encoder_name=EMBED_MODEL,
            top_k=args.ann_top_k,
            negs_per_user=args.negs_per_user,
        )
        n_with = sum(1 for v in hard_negatives.values() if v)
        print(f"Hard negatives: {n_with} users with at least one neg.")

    examples = build_training_examples(
        user_positives,
        items_meta,
        user_texts,
        hard_negatives=hard_negatives if hard_negatives else None,
        max_positive_pairs=args.max_samples,
        neg_per_pos_ratio=args.neg_per_pos_ratio,
    )
    if not examples:
        raise SystemExit("No training examples built.")

    print(f"Training examples: {len(examples)} (positives ~{sum(1 for e in examples if e.label == 1.0)}, negatives ~{sum(1 for e in examples if e.label == 0.0)})")
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    model = CrossEncoder(args.model_name, num_labels=1)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    optimizer_class = torch.optim.AdamW
    optimizer_params = {"lr": args.lr}
    if args.optimizer == "sgd":
        optimizer_class = torch.optim.SGD
        optimizer_params = {"lr": args.lr, "momentum": args.sgd_momentum}
    elif args.optimizer == "adafactor":
        try:
            from transformers import Adafactor
        except Exception as e:
            raise SystemExit(f"Adafactor requested, but transformers.Adafactor unavailable: {e}")
        optimizer_class = Adafactor
        optimizer_params = {"lr": args.lr, "scale_parameter": False, "relative_step": False}

    if args.dry_run:
        print(
            f"[DRY-RUN] optimizer={args.optimizer}, scheduler={args.scheduler}, "
            f"lr={args.lr}, weight_decay={args.weight_decay}, init_seed={args.init_seed}, "
            f"epochs={args.epochs}, batch_size={args.batch_size}, examples={len(examples)}"
        )
        return
    model.fit(
        train_dataloader=train_dataloader,
        epochs=args.epochs,
        optimizer_class=optimizer_class,
        optimizer_params=optimizer_params,
        scheduler=args.scheduler,
        weight_decay=args.weight_decay,
        warmup_steps=min(500, int(0.1 * len(train_dataloader))),
        output_path=args.output_dir,
    )
    print(f"Saved fine-tuned model to {args.output_dir}")
    print("Run ablation: python -m src.run_reranker_variants  (zero-shot vs fine-tuned)")


if __name__ == "__main__":
    main()
