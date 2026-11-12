from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Tuple

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.run_experiment import run_with_logging
from src.create_splits import build_splits


OUT_DIR = project_root / "experiments" / "robustness"
NOISE_DIR = OUT_DIR / "noise"
SHIFT_DIR = OUT_DIR / "temporal_shift"
RESULTS_DIR = project_root / "results" / "robustness"

DEFAULT_BASE_TRAIN = project_root / "experiments" / "training_interactions.csv"
DEFAULT_INPUT_CSV = project_root / "data" / "serendipity-sac2018" / "training.csv"


def _read_train_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        rows = [row for row in r if row]
        fieldnames = r.fieldnames or []
    return rows, fieldnames


def _write_rows(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def make_noisy_train_drop(
    base_train_csv: Path,
    out_csv: Path,
    drop_p: float,
    seed: int,
) -> None:
    rows, fieldnames = _read_train_rows(base_train_csv)
    if not rows:
        raise FileNotFoundError(f"No rows in base train CSV: {base_train_csv}")
    if drop_p <= 0:
        _write_rows(out_csv, rows, fieldnames)
        return

    rng = random.Random(seed)
    by_user: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        uid = str(row.get("user_id") or row.get("user") or row.get("userId") or "")
        by_user[uid].append(row)

    kept: List[Dict[str, str]] = []
    for uid, urows in by_user.items():
        if not uid or not urows:
            continue
        n = len(urows)
        n_drop = int(round(n * drop_p))
        if n_drop <= 0:
            kept.extend(urows)
            continue
        if n_drop >= n:
            n_drop = n - 1
        drop_idx = set(rng.sample(range(n), n_drop))
        for i, row in enumerate(urows):
            if i not in drop_idx:
                kept.append(row)

    _write_rows(out_csv, kept, fieldnames)


def run_noise_suite(n_users: int, seeds: List[int], dataset: str, fast: bool) -> None:
    NOISE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    base_train = DEFAULT_BASE_TRAIN if DEFAULT_BASE_TRAIN.exists() else DEFAULT_INPUT_CSV
    if not base_train.exists():
        raise FileNotFoundError(f"Base train CSV not found at {base_train}")

    runs_log = OUT_DIR / "runs_noise.jsonl"
    drop_levels = [0.0, 0.2] if fast else [0.0, 0.1, 0.2, 0.3]

    model_configs = [
        ("popularity", {"baseline": "popularity", "use_reranker": False, "topk": 10}),
        ("bm25", {"baseline": "content_bm25", "use_reranker": False, "topk": 10}),
        ("ours_hybrid_retrieval", {"baseline": None, "use_reranker": False, "topk": 10, "retrieval_mode": "hybrid", "candidate_pool_size": 1000}),
    ]

    for drop_p in drop_levels:
        for seed in seeds:
            noisy_csv = NOISE_DIR / f"train_drop_p{int(drop_p*100):02d}_seed{seed}.csv"
            make_noisy_train_drop(base_train, noisy_csv, drop_p=drop_p, seed=seed)

            for short_name, cfg in model_configs:
                run_id = f"robust_noise_drop_p{int(drop_p*100):02d}_{short_name}_{dataset}_seed{seed}_n{n_users}"
                run_with_logging(
                    run_id=run_id,
                    n_users=n_users,
                    seed=seed,
                    config={
                        **cfg,
                        "dataset": dataset,
                        "train_interactions_path": str(noisy_csv),
                        "runs_log_path": str(runs_log),
                        "results_dir": str(RESULTS_DIR / "noise"),
                        "robustness": {"type": "noise_drop", "drop_p": drop_p},
                    },
                    dataset=dataset,
                )


def run_temporal_shift_suite(n_users: int, seeds: List[int], dataset: str, fast: bool) -> None:
    SHIFT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DEFAULT_INPUT_CSV.exists():
        raise FileNotFoundError(f"Input interactions CSV not found at {DEFAULT_INPUT_CSV}")

    runs_log = OUT_DIR / "runs_temporal_shift.jsonl"
    shifts = [(0.8, 0.1, 0.1), (0.7, 0.1, 0.2)] if fast else [(0.85, 0.05, 0.10), (0.80, 0.10, 0.10), (0.70, 0.10, 0.20), (0.60, 0.10, 0.30)]

    model_configs = [
        ("popularity", {"baseline": "popularity", "use_reranker": False, "topk": 10}),
        ("bm25", {"baseline": "content_bm25", "use_reranker": False, "topk": 10}),
        ("ours_hybrid_retrieval", {"baseline": None, "use_reranker": False, "topk": 10, "retrieval_mode": "hybrid", "candidate_pool_size": 1000}),
    ]

    for (tr, va, te) in shifts:
        shift_tag = f"tr{int(tr*100):02d}_te{int(te*100):02d}"
        split_out = SHIFT_DIR / shift_tag
        split_out.mkdir(parents=True, exist_ok=True)

        out_train = split_out / "training_interactions.csv"
        out_val = split_out / "val_interactions.csv"
        out_test = split_out / "test_interactions.csv"
        out_gt = split_out / "ground_truth.json"
        out_meta = split_out / "split_metadata.json"

        build_splits(
            interactions_path=DEFAULT_INPUT_CSV,
            out_train=out_train,
            out_val=out_val,
            out_test=out_test,
            out_gt=out_gt,
            out_meta=out_meta,
            by_time=True,
            ratios=(tr, va, te),
            seed=42,
        )

        for seed in seeds:
            for short_name, cfg in model_configs:
                run_id = f"robust_shift_{shift_tag}_{short_name}_{dataset}_seed{seed}_n{n_users}"
                run_with_logging(
                    run_id=run_id,
                    n_users=n_users,
                    seed=seed,
                    config={
                        **cfg,
                        "dataset": dataset,
                        "train_interactions_path": str(out_train),
                        "ground_truth_path": str(out_gt),
                        "split_metadata_path": str(out_meta),
                        "runs_log_path": str(runs_log),
                        "results_dir": str(RESULTS_DIR / "temporal_shift" / shift_tag),
                        "robustness": {"type": "temporal_shift", "shift_tag": shift_tag, "ratios": [tr, va, te]},
                    },
                    dataset=dataset,
                )


def main():
    p = argparse.ArgumentParser(description="Run robustness suite (noise, temporal shift, sensitivity report).")
    p.add_argument("--fast", action="store_true", help="Quick robustness: fewer noise levels and shift points.")
    p.add_argument("--n-users", type=int, default=200, help="Users for robustness runs (default: 200).")
    p.add_argument("--seeds", nargs="+", type=int, default=[42], help="Seeds for robustness runs (default: 42).")
    p.add_argument("--dataset", type=str, default="serendipity", choices=["serendipity", "taobao"])
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_noise_suite(n_users=args.n_users, seeds=args.seeds, dataset=args.dataset, fast=args.fast)
    run_temporal_shift_suite(n_users=args.n_users, seeds=args.seeds, dataset=args.dataset, fast=args.fast)

    from tools.robustness_report import generate_robustness_report
    generate_robustness_report(project_root=project_root)

    print(f"Robustness outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()

