
import csv
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional, Set

from .evaluation_config import (
    SPLIT_RATIOS,
    SPLIT_METHOD,
    COLD_START_SCENARIOS,
    NEW_USER_MAX_TRAIN,
    INTERACTION_BUCKETS,
    get_eval_paths,
    get_default_raw_interactions_csv,
    _ds_key,
)

from .utils import logger

# TRAINING_CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "serendipity-sac2018" / "training.csv"


def _bucket_label(n: int) -> str:
    for (lo, hi) in INTERACTION_BUCKETS:
        if hi is None:
            if n >= lo:
                return f"{lo}+"
            continue
        if lo <= n <= hi:
            return f"{lo}-{hi}"
    return "0" 


def _parse_ts(val: Optional[str]):

    if val is None or val == "":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def load_interactions(
    path: Path,
    user_col: str = "user_id",
    item_col: str = "item_id",
    time_col: Optional[str] = "timestamp",
) -> List[Dict[str, Any]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            uid = r.get(user_col) or r.get("user") or r.get("userId")
            iid = r.get(item_col) or r.get("item") or r.get("movieId") or r.get("itemId")
            if uid is None or iid is None:
                vals = list(r.values())
                if len(vals) >= 2:
                    uid, iid = str(vals[0]), str(vals[1])
                else:
                    continue
            ts = r.get(time_col) or r.get("timestamp") or r.get("ts")
            rows.append({"user_id": str(uid), "item_id": str(iid), "timestamp": ts})
    return rows


def split_by_time(
    rows: List[Dict[str, Any]],
    ratios: Tuple[float, float, float] = SPLIT_RATIOS,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    train_r, val_r, test_r = ratios
    if abs(train_r + val_r + test_r - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1.0")

    with_ts = [(r, _parse_ts(r.get("timestamp"))) for r in rows]
    with_ts.sort(key=lambda x: (x[1] is None, x[1] or 0))
    n = len(with_ts)
    n_train = int(n * train_r)
    n_val = int(n * val_r)
    n_test = n - n_train - n_val

    train = [r for r, _ in with_ts[:n_train]]
    val = [r for r, _ in with_ts[n_train : n_train + n_val]]
    test = [r for r, _ in with_ts[n_train + n_val :]]
    return train, val, test


def split_random(
    rows: List[Dict[str, Any]],
    ratios: Tuple[float, float, float] = SPLIT_RATIOS,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    import random
    rng = random.Random(seed)
    shuffled = rows.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    n_test = n - n_train - n_val
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def split_leave_last_out(
    rows: List[Dict[str, Any]],
    min_user_interactions: int = 1,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Leave-one-out per user:
      - last interaction -> test
      - penultimate -> val
      - rest -> train
    Only keep users in val/test if they have at least `min_user_interactions`
    interactions in train.
    """
    by_user: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        uid = str(r["user_id"])
        by_user[uid].append(r)

    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []

    for uid, urows in by_user.items():
        if not urows:
            continue
        # sort by timestamp (if present)
        with_ts = [(r, _parse_ts(r.get("timestamp"))) for r in urows]
        with_ts.sort(key=lambda x: (x[1] is None, x[1] or 0))
        ordered = [r for r, _ in with_ts]
        n = len(ordered)
        if n == 1:
            # Only one interaction – goes to test, but this user will be
            # filtered out from test if min_user_interactions > 0 (no train).
            train_rows: List[Dict[str, Any]] = []
            val_row = None
            test_row = ordered[-1]
        else:
            test_row = ordered[-1]
            val_row = ordered[-2] if n >= 2 else None
            train_rows = ordered[:-2] if n > 2 else []

        train.extend(train_rows)
        # keep user in val/test only if enough train interactions
        if len(train_rows) >= max(0, min_user_interactions):
            if val_row is not None:
                val.append(val_row)
            test.append(test_row)

    return train, val, test


def _stable_group_fold_map(group_ids: Set[str], n_folds: int, split_seed: int = 42) -> Dict[str, int]:
    """
    Deterministic fold assignment for groups (users/items).
    Same ids -> same fold map for fixed n_folds.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    ordered = sorted(str(g) for g in group_ids)
    import random
    rng = random.Random(int(split_seed))
    rng.shuffle(ordered)
    return {gid: (idx % n_folds) for idx, gid in enumerate(ordered)}


def _write_interactions_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["user_id", "item_id", "timestamp"])
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "user_id": r["user_id"],
                    "item_id": r["item_id"],
                    "timestamp": r.get("timestamp") or "",
                }
            )


def _write_ground_truth_csv(path: Path, gt_map: Dict[str, List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["user_id", "item_id"])
        w.writeheader()
        for uid in sorted(gt_map.keys()):
            for iid in gt_map[uid]:
                w.writerow({"user_id": uid, "item_id": iid})


def split_group_kfold(
    rows: List[Dict[str, Any]],
    cv_mode: str,
    n_folds: int,
    fold_id: int,
    split_seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    if fold_id < 0 or fold_id >= n_folds:
        raise ValueError(f"fold_id must be in [0, {n_folds - 1}]")

    val_fold = (fold_id + 1) % n_folds
    users = {r["user_id"] for r in rows}
    items = {r["item_id"] for r in rows}

    if cv_mode == "user_kfold":
        u2f = _stable_group_fold_map(users, n_folds, split_seed=split_seed)
        train = [r for r in rows if u2f[r["user_id"]] not in {fold_id, val_fold}]
        val = [r for r in rows if u2f[r["user_id"]] == val_fold]
        test = [r for r in rows if u2f[r["user_id"]] == fold_id]
        fold_info = {
            "cv_mode": cv_mode,
            "n_folds": n_folds,
            "fold_id": fold_id,
            "val_fold_id": val_fold,
            "group_key": "user_id",
            "split_seed": int(split_seed),
            "n_groups": len(users),
            "fold_counts": dict(defaultdict(int, {str(fid): sum(1 for _, f in u2f.items() if f == fid) for fid in range(n_folds)})),
        }
        return train, val, test, fold_info

    if cv_mode == "item_kfold":
        i2f = _stable_group_fold_map(items, n_folds, split_seed=split_seed)
        train = [r for r in rows if i2f[r["item_id"]] not in {fold_id, val_fold}]
        val = [r for r in rows if i2f[r["item_id"]] == val_fold]
        test = [r for r in rows if i2f[r["item_id"]] == fold_id]
        fold_info = {
            "cv_mode": cv_mode,
            "n_folds": n_folds,
            "fold_id": fold_id,
            "val_fold_id": val_fold,
            "group_key": "item_id",
            "split_seed": int(split_seed),
            "n_groups": len(items),
            "fold_counts": dict(defaultdict(int, {str(fid): sum(1 for _, f in i2f.items() if f == fid) for fid in range(n_folds)})),
        }
        return train, val, test, fold_info

    if cv_mode == "both_kfold":
        u2f = _stable_group_fold_map(users, n_folds, split_seed=split_seed)
        i2f = _stable_group_fold_map(items, n_folds, split_seed=split_seed)
        # Cartesian-style fold protocol:
        # - test: user and item both assigned to fold_id
        # - val: user and item both assigned to val_fold
        # - train: interactions that contain neither test-fold users nor test-fold items
        train = [r for r in rows if u2f[r["user_id"]] != fold_id and i2f[r["item_id"]] != fold_id and u2f[r["user_id"]] != val_fold and i2f[r["item_id"]] != val_fold]
        val = [r for r in rows if u2f[r["user_id"]] == val_fold and i2f[r["item_id"]] == val_fold]
        test = [r for r in rows if u2f[r["user_id"]] == fold_id and i2f[r["item_id"]] == fold_id]
        fold_info = {
            "cv_mode": cv_mode,
            "n_folds": n_folds,
            "fold_id": fold_id,
            "val_fold_id": val_fold,
            "group_key": "user_id+item_id",
            "split_seed": int(split_seed),
            "n_user_groups": len(users),
            "n_item_groups": len(items),
        }
        return train, val, test, fold_info

    raise ValueError(f"Unsupported cv_mode: {cv_mode}")


def make_few_shot_test_protocol(
    test_rows: List[Dict[str, Any]],
    few_shot_n: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    For each test user, keep first N interactions (time-ordered) as observed profile.
    The rest become targets for ground-truth evaluation.
    """
    if few_shot_n <= 0:
        return test_rows, test_rows

    by_user: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in test_rows:
        by_user[r["user_id"]].append(r)

    observed: List[Dict[str, Any]] = []
    target: List[Dict[str, Any]] = []
    for uid, urows in by_user.items():
        with_ts = [(r, _parse_ts(r.get("timestamp"))) for r in urows]
        with_ts.sort(key=lambda x: (x[1] is None, x[1] or 0))
        ordered = [r for r, _ in with_ts]
        observed.extend(ordered[:few_shot_n])
        target.extend(ordered[few_shot_n:])

    return observed, target


def build_splits(
    interactions_path: Path,
    dataset: str = "serendipity",
    out_train: Path = None,
    out_val: Path = None,
    out_test: Path = None,
    out_gt: Path = None,
    out_meta: Path = None,
    by_time: Optional[bool] = None,
    ratios: Optional[Tuple[float, float, float]] = None,
    seed: int = 42,
    user_col: str = "user_id",
    item_col: str = "item_id",
    cv_mode: str = "none",
    n_folds: int = 5,
    fold_id: int = 0,
    few_shot_n: int = 0,
    split_seed: int = 42,
) -> Dict[str, Any]:
    rows = load_interactions(interactions_path, user_col=user_col, item_col=item_col)
    if by_time is None:
        by_time = (SPLIT_METHOD == "time_based")

    if by_time:
        has_any_ts = any(_parse_ts(r.get("timestamp")) is not None for r in rows)
        if not has_any_ts:
            logger.warning("No usable timestamps found; falling back to random split.")
            by_time = False
    paths = get_eval_paths(dataset)

    ds_key = _ds_key(dataset)
    if cv_mode == "none":
        out_train = Path(out_train) if out_train else paths["train"]
        out_val = Path(out_val) if out_val else paths["val"]
        out_test = Path(out_test) if out_test else paths["test"]
        out_gt = Path(out_gt) if out_gt else paths["gt"]
        out_meta = Path(out_meta) if out_meta else paths["split_meta"]
    else:
        fold_root = Path(out_train).parent if out_train else (
            Path(__file__).resolve().parents[1] / "data" / "processed" / ds_key / f"fold_{fold_id}"
        )
        out_train = Path(out_train) if out_train else (fold_root / "train_interactions.csv")
        out_val = Path(out_val) if out_val else (fold_root / "val_interactions.csv")
        out_test = Path(out_test) if out_test else (fold_root / "test_interactions.csv")
        out_gt = Path(out_gt) if out_gt else (fold_root / "ground_truth.csv")
        out_meta = Path(out_meta) if out_meta else (fold_root / "fold_metadata.json")
    if not rows:
        raise SystemExit(f"No rows loaded from {interactions_path}")

    # For MovieLens, use per-user leave-one-out splitting:
    # last -> test, penultimate -> val, rest -> train; then filter test users
    # so that everyone in test has at least 1 train interaction.
    split_ratios = ratios if ratios is not None else SPLIT_RATIOS
    fold_info: Dict[str, Any] = {}

    if cv_mode != "none":
        train, val, test, fold_info = split_group_kfold(
            rows=rows,
            cv_mode=cv_mode,
            n_folds=n_folds,
            fold_id=fold_id,
            split_seed=split_seed,
        )
        logger.info("Using CV split mode=%s, fold=%d/%d", cv_mode, fold_id, n_folds)
    elif ds_key == "movielens":
        min_user_interactions = 0
        logger.info(
            "Using leave-one-out split for MovieLens (min_user_interactions_in_train=%d)",
            min_user_interactions,
        )
        train, val, test = split_leave_last_out(
            rows,
            min_user_interactions=min_user_interactions,
        )
    else:
        if by_time is None:
            by_time = (SPLIT_METHOD == "time_based")
        
        logger.info("Split method: %s (by_time=%s)", SPLIT_METHOD, by_time)
        
        if by_time:
            train, val, test = split_by_time(rows, ratios=split_ratios)
        else:
            train, val, test = split_random(rows, ratios=split_ratios, seed=seed)

    if few_shot_n > 0:
        test_observed, gt_source_rows = make_few_shot_test_protocol(test, few_shot_n=few_shot_n)
    else:
        test_observed, gt_source_rows = test, test

    train_items = set(r["item_id"] for r in train)

    train_count_by_user = defaultdict(int)
    for r in train:
        train_count_by_user[r["user_id"]] += 1

    gt = defaultdict(list)
    test_users = set()
    for r in gt_source_rows:
        uid, iid = r["user_id"], r["item_id"]
        gt[uid].append(iid)
        test_users.add(uid)
    gt = {uid: list(dict.fromkeys(items)) for uid, items in gt.items()}


    user_meta = {}
    for uid in test_users:
        n_train = train_count_by_user.get(uid, 0)
        user_meta[uid] = {
            "n_train_interactions": n_train,
            "bucket": _bucket_label(n_train),
            "is_new_user": n_train <= NEW_USER_MAX_TRAIN,
        }

    users_new_items = set()
    for r in gt_source_rows:
        uid, iid = r["user_id"], r["item_id"]
        if iid not in train_items:
            users_new_items.add(uid)

    scenario_to_users = {
        "new_users": [uid for uid in test_users if user_meta[uid]["is_new_user"]],
        "new_items": list(users_new_items),
        "both": [u for u in test_users if user_meta[u]["is_new_user"] and u in users_new_items],
    }

    bucket_to_users = defaultdict(list)
    for uid in test_users:
        bucket_to_users[user_meta[uid]["bucket"]].append(uid)

    split_metadata = {
        "dataset": ds_key,
        "cv_mode": cv_mode,
        "n_folds": n_folds if cv_mode != "none" else None,
        "fold_id": fold_id if cv_mode != "none" else None,
        "few_shot_n": few_shot_n,
        "split_seed": int(split_seed),
        "fold_info": fold_info,
        "split_ratios": list(split_ratios),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "n_test_observed": len(test_observed),
        "n_test_targets": len(gt_source_rows),
        "n_test_users": len(test_users),
        "user_meta": user_meta,
        "scenario_to_users": scenario_to_users,
        "bucket_to_users": dict(bucket_to_users),
        "train_items": list(train_items),
        "new_user_max_train": NEW_USER_MAX_TRAIN,
    }


    _write_interactions_csv(out_train, train)
    _write_interactions_csv(out_val, val)
    _write_interactions_csv(out_test, test_observed)
    if out_gt.suffix.lower() == ".csv":
        _write_ground_truth_csv(out_gt, gt)
    else:
        with open(out_gt, "w", encoding="utf-8") as f:
            json.dump({"data": gt, "_split": "test"}, f, indent=2, ensure_ascii=False)

    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(split_metadata, f, indent=2, ensure_ascii=False)

    logger.info(
        "Splits written: train=%s (%d), val=%s (%d), test=%s (%d), gt=%s, meta=%s",
        out_train, len(train), out_val, len(val), out_test, len(test), out_gt, out_meta,
    )
    return {
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "n_test_observed": len(test_observed),
        "n_test_targets": len(gt_source_rows),
        "n_test_users": len(test_users),
        "scenario_sizes": {s: len(scenario_to_users[s]) for s in COLD_START_SCENARIOS},
        "bucket_sizes": {b: len(bucket_to_users[b]) for b in sorted(bucket_to_users.keys(), key=lambda x: (0 if x == "0" else int(x.split("-")[0]) if "-" in x else int(x.replace("+", "") or 21)))},
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Create standardized train/val/test splits for cold-start evaluation.")
    p.add_argument("--dataset", default="serendipity", choices=["serendipity", "taobao", "movielens"])
    p.add_argument("--method", type=str, default="time_based",
                    choices=["time_based", "leave_last_out"])
    p.add_argument("--min_user_interactions", type=int, default=5)
    p.add_argument("--csv", default=None, help="Path to interactions CSV (optional; defaults per dataset)")
    p.add_argument("--by-time", action="store_true", default=True, help="Split by timestamp (default: True)")
    p.add_argument("--random", action="store_true", help="Split randomly instead of by time")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ratios", nargs=3, type=float, default=None, help="Split ratios: train val test (must sum to 1.0)")
    p.add_argument("--cv-mode", type=str, default="none", choices=["none", "user_kfold", "item_kfold", "both_kfold"])
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--fold-id", type=int, default=0)
    p.add_argument("--few-shot-n", type=int, default=0, help="For test users, first N interactions are observed profile; the rest are targets")
    p.add_argument("--split-seed", type=int, default=42, help="Random seed for entity-to-fold assignment in CV modes.")
    p.add_argument("--out-train", default=None, help="Output path for training.csv")
    p.add_argument("--out-val", default=None, help="Output path for val_interactions.csv")
    p.add_argument("--out-test", default=None, help="Output path for test_interactions.csv")
    p.add_argument("--out-meta", default=None, help="Output path for split_metadata.json")
    p.add_argument("--out-gt", default=None, help="Output path for ground_truth.json")
    args = p.parse_args()
    from .evaluation_config import get_eval_paths, get_default_raw_interactions_csv

    paths = get_eval_paths(args.dataset)

    csv_path = Path(args.csv) if args.csv else get_default_raw_interactions_csv(args.dataset)

    if args.cv_mode == "none":
        out_train = Path(args.out_train) if args.out_train else paths["train"]
        out_val = Path(args.out_val) if args.out_val else paths["val"]
        out_test = Path(args.out_test) if args.out_test else paths["test"]
        out_meta = Path(args.out_meta) if args.out_meta else paths["split_meta"]
        out_gt = Path(args.out_gt) if args.out_gt else paths["gt"]
    else:
        out_train = Path(args.out_train) if args.out_train else None
        out_val = Path(args.out_val) if args.out_val else None
        out_test = Path(args.out_test) if args.out_test else None
        out_meta = Path(args.out_meta) if args.out_meta else None
        out_gt = Path(args.out_gt) if args.out_gt else None

    summary = build_splits(
        csv_path,
        dataset=args.dataset,
        out_train=out_train,
        out_val=out_val,
        out_test=out_test,
        out_gt=out_gt,
        out_meta=out_meta,
        by_time=args.by_time and not args.random,
        seed=args.seed,
        ratios=tuple(args.ratios) if args.ratios else None,
        cv_mode=args.cv_mode,
        n_folds=args.n_folds,
        fold_id=args.fold_id,
        few_shot_n=args.few_shot_n,
        split_seed=args.split_seed,
    )
    print("Summary:", json.dumps(summary, indent=2))
