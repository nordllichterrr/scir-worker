
import csv
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import argparse

def try_parse_time(s):
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d-%m-%Y %H:%M:%S", "%s"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue

    try:
        return datetime.fromtimestamp(float(s))
    except Exception:
        return None

def create_from_csv(path, out_path, rating_threshold=4.0, last_n=None, user_col='user_id', item_col='item_id', rating_col=None, time_col=None):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        for r in reader:
            rows.append(r)
    if not rows:
        raise SystemExit(f"No rows read from {path}")

    if rating_col is None:
        for c in ['rating','rating_value','score']:
            if c in header:
                rating_col = c
                break
    if time_col is None:
        for c in ['timestamp','time','ts','datetime']:
            if c in header:
                time_col = c
                break
    if user_col not in header or item_col not in header:
        raise SystemExit(f"Expected columns not found in {path}. Available: {header}")


    if last_n is not None or time_col is not None:

        d = defaultdict(list)
        for r in rows:
            uid = str(r[user_col])
            iid = str(r[item_col])
            ts = None
            if time_col and r.get(time_col):
                ts = try_parse_time(r[time_col])
            d[uid].append((ts, iid, r))
        gt = {}
        for uid, recs in d.items():
          
            recs_sorted = sorted(recs, key=lambda x: (x[0] is None, x[0])) if any(x[0] for x in recs) else recs
            if last_n is None:

                picked = [iid for (_, iid, _) in recs_sorted[-1:]]
            else:
                picked = [iid for (_, iid, _) in recs_sorted[-last_n:]]
            gt[uid] = list(dict.fromkeys(picked)) 
    else:

        if rating_col is None:
            raise SystemExit("No timestamp and no rating column detected. Provide --last-n or use a CSV with rating/timestamp.")
        d = defaultdict(list)
        for r in rows:
            try:
                val = float(r.get(rating_col, 0) or 0)
            except:
                val = 0
            if val >= rating_threshold:
                d[str(r[user_col])].append(str(r[item_col]))
        gt = {uid: list(dict.fromkeys(items)) for uid, items in d.items()}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2, ensure_ascii=False)
    print("Saved ground-truth ->", out_path)
    return gt

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to ratings/interactions CSV")
    p.add_argument("--out", default="experiments/ground_truth.json")
    p.add_argument("--rating-threshold", type=float, default=4.0, help="Threshold for ratings -> positive")
    p.add_argument("--last-n", type=int, default=None, help="If provided, pick last-N interactions per user (requires timestamp or will use order).")
    p.add_argument("--user-col", default="user_id")
    p.add_argument("--item-col", default="item_id")
    p.add_argument("--rating-col", default=None)
    p.add_argument("--time-col", default=None)
    args = p.parse_args()

    create_from_csv(
        Path(args.csv),
        Path(args.out),
        rating_threshold=args.rating_threshold,
        last_n=args.last_n,
        user_col=args.user_col,
        item_col=args.item_col,
        rating_col=args.rating_col,
        time_col=args.time_col
    )
