
import csv
import json
from pathlib import Path
from collections import defaultdict

TRAIN = Path("data/serendipity-sac2018/training.csv")
OUT = Path("experiments/ground_truth.json")

def build_gt():
    if not TRAIN.exists():
        print("training.csv not found:", TRAIN)
        return
    gt = defaultdict(list)
    with open(TRAIN, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
      
            uid = r.get("user_id") or r.get("user") or r.get("uid") or r.get("userId")
            iid = r.get("item_id") or r.get("item") or r.get("itemId")
            if uid is None or iid is None:
    
                vals = list(r.values())
                if len(vals) >= 2:
                    uid, iid = vals[0], vals[1]
                else:
                    continue
            gt[str(uid)].append(str(iid))
    out = {"_synthetic": False, "data": dict(gt)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("Wrote ground-truth:", OUT)

if __name__ == "__main__":
    build_gt()
