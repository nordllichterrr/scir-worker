
import json
from pathlib import Path
from collections import Counter

RANKINGS = Path("results") / "demo_rankings.json"

def norm_recs_list(r):
    if not r:
        return []
    if isinstance(r, list):
        if isinstance(r[0], dict):
            return [str(x.get("item_id") or x.get("id") or x.get("doc_id")) for x in r]
        return [str(x) for x in r]
    if isinstance(r, dict):
        v = r.get("item_id") or r.get("id") or r.get("doc_id")
        return [str(v)] if v is not None else []
    return [str(r)]

def main():
    R = json.load(open(RANKINGS, encoding="utf-8"))
    top1 = []
    unique_items = set()
    for u, recs in R.items():
        ids = norm_recs_list(recs)
        if ids:
            top1.append(ids[0])
            unique_items.update(ids[:10])
    c = Counter(top1)
    print("users:", len(R))
    print("unique items in top-10 across users (est):", len(unique_items))
    print("unique top-1 items:", len(set(top1)))
    print("Top-20 most common top-1:")
    for it, cnt in c.most_common(20):
        print(f"  {it}: {cnt}")

if __name__ == "__main__":
    main()
