import json
from pathlib import Path
import math

RESULTS = Path("results") / "demo_rankings.json"
GT = Path("experiments") / "ground_truth.json"
K = 10

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_recs_list(r):
    if not r:
        return []
    if isinstance(r, list):
        if isinstance(r[0], dict):
            return [str(x.get("item_id")) for x in r]
        return [str(x) for x in r]
    if isinstance(r, dict):
        return [str(r.get("item_id"))]
    return [str(r)]

def hr_at_k(gt_list, rec_list, k):
    gtset = set(str(x) for x in (gt_list or []))
    reck = rec_list[:k]
    return 1.0 if any(x in gtset for x in reck) else 0.0

def dcg_at_k(gt_list, rec_list, k):
    gtset = set(str(x) for x in (gt_list or []))
    dcg = 0.0
    for i, it in enumerate(rec_list[:k]):
        if str(it) in gtset:
            dcg += 1.0 / math.log2(i+2)
    return dcg

def idcg_at_k(gt_list, k):
    rel = min(len(gt_list or []), k)
    idcg = sum(1.0 / math.log2(i+2) for i in range(rel))
    return idcg

def ndcg_at_k(gt_list, rec_list, k):
    idcg = idcg_at_k(gt_list, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(gt_list, rec_list, k) / idcg

def main():
    R = load_json(RESULTS)
    Graw = load_json(GT)
    if isinstance(Graw, dict) and "_synthetic" in Graw and "data" in Graw:
        GTmap = Graw["data"]
    else:
        GTmap = Graw

    users = sorted(list(R.keys()))
    hr_vals = []
    ndcg_vals = []
    hits = []
    for u in users:
        recs = normalize_recs_list(R[u])
        gt = GTmap.get(str(u), [])
        if gt and isinstance(gt[0], dict):
            gt = [str(x.get("item_id")) for x in gt]
        hr = hr_at_k(gt, recs, K)
        ndcg = ndcg_at_k(gt, recs, K)
        if hr > 0:
            hits.append((u, list(set(gt) & set(recs))))
        hr_vals.append(hr)
        ndcg_vals.append(ndcg)
        print(f"user={u} hr@{K}={hr:.3f} ndcg@{K}={ndcg:.3f} gt_count={len(gt)} rec_topk={recs[:K]}")

    import numpy as np
    print("\nSUMMARY")
    print("n_users:", len(users))
    print("hr_mean:", float(np.mean(hr_vals)))
    print("hr_std:", float(np.std(hr_vals)))
    print("ndcg_mean:", float(np.mean(ndcg_vals)))
    print("ndcg_std:", float(np.std(ndcg_vals)))
    print("hits count users:", len(hits))
    print("examples hits (up to 10):")
    for h in hits[:10]:
        print(h)

if __name__ == '__main__':
    main()
