
import json
from pathlib import Path
from collections import defaultdict, Counter

RESULTS_DIR = Path("results")
GT_PATH = Path("experiments") / "ground_truth.json"
CAND_PREFIX = "candidates_" 

def load_gt():
    raw = json.load(open(GT_PATH, encoding="utf-8"))
    if isinstance(raw, dict) and "data" in raw:
        return raw["data"]
    return raw

def norm_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        if not x: return []
        first = x[0]
        if isinstance(first, dict):
            return [str(d.get("item_id") or d.get("id") or d.get("doc_id")) for d in x]
        return [str(i) for i in x]
    if isinstance(x, dict):
        v = x.get("item_id") or x.get("id") or x.get("doc_id")
        return [str(v)] if v is not None else []
    return [str(x)]

def main():
    gt = load_gt()
    cand_files = sorted(RESULTS_DIR.glob(f"{CAND_PREFIX}*.json"))
    total_users = 0
    users_with_any = 0
    per_user_positions = {}
    recall_at = {50:0, 200:0, 1000:0}
    missing_gt_ids = set()
    for p in cand_files:
        uid = p.stem.replace(CAND_PREFIX, "")
        total_users += 1
        try:
            cand = json.load(open(p, encoding="utf-8"))
        except Exception as e:
            print("fail load", p, e)
            continue
        cand_ids = norm_list(cand)
        raw_gt = gt.get(str(uid), [])
        if raw_gt and isinstance(raw_gt, list) and isinstance(raw_gt[0], dict):
            gt_ids = [str(d.get("item_id") or d.get("id") or d.get("doc_id")) for d in raw_gt]
        else:
            gt_ids = [str(x) for x in (raw_gt or [])]
        if not gt_ids:
            continue
        found_any = False
        pos_list = []
        for g in gt_ids:
            if g in cand_ids:
                found_any = True
                pos = cand_ids.index(g) 
                pos_list.append((g, pos+1))
        if found_any:
            users_with_any += 1
            per_user_positions[uid] = pos_list

        for k in recall_at.keys():
            topk = set(cand_ids[:k])
            if set(gt_ids) & topk:
                recall_at[k] += 1
    print("candidate files checked:", total_users)
    print("users with any GT in candidate pool:", users_with_any)
    print("recall@Ks (users hit count):")
    for k,v in recall_at.items():
        print(f"  recall@{k}: {v} / {total_users}  (≈ {v/total_users:.3f})")
    print("\nExamples of found GT positions (up to 10 users):")
    for i,(u,ps) in enumerate(per_user_positions.items()):
        if i>=10: break
        print(u, ps)
    gt_keys = set(str(k) for k in gt.keys())
    cand_uids = set(p.stem.replace(CAND_PREFIX,"") for p in cand_files)
    missing_cands = sorted(list(gt_keys - cand_uids))[:10]
    if missing_cands:
        print("\nGT users missing candidate file (example):", missing_cands[:10])

if __name__ == "__main__":
    main()
