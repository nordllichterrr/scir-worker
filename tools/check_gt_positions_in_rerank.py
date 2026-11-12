
import json
from pathlib import Path
from collections import defaultdict

R = json.load(open("results/demo_rankings.json", encoding="utf-8"))
Graw = json.load(open("experiments/ground_truth.json", encoding="utf-8"))
GT = Graw.get("data", Graw) if isinstance(Graw, dict) else Graw

def norm_recs(recs):
    if not recs: return []
    if isinstance(recs[0], dict):
        return [str(x.get("item_id") or x.get("id") or x.get("doc_id")) for x in recs]
    return [str(x) for x in recs]

pos_counts = defaultdict(int)
pos_examples = defaultdict(list)

for u, recs in R.items():
    rec_ids = norm_recs(recs)
    gt_raw = GT.get(str(u), [])
    if gt_raw and isinstance(gt_raw[0], dict):
        gt_ids = [str(x.get("item_id") or x.get("id")) for x in gt_raw]
    else:
        gt_ids = [str(x) for x in (gt_raw or [])]
    for g in gt_ids:
        if g in rec_ids:
            pos = rec_ids.index(g) + 1
            pos_counts[pos] += 1
            if len(pos_examples[pos]) < 5:
                pos_examples[pos].append((u, g))
            break 

print("Position counts of GT within reranked lists (positions start at 1):")
for pos in sorted(pos_counts):
    print(f" pos {pos}: {pos_counts[pos]}")
print("Examples:")
for pos in sorted(pos_examples):
    print(pos, pos_examples[pos])
