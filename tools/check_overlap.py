import json
r = json.load(open("results/demo_rankings.json", "r", encoding="utf-8"))
g = json.load(open("experiments/ground_truth.json", "r", encoding="utf-8"))
gt = g.get("data", g)
count_users = 0
count_with_overlap = 0
total_overlaps = 0
for uid, recs in r.items():
    rec_ids = [str(x.get("item_id")) if isinstance(x, dict) else str(x) for x in recs]
    pos = gt.get(str(uid), [])
    if pos is None: pos = []
    overlap = set(rec_ids) & set(map(str, pos))
    count_users += 1
    if overlap:
        count_with_overlap += 1
        total_overlaps += len(overlap)
        print("HIT user", uid, "overlap_count=", len(overlap), "examples=", list(overlap)[:5])
print("users_checked:", count_users)
print("users_with_overlap:", count_with_overlap)
print("total_overlap_items:", total_overlaps)
