import json
r = json.load(open("results/demo_rankings.json", "r", encoding="utf-8"))
g = json.load(open("experiments/ground_truth.json", "r", encoding="utf-8"))
gt = g.get("data", g)
print("results keys sample:", list(r.keys())[:10])
print("gt keys sample:", list(gt.keys())[:10])
print("intersection size:", len(set(r.keys()) & set(gt.keys())))
for k in list(r.keys())[:5]:
    v = r[k]
    print("user", k, "type(recs)=", type(v), "len=", len(v), "first=", (v[0] if v else None))
