
import math
import numpy as np

def hr_at_k(recommended: list, ground_truth: set, k=10):
    topk = [r["item_id"] for r in recommended[:k]]
    hits = sum(1 for it in topk if str(it) in set(map(str, ground_truth)))
    return hits / len(ground_truth) if len(ground_truth)>0 else 0.0

def ndcg_at_k(recommended: list, ground_truth: set, k=10):
    topk = [r["item_id"] for r in recommended[:k]]
    dcg = 0.0
    for i, it in enumerate(topk):
        rel = 1.0 if str(it) in set(map(str, ground_truth)) else 0.0
        denom = math.log2(i+2)
        dcg += (2**rel - 1) / denom

    ideal_rels = [1.0]*min(len(ground_truth), k)
    idcg = 0.0
    for i, rel in enumerate(ideal_rels):
        idcg += (2**rel - 1) / math.log2(i+2)
    return dcg / idcg if idcg > 0 else 0.0

def coverage_at_k(recommendations_all_users: dict, k=10):
    items = set()
    for user, recs in recommendations_all_users.items():
        for r in recs[:k]:
            items.add(str(r["item_id"]))
    return len(items)

def mae(preds, truths):
    preds = np.array(preds)
    truths = np.array(truths)
    return float(np.mean(np.abs(preds - truths)))

def rmse(preds, truths):
    preds = np.array(preds)
    truths = np.array(truths)
    return float(np.sqrt(np.mean((preds - truths)**2)))

def pearson(preds, truths):
    import numpy as np
    if len(preds) == 0:
        return 0.0
    p = np.corrcoef(preds, truths)[0,1]
    if np.isnan(p):
        return 0.0
    return float(p)
