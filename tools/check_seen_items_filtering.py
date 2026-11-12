
import sys
from pathlib import Path


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import csv
from collections import defaultdict
from src.evaluate_results import load_gt_struct

def load_training_interactions():
    train_path = Path("data/serendipity-sac2018/training.csv")
    if not train_path.exists():
        return {}
    
    train_items_by_user = defaultdict(set)
    with open(train_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row.get("user_id") or row.get("user") or row.get("uid")
            iid = row.get("item_id") or row.get("item") or row.get("itemId")
            if uid and iid:
                train_items_by_user[str(uid)].add(str(iid))
    
    return train_items_by_user

def check_seen_items_filtering(run_id: str):
    print("=" * 70)
    print(f"SEEN ITEMS FILTERING CHECK: {run_id}")
    print("=" * 70)
    

    results_path = Path("results") / f"{run_id}.json"
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return
    
    with open(results_path, "r", encoding="utf-8") as f:
        results_data = json.load(f)
    
    results = results_data.get("results", {})
    candidate_pools = results_data.get("candidate_pools", {})
    

    gt_wrapper = load_gt_struct()
    gt = gt_wrapper["data"]
    

    train_items_by_user = load_training_interactions()
    
    print(f"\nTraining data:")
    print(f"  Users in training: {len(train_items_by_user)}")
    if train_items_by_user:
        avg_train_items = sum(len(items) for items in train_items_by_user.values()) / len(train_items_by_user)
        print(f"  Avg items per user in training: {avg_train_items:.1f}")
    

    gt_in_training = 0
    gt_not_in_training = 0
    users_with_gt_in_training = 0
    
    for uid, gt_items in gt.items():
        train_items = train_items_by_user.get(uid, set())
        gt_set = set(str(x) for x in gt_items)
        
        overlap = gt_set & train_items
        if overlap:
            users_with_gt_in_training += 1
            gt_in_training += len(overlap)
        gt_not_in_training += len(gt_set - train_items)
    
    print(f"\nGT vs Training overlap:")
    print(f"  GT items in training: {gt_in_training}")
    print(f"  GT items NOT in training: {gt_not_in_training}")
    print(f"  Users with GT in training: {users_with_gt_in_training} / {len(gt)}")
    
    if gt_in_training > 0:
        print(f"\n  WARNING: {gt_in_training} GT items are from training set!")
        print(f"  This suggests train/test split issue, not seen items filtering.")
    

    gt_in_recs = 0
    gt_in_candidates = 0
    total_gt_items = 0
    users_with_gt_in_recs = 0
    users_with_gt_in_candidates = 0
    
    for uid, recs in results.items():
        gt_items = set(str(x) for x in gt.get(uid, []))
        if not gt_items:
            continue
        
        total_gt_items += len(gt_items)
        

        rec_ids = set(str(r.get("item_id")) for r in recs if r.get("item_id"))
        gt_in_recs_count = len(gt_items & rec_ids)
        if gt_in_recs_count > 0:
            users_with_gt_in_recs += 1
            gt_in_recs += gt_in_recs_count
        
        candidates = candidate_pools.get(uid, [])
        cand_set = set(str(c) for c in candidates)
        gt_in_candidates_count = len(gt_items & cand_set)
        if gt_in_candidates_count > 0:
            users_with_gt_in_candidates += 1
            gt_in_candidates += gt_in_candidates_count
    
    print(f"\nGT Items in Pipeline:")
    print(f"  Total GT items: {total_gt_items}")
    print(f"  GT in candidate pools: {gt_in_candidates} ({gt_in_candidates/total_gt_items:.1%})")
    print(f"  GT in recommendations: {gt_in_recs} ({gt_in_recs/total_gt_items:.1%})")
    print(f"  Users with GT in candidates: {users_with_gt_in_candidates}")
    print(f"  Users with GT in recommendations: {users_with_gt_in_recs}")
    
    print(f"\nFiltering Analysis:")
    
    if gt_in_candidates == 0:
        print(f"  ERROR: No GT items in candidate pools!")
        print(f"  Possible causes:")
        print(f"    - GT items filtered as 'seen items' (incorrect for cold-start)")
        print(f"    - GT items not in catalog")
        print(f"    - Retrieval not finding GT items")
    elif gt_in_candidates < total_gt_items * 0.1:
        print(f"  WARNING: Only {gt_in_candidates/total_gt_items:.1%} of GT items in candidate pools")
        print(f"  This is very low - possible retrieval issue")
    else:
        print(f"  OK: {gt_in_candidates/total_gt_items:.1%} of GT items in candidate pools")
    
    if gt_in_candidates > 0 and gt_in_recs == 0:
        print(f"  WARNING: GT items in candidates but not in recommendations")
        print(f"  Possible causes:")
        print(f"    - Reranker ranking them too low")
        print(f"    - Filtering after rerank")
    
    print(f"\nCold-Start Assumption:")
    print(f"  For cold-start, users should have NO training history")
    print(f"  Therefore, NO seen items filtering should be applied")
    
    if train_items_by_user:
        users_with_history = sum(1 for uid in results.keys() if uid in train_items_by_user and train_items_by_user[uid])
        print(f"  Users with training history: {users_with_history} / {len(results)}")
        if users_with_history > 0:
            print(f"  NOTE: Some users have training history, but this is OK for cold-start evaluation")
    
    return {
        "gt_in_training": gt_in_training,
        "gt_not_in_training": gt_not_in_training,
        "gt_in_candidates": gt_in_candidates,
        "gt_in_recs": gt_in_recs,
        "total_gt": total_gt_items,
        "coverage_candidates": gt_in_candidates / total_gt_items if total_gt_items > 0 else 0,
        "coverage_recs": gt_in_recs / total_gt_items if total_gt_items > 0 else 0
    }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python tools/check_seen_items_filtering.py <run_id>")
        print("\nExample:")
        print("  python tools/check_seen_items_filtering.py ablation_with_reranker_serendipity_seed42_n500")
        sys.exit(1)
    
    run_id = sys.argv[1]
    check_seen_items_filtering(run_id)
