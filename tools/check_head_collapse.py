
import sys
from pathlib import Path


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
from collections import Counter
from src.evaluate_results import load_gt_struct

def check_head_collapse(run_id: str):
    print("=" * 70)
    print(f"HEAD COLLAPSE CHECK: {run_id}")
    print("=" * 70)
    
    results_path = Path("results") / f"{run_id}.json"
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return
    
    with open(results_path, "r", encoding="utf-8") as f:
        results_data = json.load(f)
    
    results = results_data.get("results", {})
    
    if not results:
        print("No results found")
        return
    
    top1_items = []
    top10_items = []
    
    for uid, recs in results.items():
        if recs and len(recs) > 0:
            top1_items.append(str(recs[0].get("item_id")))
        if recs and len(recs) >= 10:
            top10_list = [str(r.get("item_id")) for r in recs[:10]]
            top10_items.append(tuple(top10_list))
    

    top1_counter = Counter(top1_items)
    unique_top1 = len(top1_counter)
    total_users = len(top1_items)
    
    print(f"\nTop-1 Analysis:")
    print(f"  Total users: {total_users}")
    print(f"  Unique top-1 items: {unique_top1}")
    print(f"  Diversity: {unique_top1 / total_users:.2%}")
    
    if unique_top1 < total_users * 0.1: 
        print(f"  WARNING: Head collapse detected! Only {unique_top1} unique top-1 items for {total_users} users")
    
    print(f"\nMost common top-1 items:")
    for item_id, count in top1_counter.most_common(10):
        print(f"  {item_id}: {count} users ({count/total_users:.1%})")
    

    top10_counter = Counter(top10_items)
    unique_top10 = len(top10_counter)
    
    print(f"\nTop-10 Analysis:")
    print(f"  Unique top-10 sequences: {unique_top10}")
    print(f"  Diversity: {unique_top10 / len(top10_items):.2%}")
    
    if unique_top10 < len(top10_items) * 0.1:
        print(f"  WARNING: Head collapse in top-10! Only {unique_top10} unique sequences")
    

    print(f"\nMost common top-10 sequences:")
    for seq, count in top10_counter.most_common(5):
        print(f"  Count: {count} users")
        print(f"  Items: {' '.join(seq[:5])}...")
    

    all_items = []
    for recs in results.values():
        for rec in recs[:10]:
            all_items.append(str(rec.get("item_id")))
    
    item_counter = Counter(all_items)
    print(f"\nItems appearing in top-10 (most frequent):")
    for item_id, count in item_counter.most_common(10):
        print(f"  {item_id}: {count} times ({count/len(all_items):.1%})")
    
    return {
        "unique_top1": unique_top1,
        "total_users": total_users,
        "diversity_top1": unique_top1 / total_users if total_users > 0 else 0,
        "unique_top10": unique_top10,
        "diversity_top10": unique_top10 / len(top10_items) if top10_items else 0,
        "most_common_top1": dict(top1_counter.most_common(5))
    }

def check_seen_items_filtering(run_id: str):
    """Check if seen items are being filtered incorrectly."""
    print("\n" + "=" * 70)
    print("SEEN ITEMS FILTERING CHECK")
    print("=" * 70)
    

    results_path = Path("results") / f"{run_id}.json"
    if not results_path.exists():
        return
    
    with open(results_path, "r", encoding="utf-8") as f:
        results_data = json.load(f)
    
    results = results_data.get("results", {})
    candidate_pools = results_data.get("candidate_pools", {})
    

    gt_wrapper = load_gt_struct()
    gt = gt_wrapper["data"]
    

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
    
    print(f"\nGT Items Analysis:")
    print(f"  Total GT items across all users: {total_gt_items}")
    print(f"  GT items in candidate pools: {gt_in_candidates} ({gt_in_candidates/total_gt_items:.1%})")
    print(f"  GT items in final recommendations: {gt_in_recs} ({gt_in_recs/total_gt_items:.1%})")
    print(f"  Users with GT in candidates: {users_with_gt_in_candidates}")
    print(f"  Users with GT in recommendations: {users_with_gt_in_recs}")
    
    if gt_in_candidates == 0:
        print(f"\n  ERROR: No GT items in candidate pools! Possible filtering issue.")
    elif gt_in_candidates < total_gt_items * 0.5:
        print(f"\n  WARNING: Only {gt_in_candidates/total_gt_items:.1%} of GT items in candidate pools")
    
    if gt_in_recs == 0 and gt_in_candidates > 0:
        print(f"\n  WARNING: GT items are in candidates but not in recommendations (reranker issue?)")
    
    return {
        "gt_in_candidates": gt_in_candidates,
        "gt_in_recs": gt_in_recs,
        "total_gt": total_gt_items,
        "coverage_candidates": gt_in_candidates / total_gt_items if total_gt_items > 0 else 0,
        "coverage_recs": gt_in_recs / total_gt_items if total_gt_items > 0 else 0
    }

def check_user_profile_diversity():
    print("\n" + "=" * 70)
    print("USER PROFILE DIVERSITY CHECK")
    print("=" * 70)
    
    from src.vark_simulator import build_user_profile_from_minimal
    

    profiles = []
    for i in range(20):
        info = {
            "user_id": f"test_user_{i}",
            "goal": "",  
            "time_of_day": None,
            "session_len": None
        }
        profile = build_user_profile_from_minimal(info)
        profiles.append(profile)

    vark_values = [p.get("vark") for p in profiles]
    vark_counter = Counter(vark_values)
    
    print(f"\nVARK Diversity (20 test users):")
    print(f"  Unique VARK values: {len(vark_counter)}")
    for vark, count in vark_counter.items():
        print(f"  {vark}: {count} users")
    

    text_profiles = [p.get("text_profile", "") for p in profiles]
    unique_text = len(set(text_profiles))
    
    print(f"\nText Profile Diversity:")
    print(f"  Unique text profiles: {unique_text} / {len(text_profiles)}")
    
    if unique_text == 1:
        print(f"  ERROR: All users have the same text_profile! This causes head collapse.")
    elif unique_text < len(text_profiles) * 0.5:
        print(f"  WARNING: Low text_profile diversity ({unique_text}/{len(text_profiles)})")
    
    return {
        "vark_diversity": len(vark_counter),
        "text_profile_diversity": unique_text,
        "total_users": len(profiles)
    }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python tools/check_head_collapse.py <run_id>")
        print("\nExample:")
        print("  python tools/check_head_collapse.py ablation_with_reranker_serendipity_seed42_n500")
        sys.exit(1)
    
    run_id = sys.argv[1]
    

    collapse_info = check_head_collapse(run_id)
    

    filtering_info = check_seen_items_filtering(run_id)
 
    profile_info = check_user_profile_diversity()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    issues = []
    if collapse_info and collapse_info.get("diversity_top1", 1) < 0.1:
        issues.append("Head collapse detected (top-1 diversity < 10%)")
    if filtering_info and filtering_info.get("coverage_candidates", 1) < 0.5:
        issues.append("GT items missing from candidate pools")
    if profile_info and profile_info.get("text_profile_diversity", 1) == 1:
        issues.append("All users have identical text profiles")
    
    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("No major issues detected")
