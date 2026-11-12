
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import random
from src.run_experiment import run_experiment
from src.run_logger import compute_diagnostics
from src.evaluate_results import load_gt_struct

def main():
    print("=" * 60)
    print("DEBUG: Random Baseline Recall@K")
    print("=" * 60)
    
    seed = 42
    random.seed(seed)
    
    config = {
        "baseline": "random",
        "use_reranker": False,
        "topk": 10,
        "dataset": "serendipity",
    }
    

    results_meta = run_experiment(
        n_users=5,
        seed=seed,
        config=config,
        dataset="serendipity"
    )
    
    results = results_meta["results"]
    candidate_pools = results_meta.get("candidate_pools", {})
    

    gt_wrapper = load_gt_struct()
    gt = gt_wrapper["data"]

    if candidate_pools:
        first_uid = list(candidate_pools.keys())[0]
        catalog_size = len(candidate_pools[first_uid])
        print(f"\nCatalog size (from candidate_pool): {catalog_size}")
        print(f"Expected: ~49157")
    

    diagnostics = compute_diagnostics(results, gt, candidate_pools)
    
    print(f"\nOverall Recall@K (averaged across users):")
    print(f"  Recall@50:  {diagnostics.get('recall@50'):.6f}")
    print(f"  Recall@200: {diagnostics.get('recall@200'):.6f}")
    print(f"  Recall@1000: {diagnostics.get('recall@1000'):.6f}")
    

    if catalog_size:
        expected_50 = 50 / catalog_size
        expected_200 = 200 / catalog_size
        expected_1000 = 1000 / catalog_size
        print(f"\nExpected Recall@K (K / N):")
        print(f"  Recall@50:  {expected_50:.6f} (50 / {catalog_size})")
        print(f"  Recall@200: {expected_200:.6f} (200 / {catalog_size})")
        print(f"  Recall@1000: {expected_1000:.6f} (1000 / {catalog_size})")
    

    print(f"\n{'='*60}")
    print("Per-user Recall@K analysis:")
    print(f"{'='*60}")
    print(f"{'User ID':<12} {'GT size':<10} {'Pool size':<12} {'R@50':<10} {'R@200':<10} {'R@1000':<10}")
    print("-" * 70)
    
    per_user_recalls = []
    for uid in sorted(results.keys()):
        gt_items = gt.get(uid, [])
        if not gt_items:
            continue
        candidates = candidate_pools.get(uid, [])
        if not candidates:
            continue
        
        gt_ids = set(str(x) for x in gt_items)
        total_relevant = len(gt_ids)
        pool_size = len(candidates)
        
        cand_list_50 = [str(x) for x in candidates[:50]]
        cand_list_200 = [str(x) for x in candidates[:200]]
        cand_list_1000 = [str(x) for x in candidates[:1000]]
        
        relevant_in_50 = len(gt_ids & set(cand_list_50))
        relevant_in_200 = len(gt_ids & set(cand_list_200))
        relevant_in_1000 = len(gt_ids & set(cand_list_1000))
        
        recall_50 = relevant_in_50 / total_relevant if total_relevant > 0 else 0.0
        recall_200 = relevant_in_200 / total_relevant if total_relevant > 0 else 0.0
        recall_1000 = relevant_in_1000 / total_relevant if total_relevant > 0 else 0.0
        
        per_user_recalls.append({
            'uid': uid,
            'gt_size': total_relevant,
            'pool_size': pool_size,
            'recall_50': recall_50,
            'recall_200': recall_200,
            'recall_1000': recall_1000,
            'relevant_in_50': relevant_in_50,
            'relevant_in_200': relevant_in_200,
            'relevant_in_1000': relevant_in_1000,
        })
        
        print(f"{uid:<12} {total_relevant:<10} {pool_size:<12} {recall_50:<10.6f} {recall_200:<10.6f} {recall_1000:<10.6f}")

    pool_sizes = [r['pool_size'] for r in per_user_recalls]
    if len(set(pool_sizes)) > 1:
        print(f"\nWARNING: Different pool sizes across users: {set(pool_sizes)}")
    else:
        print(f"\nOK: All users have the same pool size: {pool_sizes[0] if pool_sizes else 'N/A'}")
    
    print(f"\n{'='*60}")
    print("Expected vs Actual Recall@K per user:")
    print(f"{'='*60}")
    if pool_sizes:
        N = pool_sizes[0]
        print(f"\nFor each user, expected recall@K = K / N = K / {N}")
        print(f"{'User ID':<12} {'GT size':<10} {'Expected R@50':<15} {'Actual R@50':<15} {'Expected R@200':<15} {'Actual R@200':<15}")
        print("-" * 85)
        
        for r in per_user_recalls:
            expected_50 = 50 / N
            expected_200 = 200 / N
            actual_50 = r['recall_50']
            actual_200 = r['recall_200']
            
            print(f"{r['uid']:<12} {r['gt_size']:<10} {expected_50:<15.6f} {actual_50:<15.6f} {expected_200:<15.6f} {actual_200:<15.6f}")
    
    print(f"\n{'='*60}")
    print("Global vs Per-user averaging:")
    print(f"{'='*60}")
    

    avg_recall_50 = sum(r['recall_50'] for r in per_user_recalls) / len(per_user_recalls) if per_user_recalls else 0
    avg_recall_200 = sum(r['recall_200'] for r in per_user_recalls) / len(per_user_recalls) if per_user_recalls else 0
    avg_recall_1000 = sum(r['recall_1000'] for r in per_user_recalls) / len(per_user_recalls) if per_user_recalls else 0
    

    total_gt = sum(r['gt_size'] for r in per_user_recalls)
    total_found_50 = sum(r['relevant_in_50'] for r in per_user_recalls)
    total_found_200 = sum(r['relevant_in_200'] for r in per_user_recalls)
    total_found_1000 = sum(r['relevant_in_1000'] for r in per_user_recalls)
    
    global_recall_50 = total_found_50 / total_gt if total_gt > 0 else 0
    global_recall_200 = total_found_200 / total_gt if total_gt > 0 else 0
    global_recall_1000 = total_found_1000 / total_gt if total_gt > 0 else 0
    
    print(f"Per-user average recall@50:  {avg_recall_50:.6f}")
    print(f"Global recall@50:              {global_recall_50:.6f}")
    print(f"\nPer-user average recall@200: {avg_recall_200:.6f}")
    print(f"Global recall@200:             {global_recall_200:.6f}")
    print(f"\nPer-user average recall@1000: {avg_recall_1000:.6f}")
    print(f"Global recall@1000:             {global_recall_1000:.6f}")
    
    print(f"\nFrom diagnostics (should be global):")
    print(f"Recall@50:  {diagnostics.get('recall@50'):.6f}")
    print(f"Recall@200: {diagnostics.get('recall@200'):.6f}")
    print(f"Recall@1000: {diagnostics.get('recall@1000'):.6f}")
    
    print(f"\nTotal GT items: {total_gt}")
    print(f"Total found in top-50: {total_found_50}")
    print(f"Total found in top-200: {total_found_200}")
    print(f"Total found in top-1000: {total_found_1000}")
    
    if pool_sizes:
        N = pool_sizes[0]
        expected_global = 50 / N
        print(f"\nExpected global recall@50 = 50 / {N} = {expected_global:.6f}")
        print(f"Actual global recall@50 = {global_recall_50:.6f}")
        if abs(global_recall_50 - expected_global) > 0.0005:
            print(f"DIFFERENCE: {abs(global_recall_50 - expected_global):.6f}")
        else:
            print("OK: Global recall@50 matches expected value!")

if __name__ == "__main__":
    main()
