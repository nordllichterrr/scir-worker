
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import random
from src.run_experiment import run_experiment
from src.run_logger import compute_diagnostics
from src.evaluate_results import load_gt_struct
from src.evaluation_config import N_SANITY_USERS

def main():
    print("=" * 60)
    print("SANITY TEST: Recall@K Calculation")
    print("=" * 60)
    print("\nTesting with %d users (sanity only; paper uses 1000)" % N_SANITY_USERS)
    print("We'll manually verify recall@K for each baseline\n")
    

    seed = 42
    random.seed(seed)
    
    baselines = ["random", "popularity", "embedding_cosine"]
    results_by_baseline = {}
    
    for baseline in baselines:
        print(f"\n{'='*60}")
        print(f"Running baseline: {baseline}")
        print(f"{'='*60}")
        
        config = {
            "baseline": baseline,
            "use_reranker": False,
            "topk": 10,
            "dataset": "serendipity",
        }
        
        results_meta = run_experiment(
            n_users=N_SANITY_USERS,
            seed=seed,
            config=config,
            dataset="serendipity"
        )
        
        results = results_meta["results"]
        candidate_pools = results_meta.get("candidate_pools", {})
        
        gt_wrapper = load_gt_struct()
        gt = gt_wrapper["data"]
        
        diagnostics = compute_diagnostics(results, gt, candidate_pools)
        
        results_by_baseline[baseline] = {
            "results": results,
            "candidate_pools": candidate_pools,
            "diagnostics": diagnostics
        }
        
        print(f"\nResults for {baseline}:")
        print(f"  Recall@50:  {diagnostics.get('recall@50'):.4f}")
        print(f"  Recall@200: {diagnostics.get('recall@200'):.4f}")
        print(f"  Recall@1000: {diagnostics.get('recall@1000'):.4f}")
        
        if results:
            first_uid = list(results.keys())[0]
            print(f"\n  First user ID: {first_uid}")
            
            gt_items = gt.get(first_uid, [])
            gt_ids = set(str(x) for x in gt_items)
            print(f"  GT items ({len(gt_ids)}): {sorted(list(gt_ids))[:10]}...")
            
            candidates = candidate_pools.get(first_uid, [])
            print(f"  Candidate pool size: {len(candidates)}")
            print(f"  Top-10 candidates: {candidates[:10]}")
            
            for k in [50, 200, 1000]:
                topk_candidates = set(candidates[:k])
                relevant_in_topk = len(gt_ids & topk_candidates)
                recall_k = relevant_in_topk / len(gt_ids) if len(gt_ids) > 0 else 0.0
                print(f"  Manual Recall@{k}: {relevant_in_topk}/{len(gt_ids)} = {recall_k:.4f}")
    

    print(f"\n{'='*60}")
    print("COMPARISON: Recall@K across baselines")
    print(f"{'='*60}")
    print(f"{'Baseline':<20} {'Recall@50':<12} {'Recall@200':<12} {'Recall@1000':<12}")
    print("-" * 60)
    for baseline in baselines:
        diag = results_by_baseline[baseline]["diagnostics"]
        r50 = diag.get('recall@50', 0)
        r200 = diag.get('recall@200', 0)
        r1000 = diag.get('recall@1000', 0)
        print(f"{baseline:<20} {r50:<12.4f} {r200:<12.4f} {r1000:<12.4f}")
    
    print(f"\n{'='*60}")
    print("VERIFICATION: Are baselines different?")
    print(f"{'='*60}")
    
    r50_values = [results_by_baseline[b]["diagnostics"].get('recall@50', 0) for b in baselines]
    r200_values = [results_by_baseline[b]["diagnostics"].get('recall@200', 0) for b in baselines]
    r1000_values = [results_by_baseline[b]["diagnostics"].get('recall@1000', 0) for b in baselines]
    
    if len(set(r50_values)) == 1:
        print("WARNING: All baselines have the same Recall@50!")
    else:
        print("OK: Recall@50 values are different across baselines")
    
    if len(set(r200_values)) == 1:
        print("WARNING: All baselines have the same Recall@200!")
    else:
        print("OK: Recall@200 values are different across baselines")
    
    if len(set(r1000_values)) == 1:
        print("WARNING: All baselines have the same Recall@1000!")
    else:
        print("OK: Recall@1000 values are different across baselines")
    
    print(f"\n{'='*60}")
    print("DETAILED: First user candidate pools comparison")
    print(f"{'='*60}")
    
    if results_by_baseline[baselines[0]]["results"]:
        first_uid = list(results_by_baseline[baselines[0]]["results"].keys())[0]
        print(f"User ID: {first_uid}\n")
        
        for baseline in baselines:
            candidates = results_by_baseline[baseline]["candidate_pools"].get(first_uid, [])
            print(f"{baseline}:")
            print(f"  Top-20: {candidates[:20]}")
            print()
    
    print("=" * 60)
    print("Sanity test complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
