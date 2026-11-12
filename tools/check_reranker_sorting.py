
import sys
from pathlib import Path


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
from src.evaluate_results import load_gt_struct

def check_reranker_sorting(run_id: str):

    print(f"Checking reranker sorting for: {run_id}")
    
    results_path = Path("results") / f"{run_id}.json"
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return
    
    with open(results_path, "r", encoding="utf-8") as f:
        results_data = json.load(f)
    
    results = results_data.get("results", {})
    reranker_scores_all = results_data.get("reranker_scores", {})
    
    if not reranker_scores_all:
        print("No reranker scores found")
        return

    gt_wrapper = load_gt_struct()
    gt = gt_wrapper["data"]
    
    sorting_issues = []
    correct_sortings = 0
    
    for uid, recs in results.items():
        if not recs or len(recs) < 2:
            continue
        

        scores = []
        for rec in recs:
            score = rec.get("score")
            if score is not None:
                scores.append(float(score))
        
        if len(scores) < 2:
            continue
        
        is_descending = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
        
        if not is_descending:
            sorting_issues.append({
                "uid": uid,
                "scores": scores[:5], 
                "issue": "Not sorted in descending order"
            })
        else:
            correct_sortings += 1
    
    print(f"\nSorting check results:")
    print(f"  Correctly sorted: {correct_sortings}")
    print(f"  Issues found: {len(sorting_issues)}")
    
    if sorting_issues:
        print(f"\n⚠️  WARNING: Found {len(sorting_issues)} users with incorrect sorting!")
        print("\nFirst 5 examples:")
        for issue in sorting_issues[:5]:
            print(f"  User {issue['uid']}: scores = {issue['scores']}")
    else:
        print("\nOK: All reranker results are correctly sorted (descending order)")
    
 
    print(f"\n{'='*60}")
    print("Score Statistics:")
    print(f"{'='*60}")
    
    relevant_scores = []
    irrelevant_scores = []
    
    for uid, scores_dict in reranker_scores_all.items():
        gt_items = set(str(x) for x in gt.get(str(uid), []))
        
        for item_id, score in scores_dict.items():
            if item_id in gt_items:
                relevant_scores.append(score)
            else:
                irrelevant_scores.append(score)
    
    if relevant_scores and irrelevant_scores:
        import numpy as np
        rel_mean = np.mean(relevant_scores)
        irrel_mean = np.mean(irrelevant_scores)
        
        print(f"Relevant items:   mean = {rel_mean:.4f}, n = {len(relevant_scores)}")
        print(f"Irrelevant items: mean = {irrel_mean:.4f}, n = {len(irrelevant_scores)}")
        print(f"\nDifference: {rel_mean - irrel_mean:.4f}")
        
        if rel_mean > irrel_mean:
            print("OK: Relevant items have HIGHER scores (correct)")
            print("  -> Should sort in DESCENDING order (reverse=True)")
        else:
            print("WARNING: Relevant items have LOWER scores (unexpected)")
            print("  -> May need to sort in ASCENDING order or invert scores")
    
    return len(sorting_issues) == 0

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python tools/check_reranker_sorting.py <run_id>")
        print("\nExample:")
        print("  python tools/check_reranker_sorting.py ablation_with_reranker_serendipity_seed42_n500")
        sys.exit(1)
    
    run_id = sys.argv[1]
    check_reranker_sorting(run_id)
