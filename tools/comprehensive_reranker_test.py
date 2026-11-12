
import sys
from pathlib import Path


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import random
from src.run_experiment import run_experiment
from src.evaluate_results import load_gt_struct, evaluate_single

def check_sorting_assertion(items, name="items"):
    scores = []
    for item in items:
        if isinstance(item, dict):
            score = item.get("score")
            if score is not None:
                scores.append(float(score))
        else:
            return True  
    
    if len(scores) < 2:
        return True
    

    for i in range(len(scores) - 1):
        if scores[i] < scores[i+1]:
            print(f"ERROR: {name} not sorted correctly at position {i}")
            print(f"  scores[{i}] = {scores[i]}, scores[{i+1}] = {scores[i+1]}")
            return False
    return True

def check_id_score_match(reranked_full, candidates_to_rerank):

    reranked_ids = [item.get("item_id") for item in reranked_full if item.get("item_id")]
    candidates_set = set(str(c) for c in candidates_to_rerank)
    reranked_set = set(str(id) for id in reranked_ids)

    extra_ids = reranked_set - candidates_set
    if extra_ids:
        print(f"WARNING: Found {len(extra_ids)} IDs in reranked that weren't in candidates")
        return False
    
    return True

def check_no_deduplication_after_rerank(reranked_full):

    ids = [item.get("item_id") for item in reranked_full if item.get("item_id")]
    if len(ids) != len(set(ids)):
        print("ERROR: Duplicate IDs found in reranked results (deduplication may have broken order)")
        return False
    return True

def check_final_results_integrity(results, reranker_scores_all):

    issues = []
    
    for uid, recs in results.items():
        if not recs or len(recs) < 2:
            continue
        

        final_scores = []
        for rec in recs:
            score = rec.get("score")
            if score is not None:
                final_scores.append(float(score))
        
        if len(final_scores) < 2:
            continue
        
        for i in range(len(final_scores) - 1):
            if final_scores[i] < final_scores[i+1]:
                issues.append({
                    "uid": uid,
                    "position": i,
                    "score_i": final_scores[i],
                    "score_i+1": final_scores[i+1]
                })
    
    return issues

def test_reranker_integrity():

    print("=" * 70)
    print("COMPREHENSIVE RERANKER INTEGRITY TEST")
    print("=" * 70)
    
    seed = 42
    random.seed(seed)
    
    config = {
        "baseline": None,
        "use_reranker": True,
        "topk": 10,
        "candidate_pool_size": 1000,
        "rerank_pool_size": 500,
        "dataset": "serendipity",
    }
    
    print("\nRunning experiment with reranker...")
    results_meta = run_experiment(
        n_users=10,  
        seed=seed,
        config=config,
        dataset="serendipity"
    )
    
    results = results_meta["results"]
    reranker_scores_all = results_meta.get("reranker_scores", {})
    
    print(f"\nTested {len(results)} users")
    

    print("\n" + "=" * 70)
    print("TEST 1: Final Results Sorting")
    print("=" * 70)
    
    final_issues = check_final_results_integrity(results, reranker_scores_all)
    if final_issues:
        print(f"FAILED: Found {len(final_issues)} users with incorrect sorting in final results")
        for issue in final_issues[:5]:
            print(f"  User {issue['uid']}: score[{issue['position']}] = {issue['score_i']:.4f} < score[{issue['position']+1}] = {issue['score_i+1']:.4f}")
    else:
        print("PASSED: All final results are correctly sorted")
    

    print("\n" + "=" * 70)
    print("TEST 2: ID-Score Consistency in Final Results")
    print("=" * 70)
    
    id_score_issues = []
    for uid, recs in results.items():
        user_scores = reranker_scores_all.get(uid, {})
        if not user_scores:
            continue
        
        for rec in recs:
            item_id = rec.get("item_id")
            final_score = rec.get("score")
            
            if item_id and final_score is not None:
                stored_score = user_scores.get(str(item_id))
                if stored_score is not None:
                    if abs(float(final_score) - float(stored_score)) > 1e-6:
                        id_score_issues.append({
                            "uid": uid,
                            "item_id": item_id,
                            "final_score": final_score,
                            "stored_score": stored_score
                        })
    
    if id_score_issues:
        print(f"FAILED: Found {len(id_score_issues)} ID-score mismatches")
        for issue in id_score_issues[:5]:
            print(f"  User {issue['uid']}, Item {issue['item_id']}: final={issue['final_score']:.4f}, stored={issue['stored_score']:.4f}")
    else:
        print("PASSED: All ID-score pairs are consistent")
    

    print("\n" + "=" * 70)
    print("TEST 3: Deduplication Check")
    print("=" * 70)
    
    dedup_issues = []
    for uid, recs in results.items():
        ids = [rec.get("item_id") for rec in recs if rec.get("item_id")]
        if len(ids) != len(set(ids)):
            dedup_issues.append({
                "uid": uid,
                "total": len(ids),
                "unique": len(set(ids))
            })
    
    if dedup_issues:
        print(f"WARNING: Found {len(dedup_issues)} users with duplicate IDs (may indicate deduplication)")
        for issue in dedup_issues[:5]:
            print(f"  User {issue['uid']}: {issue['total']} total, {issue['unique']} unique")
    else:
        print("PASSED: No duplicate IDs found")
    

    print("\n" + "=" * 70)
    print("TEST 4: A/B Sanity Test (reverse=True vs reverse=False)")
    print("=" * 70)
  
    gt_wrapper = load_gt_struct()
    gt = gt_wrapper["data"]
    

    rows_normal, summary_normal = evaluate_single(results, gt, k=10)
    hr_normal = summary_normal.get("hr_mean", 0)
    ndcg_normal = summary_normal.get("ndcg_mean", 0)
    
   
    results_reversed = {}
    for uid, recs in results.items():
        if recs and len(recs) > 1:
         
            results_reversed[uid] = list(reversed(recs))
        else:
            results_reversed[uid] = recs
    
    rows_reversed, summary_reversed = evaluate_single(results_reversed, gt, k=10)
    hr_reversed = summary_reversed.get("hr_mean", 0)
    ndcg_reversed = summary_reversed.get("ndcg_mean", 0)
    
    print(f"Normal (descending):  HR@10 = {hr_normal:.4f}, nDCG@10 = {ndcg_normal:.4f}")
    print(f"Reversed (ascending): HR@10 = {hr_reversed:.4f}, nDCG@10 = {ndcg_reversed:.4f}")
    print(f"Difference: HR = {hr_normal - hr_reversed:.4f}, nDCG = {ndcg_normal - ndcg_reversed:.4f}")
    
    if hr_normal > hr_reversed and ndcg_normal > ndcg_reversed:
        print("PASSED: Descending order gives better metrics (reranker is working)")
    elif abs(hr_normal - hr_reversed) < 0.001 and abs(ndcg_normal - ndcg_reversed) < 0.001:
        print("WARNING: No difference between descending and ascending (reranker may not be working)")
    else:
        print("FAILED: Descending order gives worse metrics (possible sorting issue)")
    

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    all_passed = (
        len(final_issues) == 0 and
        len(id_score_issues) == 0 and
        hr_normal > hr_reversed
    )
    
    if all_passed:
        print("ALL TESTS PASSED: Reranker ranking integrity is maintained")
    else:
        print("SOME TESTS FAILED: Review the issues above")
    
    return all_passed

if __name__ == "__main__":
    success = test_reranker_integrity()
    sys.exit(0 if success else 1)
