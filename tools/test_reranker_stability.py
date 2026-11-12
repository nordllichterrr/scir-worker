import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
from src.run_experiment import run_experiment

def test_stability():
    print("=" * 70)
    print("RERANKER STABILITY TEST")
    print("=" * 70)
    
    config = {
        "baseline": None,
        "use_reranker": True,
        "topk": 10,
        "candidate_pool_size": 100,
        "rerank_pool_size": 100,
        "dataset": "serendipity",
    }
    
    print("\nRunning experiment twice with same seed (seed=42)...")
    results1_meta = run_experiment(n_users=5, seed=42, config=config, dataset="serendipity")
    results2_meta = run_experiment(n_users=5, seed=42, config=config, dataset="serendipity")
    
    results1 = results1_meta["results"]
    results2 = results2_meta["results"]

    print("\nComparing results...")
    differences = []
    
    for uid in results1.keys():
        if uid not in results2:
            differences.append(f"User {uid} missing in run 2")
            continue
        
        recs1 = results1[uid]
        recs2 = results2[uid]
        
        ids1 = [r.get("item_id") for r in recs1 if r.get("item_id")]
        ids2 = [r.get("item_id") for r in recs2 if r.get("item_id")]
        
        if ids1 != ids2:
            differences.append({
                "uid": uid,
                "run1": ids1[:5],
                "run2": ids2[:5]
            })
    
    if differences:
        print(f"WARNING: Found {len(differences)} differences between runs")
        for diff in differences[:5]:
            if isinstance(diff, dict):
                print(f"  User {diff['uid']}:")
                print(f"    Run 1: {diff['run1']}")
                print(f"    Run 2: {diff['run2']}")
            else:
                print(f"  {diff}")
    else:
        print("PASSED: Results are stable across runs with same seed")
    
    return len(differences) == 0

if __name__ == "__main__":
    success = test_stability()
    sys.exit(0 if success else 1)
