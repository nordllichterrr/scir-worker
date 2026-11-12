import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

RESULTS_DIR = Path("results")
GT_PATH = Path("experiments") / "ground_truth.json"
OUTPUT_DIR = Path("experiments") / "score_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_gt():
    if not GT_PATH.exists():
        return {}
    with open(GT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data.get("data", data)


def load_results(run_id: str) -> Dict[str, Any]:
    results_path = RESULTS_DIR / f"{run_id}.json"
    if not results_path.exists():
        return {}
    
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_score_distributions(run_id: str, gt: Dict[str, List[str]]):
    print(f"\nAnalyzing scores for: {run_id}")
    
    results_data = load_results(run_id)
    if not results_data:
        print(f"Results not found for {run_id}")
        return
    
    reranker_scores = results_data.get("reranker_scores", {})
    candidate_pools = results_data.get("candidate_pools", {})
    
    if not reranker_scores:
        print(f"No reranker scores found for {run_id}")
        return
    
    relevant_scores = []
    irrelevant_scores = []
    all_scores = []
    per_user_auc = []
    per_user_top1_hit = []
    
    for uid, scores_dict in reranker_scores.items():
        uid_str = str(uid)
        gt_items = set(str(x) for x in gt.get(uid_str, []))
        candidates = candidate_pools.get(uid, [])
        if not gt_items or not candidates:
            continue
        cand_set = set(str(c) for c in candidates)
        if not (gt_items & cand_set):
            continue

        user_scores = []
        user_labels = []
        for item_id, score in scores_dict.items():
            item_id_str = str(item_id)
            label = 1 if item_id_str in gt_items else 0
            user_scores.append(float(score))
            user_labels.append(label)
            all_scores.append(float(score))
            if label == 1:
                relevant_scores.append(float(score))
            else:
                irrelevant_scores.append(float(score))

        n_pos = sum(user_labels)
        n_neg = len(user_labels) - n_pos
        if n_pos > 0 and n_neg > 0:
            ranks = stats.rankdata(user_scores, method="average")
            sum_ranks_pos = float(np.sum(ranks[np.array(user_labels) == 1]))
            auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
            per_user_auc.append(auc)

            best_idx = int(np.argmax(user_scores))
            top1_hit = 1 if user_labels[best_idx] == 1 else 0
            per_user_top1_hit.append(top1_hit)
    
    if not relevant_scores or not irrelevant_scores:
        print("Insufficient data for analysis")
        return
    
    print(f"\nScore Statistics (oracle candidates only):")
    print(f"  Relevant items: n={len(relevant_scores)}, mean={np.mean(relevant_scores):.4f}, std={np.std(relevant_scores):.4f}")
    print(f"  Irrelevant items: n={len(irrelevant_scores)}, mean={np.mean(irrelevant_scores):.4f}, std={np.std(irrelevant_scores):.4f}")
    
    t_stat, p_value = stats.ttest_ind(relevant_scores, irrelevant_scores)
    print(f"  T-test (relevant vs irrelevant): t={t_stat:.4f}, p={p_value:.4f}")
    
    correlation_data = []
    for uid, scores_dict in reranker_scores.items():
        gt_items = set(str(x) for x in gt.get(str(uid), []))
        for item_id, score in scores_dict.items():
            relevance = 1 if item_id in gt_items else 0
            correlation_data.append((score, relevance))
    
    if correlation_data:
        scores_arr = np.array([x[0] for x in correlation_data])
        relevance_arr = np.array([x[1] for x in correlation_data])
        spearman = stats.spearmanr(scores_arr, relevance_arr)
        print(f"  Spearman correlation: r={spearman.correlation:.4f}, p={spearman.pvalue:.4f}")
    else:
        scores_arr = np.array([])
        relevance_arr = np.array([])

    try:
        if len(correlation_data) > 0:
            ranks_global = stats.rankdata(scores_arr, method="average")
            n_pos_global = int(np.sum(relevance_arr))
            n_neg_global = len(relevance_arr) - n_pos_global
            if n_pos_global > 0 and n_neg_global > 0:
                sum_ranks_pos_global = float(np.sum(ranks_global[relevance_arr == 1]))
                auc_global = (sum_ranks_pos_global - n_pos_global * (n_pos_global + 1) / 2.0) / (
                    n_pos_global * n_neg_global
                )
            else:
                auc_global = None
        else:
            auc_global = None
    except Exception:
        auc_global = None

    mean_user_auc = float(np.mean(per_user_auc)) if per_user_auc else None
    mean_top1_hit = float(np.mean(per_user_top1_hit)) if per_user_top1_hit else None

    print(f"\nPairwise ranking quality (oracle candidates):")
    if mean_user_auc is not None:
        print(f"  Mean per-user AUC (pairwise accuracy): {mean_user_auc:.4f}")
    if auc_global is not None:
        print(f"  Global AUC (all users/items): {auc_global:.4f}")
    if mean_top1_hit is not None:
        print(f"  Top-1 hit rate within scored candidates: {mean_top1_hit:.4f}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.hist(irrelevant_scores, bins=50, alpha=0.6, label='Irrelevant', color='red', density=True)
    ax1.hist(relevant_scores, bins=50, alpha=0.6, label='Relevant', color='green', density=True)
    ax1.set_xlabel("Reranker Score", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Density", fontsize=12, fontweight='bold')
    ax1.set_title("Score Distribution: Relevant vs Irrelevant", fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    if scores_arr.size > 0:
        ax2.scatter(scores_arr, relevance_arr, alpha=0.1, s=5)
    ax2.set_xlabel("Reranker Score", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Relevance (0/1)", fontsize=12, fontweight='bold')
    ax2.set_title(f"Score vs Relevance\n(Spearman r={spearman.correlation:.3f})", fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / f"score_distribution_{run_id}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    return {
        "run_id": run_id,
        "relevant_mean": float(np.mean(relevant_scores)),
        "relevant_std": float(np.std(relevant_scores)),
        "irrelevant_mean": float(np.mean(irrelevant_scores)),
        "irrelevant_std": float(np.std(irrelevant_scores)),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "spearman_r": float(spearman.correlation) if correlation_data else None,
        "spearman_p": float(spearman.pvalue) if correlation_data else None
    }


def main():
    print("Analyzing Reranker Score Distributions...")
    
    gt = load_gt()
    
    results_files = sorted(RESULTS_DIR.glob("*.json"))
    
    all_results = []
    
    for results_file in results_files:
        run_id = results_file.stem
        
        if "ablation_with_reranker" in run_id or "ours_with_reranker" in run_id:
            result = analyze_score_distributions(run_id, gt)
            if result:
                all_results.append(result)
    
    if all_results:
        output_path = OUTPUT_DIR / "score_analysis_summary.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved summary to: {output_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
