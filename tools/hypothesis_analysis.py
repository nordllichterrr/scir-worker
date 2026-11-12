import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

MASTER_JSON = Path("experiments") / "master_results.json"
RUNS_LOG_PATH = Path("experiments") / "runs.jsonl"
GT_PATH = Path("experiments") / "ground_truth.json"
OUTPUT_DIR = Path("experiments") / "hypothesis_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_master():
    if not MASTER_JSON.exists():
        raise FileNotFoundError(f"Master results not found. Run tools/build_master_results.py first.")
    with open(MASTER_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def load_gt():
    if not GT_PATH.exists():
        return {}
    with open(GT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data.get("data", data)


def load_candidate_pools(run_id: str) -> Dict[str, List[str]]:
    results_path = Path("results") / f"{run_id}.json"
    if not results_path.exists():
        return {}
    
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    if isinstance(results, dict) and "candidate_pools" in results:
        return results.get("candidate_pools", {})
    
    return {}


def analyze_coverage_retrieval(master: Dict[str, Any], gt: Dict[str, List[str]]):
    print("\n" + "="*60)
    print("HYPOTHESIS 1: Coverage / Retrieval Analysis")
    print("="*60)
    
    runs_by_model = defaultdict(list)
    
    for run in master["runs"]:
        config = run.get("config", {})
        baseline = config.get("baseline")
        use_reranker = config.get("use_reranker", False)
        
        if baseline:
            model_key = baseline
        elif use_reranker:
            model_key = "ours_with_reranker"
        else:
            model_key = "candidates_only"
        
        runs_by_model[model_key].append(run)
    
    recall_by_k = defaultdict(lambda: defaultdict(list))
    gt_positions = []
    
    for model_key, model_runs in runs_by_model.items():
        for run in model_runs:
            run_id = run.get("run_id")
            candidate_pools = load_candidate_pools(run_id)
            
            if not candidate_pools:
                continue
            
            total_relevant_items = 0
            total_found_by_k = {50: 0, 200: 0, 500: 0, 1000: 0}
            
            for uid, candidates in candidate_pools.items():
                gt_items = gt.get(str(uid), [])
                if not gt_items:
                    continue
                
                gt_ids = set(str(x) for x in gt_items)
                candidates_str = [str(c) for c in candidates]
                total_relevant = len(gt_ids)
                
                if total_relevant == 0:
                    continue
                
        
                for k in [50, 200, 500, 1000]:
                    topk_candidates = set(candidates_str[:k])
                    relevant_in_topk = len(gt_ids & topk_candidates)
                    total_found_by_k[k] += relevant_in_topk
                
                total_relevant_items += total_relevant
            

            if total_relevant_items > 0:
                for k in [50, 200, 500, 1000]:
                    recall_k = total_found_by_k[k] / total_relevant_items
                    recall_by_k[model_key][k].append(recall_k)
                
                for gt_id in gt_ids:
                    if gt_id in candidates_str:
                        pos = candidates_str.index(gt_id) + 1
                        gt_positions.append(pos)
    
    print("\nRecall@K by Model (Mean):")
    for model_key in sorted(recall_by_k.keys()):
        print(f"\n  {model_key}:")
        for k in [50, 200, 500, 1000]:
            if k in recall_by_k[model_key]:
                mean_recall = np.mean(recall_by_k[model_key][k])
                print(f"    Recall@{k}: {mean_recall:.3f} (n={len(recall_by_k[model_key][k])})")
    
    if gt_positions:
        print(f"\nGT Position in Candidates:")
        print(f"  Median: {np.median(gt_positions):.1f}")
        print(f"  Q1: {np.percentile(gt_positions, 25):.1f}")
        print(f"  Q3: {np.percentile(gt_positions, 75):.1f}")
        print(f"  Mean: {np.mean(gt_positions):.1f}")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    Ks = [50, 200, 500, 1000]
    colors = plt.cm.tab10(np.linspace(0, 1, len(recall_by_k)))
    
    for idx, (model_key, recalls) in enumerate(recall_by_k.items()):
        means = []
        for k in Ks:
            if k in recalls:
                means.append(np.mean(recalls[k]))
            else:
                means.append(0.0)
        
        display_name = model_key.replace("_", " ").title()
        ax.plot(Ks, means, marker='o', linewidth=2, markersize=8, 
               label=display_name, color=colors[idx])
    
    ax.set_xlabel("K (top-K candidates)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Recall@K", fontsize=12, fontweight='bold')
    ax.set_title("Coverage Analysis: Recall@K Curves", fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(alpha=0.3)


    for k_ref in [50, 200, 1000]:
        ax.axvline(x=k_ref, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "coverage_recall_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()
    
    if gt_positions:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(gt_positions, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_xlabel("Position of GT in Candidate Pool", fontsize=12, fontweight='bold')
        ax.set_ylabel("Frequency", fontsize=12, fontweight='bold')
        ax.set_title("Distribution of GT Positions in Candidates", fontsize=14, fontweight='bold')
        ax.axvline(np.median(gt_positions), color='red', linestyle='--', linewidth=2, label=f'Median: {np.median(gt_positions):.1f}')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        output_path = OUTPUT_DIR / "gt_positions_histogram.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()


def analyze_bias_exposure(master: Dict[str, Any]):
    print("\n" + "="*60)
    print("HYPOTHESIS 2: Bias Reranker / Exposure Analysis")
    print("="*60)
    
    runs_by_model = defaultdict(list)
    
    for run in master["runs"]:
        config = run.get("config", {})
        baseline = config.get("baseline")
        use_reranker = config.get("use_reranker", False)
        
        if baseline:
            model_key = baseline
        elif use_reranker:
            model_key = "ours_with_reranker"
        else:
            model_key = "candidates_only"
        
        runs_by_model[model_key].append(run)
    
    for model_key, model_runs in runs_by_model.items():
        all_top1_items = []
        all_exposed_items = []
        
        for run in model_runs:
            per_user = run.get("per_user_detail", {})
            diag = run.get("diagnostics", {})
            
            top1_counts = diag.get("top1_counts", [])
            for item_id, count in top1_counts:
                all_top1_items.extend([item_id] * count)
            
            for user_id, user_data in per_user.items():
                rec_ids_str = user_data.get("rec_ids", "")
                if rec_ids_str and rec_ids_str != "None":
                    rec_ids = rec_ids_str.split()[:10]
                    all_exposed_items.extend(rec_ids)
        
        if all_top1_items:
            top1_counter = Counter(all_top1_items)
            unique_top1 = len(top1_counter)
            most_common = top1_counter.most_common(20)
            
            total = sum(top1_counter.values())
            gini = compute_gini(list(top1_counter.values()))
            entropy = compute_entropy(list(top1_counter.values()))
            
            print(f"\n{model_key}:")
            print(f"  Unique top-1 items: {unique_top1}")
            print(f"  Gini coefficient: {gini:.3f}")
            print(f"  Entropy: {entropy:.3f}")
            print(f"  Top-5 most frequent top-1:")
            for item_id, count in most_common[:5]:
                fraction = count / total
                print(f"    {item_id}: {count} ({fraction:.3f})")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            items = [x[0] for x in most_common[:20]]
            counts = [x[1] for x in most_common[:20]]
            
            bars = ax.bar(range(len(items)), counts, alpha=0.7, color='coral', edgecolor='black')
            ax.set_xlabel("Item ID (Top-20 Most Frequent Top-1)", fontsize=12, fontweight='bold')
            ax.set_ylabel("Count", fontsize=12, fontweight='bold')
            ax.set_title(f"Top-1 Exposure: {model_key.replace('_', ' ').title()}", fontsize=14, fontweight='bold')
            ax.set_xticks(range(len(items)))
            ax.set_xticklabels(items, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            output_path = OUTPUT_DIR / f"top1_exposure_{model_key}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {output_path}")
            plt.close()


def compute_gini(values: List[float]) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    cumsum = np.cumsum(sorted_vals)
    return (2 * np.sum((np.arange(1, n + 1)) * sorted_vals)) / (n * np.sum(sorted_vals)) - (n + 1) / n


def compute_entropy(values: List[float]) -> float:
    if not values:
        return 0.0
    total = sum(values)
    if total == 0:
        return 0.0
    probs = [v / total for v in values if v > 0]
    return -sum(p * np.log2(p) for p in probs)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def _compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.clip(
        np.searchsorted(bin_edges[1:-1], probs, side="right"), 0, n_bins - 1
    )
    ece = 0.0
    n = len(probs)
    for b in range(n_bins):
        mask = bin_indices == b
        if not np.any(mask):
            continue
        acc = float(np.mean(labels[mask]))
        conf = float(np.mean(probs[mask]))
        weight = float(np.sum(mask)) / n
        ece += weight * abs(acc - conf)
    return ece


def analyze_score_distributions(master: Dict[str, Any], gt: Dict[str, List[str]]):
    print("\n" + "="*60)
    print("HYPOTHESIS 3: Score Distributions / Calibration")
    print("="*60)
    

    for run in master.get("runs", []):
        config = run.get("config", {})
        if not config.get("use_reranker", False):
            continue
        if config.get("baseline"):
            continue
        
        run_id = run.get("run_id")
        per_user = run.get("per_user_detail", {})
        
        scores = []
        labels = []
        
        for uid, user_data in per_user.items():
            user_gt_items = set(str(x) for x in gt.get(str(uid), []))
            reranker_scores = user_data.get("reranker_scores") or {}
            
            for item_id, score in reranker_scores.items():
                scores.append(float(score))
                labels.append(1.0 if str(item_id) in user_gt_items else 0.0)
        
        if not scores or not labels or sum(labels) == 0:
            print(f"\n{run_id}:")
            print("  Skipping calibration: not enough positive examples or scores.")
            continue
        
        logits_arr = np.array(scores)
        labels_arr = np.array(labels)

        probs_arr = _sigmoid(logits_arr)
        
        brier = float(np.mean((probs_arr - labels_arr) ** 2))
        ece = _compute_ece(probs_arr, labels_arr, n_bins=10)
        

        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.clip(
            np.searchsorted(bin_edges[1:-1], probs_arr, side="right"), 0, n_bins - 1
        )
        
        bin_centers = []
        bin_acc = []
        bin_counts = []
        
        for b in range(n_bins):
            mask = bin_indices == b
            if not np.any(mask):
                continue
            bin_probs = probs_arr[mask]
            bin_labels = labels_arr[mask]
            bin_centers.append(float(bin_probs.mean()))
            bin_acc.append(float(bin_labels.mean()))
            bin_counts.append(int(mask.sum()))
        
        if not bin_centers:
            print(f"\n{run_id}:")
            print("  Skipping calibration: empty bins after quantization.")
            continue
        
        print(f"\n{run_id}:")
        print(f"  Calibration points: {len(bin_centers)}")
        print(f"  Brier (on sigmoid(logits)): {brier:.4f}")
        print(f"  ECE (on probabilities): {ece:.4f}")
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(bin_centers, bin_acc, marker='o', linewidth=2, label='Empirical hit rate')
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
        ax1.set_xlabel("Predicted probability (sigmoid(logit), bin mean)", fontsize=12, fontweight='bold')
        ax1.set_ylabel("Hit rate in bin", fontsize=12, fontweight='bold')
        ax1.set_xlim(0.0, 1.0)
        ax1.set_ylim(0.0, 1.0)
        ax1.grid(alpha=0.3)
        ax1.set_title(f"Calibration of Reranker (probs = sigmoid(logit))\n{run_id}", fontsize=14, fontweight='bold')
        
        ax2 = ax1.twinx()
        width = 1.0 / max(n_bins, len(bin_centers))
        ax2.bar(bin_centers, bin_counts, alpha=0.2, color='orange', width=width or 0.05, label='Count')
        ax2.set_ylabel("Items per bin", fontsize=10)
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        plt.tight_layout()
        output_path = OUTPUT_DIR / f"calibration_{run_id}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()


def analyze_recall_vs_hr(master: Dict[str, Any], gt: Dict[str, List[str]]):
    print("\n" + "="*60)
    print("HYPOTHESIS 4: Recall in Candidates vs HR@10 Correlation")
    print("="*60)
    
    for run in master["runs"]:
        config = run.get("config", {})
        if config.get("baseline"):
            continue
        
        run_id = run.get("run_id")
        candidate_pools = load_candidate_pools(run_id)
        
        if not candidate_pools:
            continue
        
        recall_values = []
        hr_values = []
        
        per_user = run.get("per_user_detail", {})
        
        for uid, candidates in candidate_pools.items():
            gt_items = gt.get(str(uid), [])
            if not gt_items:
                continue
            
            user_data = per_user.get(str(uid), {})
            hr = user_data.get("hr@10", 0.0)
            
            gt_ids = set(str(x) for x in gt_items)
            candidates_str = [str(c) for c in candidates[:200]]
            
            if gt_ids & set(candidates_str):
                recall_values.append(1.0)
            else:
                recall_values.append(0.0)
            
            hr_values.append(hr)
        
        if recall_values and hr_values:
            correlation = np.corrcoef(recall_values, hr_values)[0, 1]
            print(f"\n{run_id}:")
            print(f"  Correlation (recall@200 in candidates vs HR@10): {correlation:.3f}")
            
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.scatter(recall_values, hr_values, alpha=0.5, s=20)
            ax.set_xlabel("Recall@200 in Candidates (Binary)", fontsize=12, fontweight='bold')
            ax.set_ylabel("HR@10", fontsize=12, fontweight='bold')
            ax.set_title(f"Recall in Candidates vs HR@10\n(r={correlation:.3f})", fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3)
            
            plt.tight_layout()
            output_path = OUTPUT_DIR / f"recall_vs_hr_{run_id}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {output_path}")
            plt.close()


def check_gt_catalog_quality(gt: Dict[str, List[str]]):
    print("\n" + "="*60)
    print("HYPOTHESIS 5: GT / Catalog Quality Check")
    print("="*60)
    
    items_csv = Path("data/processed/items_serendipity.csv")
    if not items_csv.exists():
        print("Items CSV not found. Skipping catalog check.")
        return
    
    catalog_items = set()
    with open(items_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_id = row.get("item_id", "")
            if item_id:
                catalog_items.add(str(item_id))
    
    all_gt_items = set()
    for uid, gt_items in gt.items():
        for item_id in gt_items:
            all_gt_items.add(str(item_id))
    
    missing_items = all_gt_items - catalog_items
    
    print(f"\nCatalog items: {len(catalog_items)}")
    print(f"GT items: {len(all_gt_items)}")
    print(f"Missing GT items in catalog: {len(missing_items)}")
    
    if missing_items:
        print(f"\nFirst 20 missing items: {list(missing_items)[:20]}")
        
        users_with_missing = 0
        for uid, gt_items in gt.items():
            gt_ids = set(str(x) for x in gt_items)
            if gt_ids & missing_items:
                users_with_missing += 1
        
        print(f"Users affected by missing items: {users_with_missing} / {len(gt)}")
    
    return missing_items


def main():
    print("Running Hypothesis Analysis...")
    
    try:
        master = load_master()
        gt = load_gt()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run: python -m tools.build_master_results")
        return
    
    analyze_coverage_retrieval(master, gt)
    analyze_bias_exposure(master)
    analyze_score_distributions(master, gt)
    analyze_recall_vs_hr(master, gt)
    check_gt_catalog_quality(gt)
    
    print(f"\n" + "="*60)
    print("All hypothesis analyses saved to:", OUTPUT_DIR)
    print("="*60)


if __name__ == "__main__":
    main()
