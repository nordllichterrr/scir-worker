import json
from pathlib import Path
from typing import Dict, Any

AGGREGATED_JSON = Path("experiments") / "aggregated_results.json"
OUTPUT_DIR = Path("experiments") / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_aggregated() -> Dict[str, Any]:
    if not AGGREGATED_JSON.exists():
        raise FileNotFoundError(f"Aggregated results not found. Run tools/aggregate_runs.py first.")
    with open(AGGREGATED_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def format_number(mean: float, std: float) -> str:
    if mean is None or std is None:
        return "N/A"
    return f"{mean:.3f} ± {std:.3f}"


def generate_latex_table(aggregated: Dict[str, Any]) -> str:
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Main Results: HR@10 and nDCG@10 (Mean ± Std across seeds; main method = retrieval-only)}")
    lines.append("\\label{tab:main_results}")
    lines.append("\\begin{tabular}{lcc}")
    lines.append("\\toprule")
    lines.append("Model & HR@10 & nDCG@10 \\\\")
    lines.append("\\midrule")
    
    order = ["random", "popularity", "embedding_cosine", "candidates_only", "ours_with_reranker"]
    model_display = {
        "random": "Random",
        "popularity": "Popularity",
        "embedding_cosine": "Embedding Cosine",
        "candidates_only": "Ours (retrieval-only)",
        "ours_with_reranker": "Ours+Rerank (ablation)"
    }
    
    for model_name in order:
        for key, data in aggregated.items():
            if data["model"] == model_name:
                hr = format_number(
                    data["metrics"]["hr@10"]["mean"],
                    data["metrics"]["hr@10"]["std"]
                )
                ndcg = format_number(
                    data["metrics"]["ndcg@10"]["mean"],
                    data["metrics"]["ndcg@10"]["std"]
                )
                display_name = model_display.get(model_name, model_name.replace("_", " ").title())
                lines.append(f"{display_name} & {hr} & {ndcg} \\\\")
                break
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def generate_markdown_table(aggregated: Dict[str, Any]) -> str:
    lines = []
    lines.append("# Main Results")
    lines.append("")
    lines.append("HR@10 and nDCG@10 (Mean ± Std across seeds; main method = retrieval-only)")
    lines.append("")
    lines.append("| Model | HR@10 | nDCG@10 | Recall@50 | Recall@200 | Recall@1000 |")
    lines.append("|-------|-------|---------|-----------|------------|-------------|")
    
    order = ["random", "popularity", "embedding_cosine", "candidates_only", "ours_with_reranker"]
    model_display = {
        "random": "Random",
        "popularity": "Popularity",
        "embedding_cosine": "Embedding Cosine",
        "candidates_only": "Ours (retrieval-only)",
        "ours_with_reranker": "Ours+Rerank (ablation)"
    }
    
    for model_name in order:
        for key, data in aggregated.items():
            if data["model"] == model_name:
                hr = format_number(
                    data["metrics"]["hr@10"]["mean"],
                    data["metrics"]["hr@10"]["std"]
                )
                ndcg = format_number(
                    data["metrics"]["ndcg@10"]["mean"],
                    data["metrics"]["ndcg@10"]["std"]
                )
                recall50 = format_number(
                    data["diagnostics"]["recall@50"]["mean"],
                    data["diagnostics"]["recall@50"]["std"]
                )
                recall200 = format_number(
                    data["diagnostics"]["recall@200"]["mean"],
                    data["diagnostics"]["recall@200"]["std"]
                )
                recall1000 = format_number(
                    data["diagnostics"]["recall@1000"]["mean"],
                    data["diagnostics"]["recall@1000"]["std"]
                )
                display_name = model_display.get(model_name, model_name.replace("_", " ").title())
                lines.append(f"| {display_name} | {hr} | {ndcg} | {recall50} | {recall200} | {recall1000} |")
                break
    
    return "\n".join(lines)


def generate_diagnostics_table(aggregated: Dict[str, Any]) -> str:
    lines = []
    lines.append("# Diagnostics")
    lines.append("")
    lines.append("| Model | Recall@50 | Recall@200 | Recall@1000 | Unique Top-1 |")
    lines.append("|-------|-----------|------------|-------------|--------------|")
    
    order = ["random", "popularity", "embedding_cosine", "candidates_only", "ours_with_reranker"]
    model_display = {
        "random": "Random",
        "popularity": "Popularity",
        "embedding_cosine": "Embedding Cosine",
        "candidates_only": "Candidates Only",
        "ours_with_reranker": "Ours (CE Rerank)"
    }
    
    for model_name in order:
        for key, data in aggregated.items():
            if data["model"] == model_name:
                recall50 = format_number(
                    data["diagnostics"]["recall@50"]["mean"],
                    data["diagnostics"]["recall@50"]["std"]
                )
                recall200 = format_number(
                    data["diagnostics"]["recall@200"]["mean"],
                    data["diagnostics"]["recall@200"]["std"]
                )
                recall1000 = format_number(
                    data["diagnostics"]["recall@1000"]["mean"],
                    data["diagnostics"]["recall@1000"]["std"]
                )
                unique_top1 = format_number(
                    data["diagnostics"]["unique_top1"]["mean"],
                    data["diagnostics"]["unique_top1"]["std"]
                )
                display_name = model_display.get(model_name, model_name.replace("_", " ").title())
                lines.append(f"| {display_name} | {recall50} | {recall200} | {recall1000} | {unique_top1} |")
                break
    
    return "\n".join(lines)


def main():
    print("Loading aggregated results...")
    try:
        aggregated = load_aggregated()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run: python -m tools.aggregate_runs")
        return
    
    print("Generating tables...")
    
    latex = generate_latex_table(aggregated)
    latex_path = OUTPUT_DIR / "main_results.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"Saved: {latex_path}")
    
    md_main = generate_markdown_table(aggregated)
    md_path = OUTPUT_DIR / "main_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_main)
    print(f"Saved: {md_path}")
    
    md_diag = generate_diagnostics_table(aggregated)
    md_diag_path = OUTPUT_DIR / "diagnostics.md"
    with open(md_diag_path, "w", encoding="utf-8") as f:
        f.write(md_diag)
    print(f"Saved: {md_diag_path}")
    
    print(f"\nAll tables saved to {OUTPUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
