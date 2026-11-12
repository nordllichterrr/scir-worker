import json
from pathlib import Path
from typing import Dict, Any, Optional

AGGREGATED_JSON = Path("experiments") / "aggregated_results.json"
OUT_DIR = Path("experiments") / "counterfactual_evaluation"
OUT_JSON = OUT_DIR / "popularity_bias_analysis.json"
OUT_MD = OUT_DIR / "popularity_bias_report.md"

METRICS = [
    ("exposure_gini", "Exposure Gini", "higher = more head concentration"),
    ("top10_share", "Top-10 exposure share", "higher = more head"),
    ("catalog_coverage_at_10", "Catalog coverage @10", "higher = more diversity"),
    ("long_tail_coverage", "Long-tail coverage", "higher = more tail items"),
    ("mean_self_information_novelty", "Novelty (self-info)", "higher = more novel"),
    ("exposure_entropy", "Exposure entropy", "higher = more uniform"),
]


def load_aggregated() -> Dict[str, Any]:
    if not AGGREGATED_JSON.exists():
        raise FileNotFoundError(f"Run aggregate_runs first: {AGGREGATED_JSON}")
    with open(AGGREGATED_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def get_diagnostic(entry: Dict, key: str) -> Optional[float]:
    d = entry.get("diagnostics", {}).get(key, {})
    if isinstance(d, dict):
        return d.get("mean")
    return d


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    aggregated = load_aggregated()
    pop_values: Dict[str, Optional[float]] = {}
    for key, data in aggregated.items():
        if data.get("model") == "popularity":
            for mkey, _label, _ in METRICS:
                pop_values[mkey] = get_diagnostic(data, mkey)
            break
    if not pop_values:
        for key, data in aggregated.items():
            if data.get("config", {}).get("baseline") == "popularity":
                for mkey, _label, _ in METRICS:
                    pop_values[mkey] = get_diagnostic(data, mkey)
                break
    table = []
    for config_key, data in aggregated.items():
        model = data.get("model", config_key)
        row = {"model": model, "config_key": config_key}
        for mkey, label, _ in METRICS:
            v = get_diagnostic(data, mkey)
            row[mkey] = v
            if pop_values and mkey in pop_values and pop_values[mkey] is not None and v is not None:
                delta = v - pop_values[mkey]
                row[f"delta_vs_popularity_{mkey}"] = delta
        table.append(row)
    def order(r):
        if r["model"] == "popularity":
            return (0, r["model"])
        return (1, r["model"])
    table.sort(key=order)
    out = {
        "reference": "popularity",
        "popularity_baseline_values": pop_values,
        "per_model": table,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    lines = [
        "# Popularity Bias Analysis",
        "",
        "How much each algorithm amplifies the \"head\" of the distribution compared to the popularity baseline.",
        "",
        "| Model | Exposure Gini | Top-10 share | Catalog cov@10 | Long-tail cov | Novelty | Δ Gini vs pop | Δ Long-tail vs pop |",
        "|-------|---------------|--------------|----------------|---------------|---------|---------------|---------------------|",
    ]
    for row in table:
        gini = row.get("exposure_gini")
        top10 = row.get("top10_share")
        cov = row.get("catalog_coverage_at_10")
        ltail = row.get("long_tail_coverage")
        nov = row.get("mean_self_information_novelty")
        dg = row.get("delta_vs_popularity_exposure_gini")
        dlt = row.get("delta_vs_popularity_long_tail_coverage")
        def fmt(x):
            if x is None: return "—"
            return f"{x:.3f}"
        def fmt_d(x):
            if x is None: return "—"
            return f"{x:+.3f}"
        lines.append(f"| {row['model']} | {fmt(gini)} | {fmt(top10)} | {fmt(cov)} | {fmt(ltail)} | {fmt(nov)} | {fmt_d(dg)} | {fmt_d(dlt)} |")
    lines.extend([
        "",
        "**Interpretation:**",
        "- **Δ Gini > 0**: model concentrates exposure more on few items than popularity (stronger head bias).",
        "- **Δ Long-tail > 0**: model recommends more from the long tail than popularity (less head bias).",
        "",
    ])
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Popularity bias analysis saved to {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
