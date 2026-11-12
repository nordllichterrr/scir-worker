import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple
import numpy as np

RUNS_JSONL = Path("experiments") / "runs.jsonl"
OUT_DIR = Path("experiments") / "segmentation"
BUCKET_ORDER = ("0", "1-2", "3-5", "6-10", "11-20", "21+")
SCENARIO_ORDER = ("new_users", "new_items", "both")


def get_config_key(run: Dict[str, Any]) -> str:
    c = run.get("config", {})
    baseline = c.get("baseline")
    use_reranker = c.get("use_reranker", False)
    pool_size = c.get("candidate_pool_size", 1000)
    if baseline:
        return f"{baseline}_no_reranker"
    return f"ours_with_reranker_pool{pool_size}" if use_reranker else f"candidates_only_pool{pool_size}"


def get_model_display_name(config_key: str) -> str:
    if config_key.startswith("ours_with_reranker"):
        return "Ours+Rerank"
    if config_key.startswith("candidates_only"):
        return "Candidates"
    base = config_key.replace("_no_reranker", "").replace("_", " ").title()
    return base


def load_runs() -> List[Dict[str, Any]]:
    if not RUNS_JSONL.exists():
        return []
    runs = []
    with open(RUNS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    runs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return runs


def agg_over_seeds(vals: List[float]) -> Tuple[float, float]:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    if not runs:
        print("No runs in runs.jsonl. Run experiments first.")
        return

    by_config: Dict[str, List[Dict]] = defaultdict(list)
    for run in runs:
        key = get_config_key(run)
        by_config[key].append(run)

    bucket_data: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for config_key, config_runs in by_config.items():
        for run in config_runs:
            by_bucket = run.get("metrics", {}).get("by_bucket") or {}
            for bucket, seg in by_bucket.items():
                if not seg or seg.get("n_users", 0) == 0:
                    continue
                hr = seg.get("hr_mean")
                ndcg = seg.get("ndcg_mean")
                if hr is not None:
                    bucket_data[(config_key, bucket)]["hr"].append(hr)
                if ndcg is not None:
                    bucket_data[(config_key, bucket)]["ndcg"].append(ndcg)

    scenario_data: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for config_key, config_runs in by_config.items():
        for run in config_runs:
            by_scenario = run.get("metrics", {}).get("by_scenario") or {}
            for scenario, seg in by_scenario.items():
                if not seg or seg.get("n_users", 0) == 0:
                    continue
                hr = seg.get("hr_mean")
                ndcg = seg.get("ndcg_mean")
                if hr is not None:
                    scenario_data[(config_key, scenario)]["hr"].append(hr)
                if ndcg is not None:
                    scenario_data[(config_key, scenario)]["ndcg"].append(ndcg)

    
    by_bucket_summary: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for (config_key, bucket), vals in bucket_data.items():
        hr_mean, hr_std = agg_over_seeds(vals["hr"])
        ndcg_mean, ndcg_std = agg_over_seeds(vals["ndcg"])
        by_bucket_summary[config_key][bucket] = {
            "hr_mean": hr_mean, "hr_std": hr_std,
            "ndcg_mean": ndcg_mean, "ndcg_std": ndcg_std,
            "n_seeds": len(vals["hr"]),
        }

    by_scenario_summary: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for (config_key, scenario), vals in scenario_data.items():
        hr_mean, hr_std = agg_over_seeds(vals["hr"])
        ndcg_mean, ndcg_std = agg_over_seeds(vals["ndcg"])
        by_scenario_summary[config_key][scenario] = {
            "hr_mean": hr_mean, "hr_std": hr_std,
            "ndcg_mean": ndcg_mean, "ndcg_std": ndcg_std,
            "n_seeds": len(vals["hr"]),
        }

    def rank_models_in_segment(segment_metrics: Dict[str, Dict[str, Any]], metric: str = "hr_mean") -> List[Tuple[str, float]]:
        pairs = []
        for config_key, seg in segment_metrics.items():
            v = seg.get(metric)
            if v is not None:
                pairs.append((config_key, v))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs

    where_win = {"by_bucket": {}, "by_scenario": {}, "why": {}}
    for bucket in BUCKET_ORDER:
        segment_metrics = {k: v.get(bucket) for k, v in by_bucket_summary.items() if v.get(bucket)}
        segment_metrics = {k: v for k, v in segment_metrics.items() if v}
        if not segment_metrics:
            continue
        ranked_hr = rank_models_in_segment(segment_metrics, "hr_mean")
        ranked_ndcg = rank_models_in_segment(segment_metrics, "ndcg_mean")
        best_hr = ranked_hr[0] if ranked_hr else (None, None)
        best_ndcg = ranked_ndcg[0] if ranked_ndcg else (None, None)
        gap_hr = (ranked_hr[0][1] - ranked_hr[1][1]) if len(ranked_hr) >= 2 else None
        gap_ndcg = (ranked_ndcg[0][1] - ranked_ndcg[1][1]) if len(ranked_ndcg) >= 2 else None
        where_win["by_bucket"][bucket] = {
            "best_hr": get_model_display_name(best_hr[0]) if best_hr[0] else None,
            "best_hr_value": best_hr[1],
            "gap_hr_to_second": gap_hr,
            "best_ndcg": get_model_display_name(best_ndcg[0]) if best_ndcg[0] else None,
            "best_ndcg_value": best_ndcg[1],
            "gap_ndcg_to_second": gap_ndcg,
        }
      
        n_train = bucket if bucket != "0" else "0"
        where_win["why"][f"bucket_{bucket}"] = _why_bucket(bucket, best_hr[0], best_ndcg[0])

    for scenario in SCENARIO_ORDER:
        segment_metrics = {k: v.get(scenario) for k, v in by_scenario_summary.items() if v.get(scenario)}
        segment_metrics = {k: v for k, v in segment_metrics.items() if v}
        if not segment_metrics:
            continue
        ranked_hr = rank_models_in_segment(segment_metrics, "hr_mean")
        ranked_ndcg = rank_models_in_segment(segment_metrics, "ndcg_mean")
        best_hr = ranked_hr[0] if ranked_hr else (None, None)
        best_ndcg = ranked_ndcg[0] if ranked_ndcg else (None, None)
        gap_hr = (ranked_hr[0][1] - ranked_hr[1][1]) if len(ranked_hr) >= 2 else None
        gap_ndcg = (ranked_ndcg[0][1] - ranked_ndcg[1][1]) if len(ranked_ndcg) >= 2 else None
        where_win["by_scenario"][scenario] = {
            "best_hr": get_model_display_name(best_hr[0]) if best_hr[0] else None,
            "best_hr_value": best_hr[1],
            "gap_hr_to_second": gap_hr,
            "best_ndcg": get_model_display_name(best_ndcg[0]) if best_ndcg[0] else None,
            "best_ndcg_value": best_ndcg[1],
            "gap_ndcg_to_second": gap_ndcg,
        }
        where_win["why"][f"scenario_{scenario}"] = _why_scenario(scenario, best_hr[0], best_ndcg[0])

    out = {
        "by_bucket": {k: v for k, v in by_bucket_summary.items()},
        "by_scenario": {k: v for k, v in by_scenario_summary.items()},
        "where_we_win": where_win,
    }
    with open(OUT_DIR / "segmentation_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    with open(OUT_DIR / "where_we_win.json", "w", encoding="utf-8") as f:
        json.dump(where_win, f, indent=2)


    lines = [
        "# Segmentation Analysis: Where We Win",
        "",
        "Metrics by **history bucket** (train interactions per user) and **cold-start scenario**.",
        "",
        "## 1. HR@10 and nDCG@10 by history bucket",
        "",
        "| Model | 0 | 1-2 | 3-5 | 6-10 | 11-20 | 21+ |",
        "|-------|" + "---|" * 6,
    ]
    for config_key in sorted(by_bucket_summary.keys(), key=lambda x: (0 if "ours" in x else 1, x)):
        display = get_model_display_name(config_key)
        row = [display]
        for b in BUCKET_ORDER:
            seg = by_bucket_summary[config_key].get(b)
            if seg and seg.get("hr_mean") is not None:
                row.append(f"{seg['hr_mean']:.3f}±{seg['hr_std'] or 0:.2f}")
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")
    lines.extend([
        "",
        "## 2. HR@10 and nDCG@10 by cold-start scenario",
        "",
        "| Model | new_users | new_items | both |",
        "|-------|-----------|-----------|------|",
    ])
    for config_key in sorted(by_scenario_summary.keys(), key=lambda x: (0 if "ours" in x else 1, x)):
        display = get_model_display_name(config_key)
        row = [display]
        for s in SCENARIO_ORDER:
            seg = by_scenario_summary[config_key].get(s)
            if seg and seg.get("hr_mean") is not None:
                row.append(f"{seg['hr_mean']:.3f}±{seg['hr_std'] or 0:.2f}")
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")
    lines.extend([
        "",
        "## 3. Where we win (best model per segment)",
        "",
    ])
    for bucket in BUCKET_ORDER:
        w = where_win["by_bucket"].get(bucket)
        if not w:
            continue
        lines.append(f"### Bucket {bucket} (train interactions)")
        lines.append(f"- Best HR@10: **{w['best_hr']}** ({w['best_hr_value']:.3f}); gap to second: {w['gap_hr_to_second']:.3f}" if w.get('gap_hr_to_second') is not None else f"- Best HR@10: **{w['best_hr']}** ({w['best_hr_value']:.3f})")
        lines.append(f"- Best nDCG@10: **{w['best_ndcg']}** ({w['best_ndcg_value']:.3f})" if w.get('best_ndcg_value') else "")
        lines.append(f"- **Why:** {where_win['why'].get(f'bucket_{bucket}', '—')}")
        lines.append("")
    for scenario in SCENARIO_ORDER:
        w = where_win["by_scenario"].get(scenario)
        if not w:
            continue
        lines.append(f"### Scenario: {scenario}")
        lines.append(f"- Best HR@10: **{w['best_hr']}** ({w['best_hr_value']:.3f}); gap to second: {w['gap_hr_to_second']:.3f}" if w.get('gap_hr_to_second') is not None else f"- Best HR@10: **{w['best_hr']}** ({w['best_hr_value']:.3f})")
        lines.append(f"- **Why:** {where_win['why'].get(f'scenario_{scenario}', '—')}")
        lines.append("")
    lines.append("---")
    lines.append("See `experiments/SEGMENTATION.md` for paper section and interpretation guide.")
    with open(OUT_DIR / "segmentation_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Segmentation analysis saved to {OUT_DIR}/segmentation_results.json and {OUT_DIR}/segmentation_report.md")


def _why_bucket(bucket: str, best_hr_config: str, best_ndcg_config: str) -> str:

    if bucket == "0":
        return "Zero train interactions: content-based (BM25) and two-tower (ANN) can still use query/text; collaborative (ItemKNN/EASE/MF) have no signal, so popularity or retrieval-based methods dominate."
    if bucket in ("1-2", "3-5"):
        return "Few interactions: weak collaborative signal; hybrid retrieval + content helps. Reranker can leverage limited profile."
    if bucket in ("6-10", "11-20"):
        return "Medium history: collaborative baselines start to help; our method can combine content and behavior."
    if bucket == "21+":
        return "Rich history: strong collaborative baselines competitive; gains from our method come from better ranking and diversity."
    return "Segment-specific: compare to popularity and collaborative baselines."


def _why_scenario(scenario: str, best_hr_config: str, best_ndcg_config: str) -> str:

    if scenario == "new_users":
        return "New users: no past behavior; content and two-tower retrieval are the only source of signal. Popularity is a strong baseline."
    if scenario == "new_items":
        return "New items in test: retrieval must cover tail; diversity and catalog coverage matter. Hybrid (ANN+BM25) and reranker help surface less popular items."
    if scenario == "both":
        return "Both new users and new items: hardest segment; content + broad retrieval (hybrid) and ranking that balances relevance and novelty win."
    return "Scenario-specific: see by_bucket for history effect."


if __name__ == "__main__":
    main()
