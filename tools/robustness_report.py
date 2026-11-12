
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import matplotlib.pyplot as plt

try:
    from .plot_style import apply_paper_style, save_fig_paper, AXES_FONTSIZE, TITLE_FONTSIZE
except ImportError:
    from plot_style import apply_paper_style, save_fig_paper, AXES_FONTSIZE, TITLE_FONTSIZE


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _model_label(run: Dict[str, Any]) -> str:
    cfg = run.get("config", {}) or {}
    baseline = cfg.get("baseline")
    if baseline == "popularity":
        return "Popularity"
    if baseline == "content_bm25":
        return "BM25"
    if baseline:
        return str(baseline)
    mode = cfg.get("retrieval_mode", "ann")
    use_reranker = bool(cfg.get("use_reranker", False))
    if use_reranker:
        return f"Ours+Rerank ({mode})"
    return f"Ours (retrieval-only, {mode})"


def _metric(run: Dict[str, Any], key: str) -> Optional[float]:
    try:
        return float(run.get("metrics", {}).get(key, {}).get("mean"))
    except Exception:
        return None


def plot_noise_curve(noise_runs: List[Dict[str, Any]], out_dir: Path) -> None:
    apply_paper_style()
    by_model: Dict[str, Dict[float, List[Tuple[Optional[float], Optional[float]]]]] = defaultdict(lambda: defaultdict(list))
    for r in noise_runs:
        cfg = r.get("config", {}) or {}
        rob = (cfg.get("robustness") or {})
        if rob.get("type") != "noise_drop":
            continue
        drop_p = float(rob.get("drop_p", 0.0))
        by_model[_model_label(r)][drop_p].append((_metric(r, "hr@10"), _metric(r, "ndcg@10")))

    if not by_model:
        return

    drop_levels = sorted({p for m in by_model.values() for p in m.keys()})
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    for model, m in sorted(by_model.items(), key=lambda kv: kv[0]):
        ys = []
        for p in drop_levels:
            vals = [x[1] for x in m.get(p, []) if x[1] is not None]  # ndcg
            ys.append(sum(vals) / len(vals) if vals else None)
        ax.plot(drop_levels, ys, marker="o", linewidth=2, label=model)

    ax.set_title("Robustness to noisy history (drop train events)", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Dropped fraction of train events", fontsize=AXES_FONTSIZE)
    ax.set_ylabel("nDCG@10 (mean over seeds)", fontsize=AXES_FONTSIZE)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=AXES_FONTSIZE - 1)
    plt.tight_layout()
    save_fig_paper(out_dir / "robustness_noise_plot")
    plt.close()


def plot_sensitivity_from_main_runs(main_runs: List[Dict[str, Any]], out_dir: Path) -> None:

    apply_paper_style()
    pool_vals: Dict[int, List[float]] = defaultdict(list)
    for r in main_runs:
        rid = str(r.get("run_id", ""))
        cfg = r.get("config", {}) or {}
        if not rid.startswith("retrieval_hybrid_candidates_only_"):
            continue
        pool = cfg.get("candidate_pool_size")
        if pool is None:
            continue
        nd = _metric(r, "ndcg@10")
        if nd is not None:
            pool_vals[int(pool)].append(float(nd))

    alpha_vals: Dict[float, List[float]] = defaultdict(list)
    for r in main_runs:
        rid = str(r.get("run_id", ""))
        if not rid.startswith("sweep_debias_popularity_alpha_"):
            continue
        cfg = r.get("config", {}) or {}
        div = (cfg.get("diversify_config") or {})
        alpha = div.get("popularity_penalty_alpha")
        if alpha is None:
            continue
        nd = _metric(r, "ndcg@10")
        if nd is not None:
            alpha_vals[float(alpha)].append(float(nd))

    if not pool_vals and not alpha_vals:
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))

    ax = axes[0]
    if pool_vals:
        xs = sorted(pool_vals.keys())
        ys = [sum(pool_vals[x]) / len(pool_vals[x]) for x in xs]
        ax.plot(xs, ys, marker="o", linewidth=2)
        ax.set_xscale("log")
        ax.set_title("Sensitivity: candidate pool size", fontsize=TITLE_FONTSIZE)
        ax.set_xlabel("candidate_pool_size (log scale)", fontsize=AXES_FONTSIZE)
        ax.set_ylabel("nDCG@10", fontsize=AXES_FONTSIZE)
        ax.grid(alpha=0.3)
    else:
        ax.axis("off")

    ax = axes[1]
    if alpha_vals:
        xs = sorted(alpha_vals.keys())
        ys = [sum(alpha_vals[x]) / len(alpha_vals[x]) for x in xs]
        ax.plot(xs, ys, marker="o", linewidth=2)
        ax.set_title("Sensitivity: popularity penalty α", fontsize=TITLE_FONTSIZE)
        ax.set_xlabel("popularity_penalty_alpha", fontsize=AXES_FONTSIZE)
        ax.set_ylabel("nDCG@10", fontsize=AXES_FONTSIZE)
        ax.grid(alpha=0.3)
    else:
        ax.axis("off")

    plt.tight_layout()
    save_fig_paper(out_dir / "robustness_sensitivity_plot")
    plt.close()


def temporal_shift_table(shift_runs: List[Dict[str, Any]], out_dir: Path) -> None:
    table: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for r in shift_runs:
        cfg = r.get("config", {}) or {}
        rob = (cfg.get("robustness") or {})
        if rob.get("type") != "temporal_shift":
            continue
        shift_tag = str(rob.get("shift_tag", "unknown"))
        model = _model_label(r)
        hr = _metric(r, "hr@10")
        nd = _metric(r, "ndcg@10")
        if hr is not None:
            table[shift_tag][model].setdefault("hr@10", 0.0)
        if nd is not None:
            table[shift_tag][model].setdefault("ndcg@10", 0.0)

    bucket: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in shift_runs:
        cfg = r.get("config", {}) or {}
        rob = (cfg.get("robustness") or {})
        if rob.get("type") != "temporal_shift":
            continue
        shift_tag = str(rob.get("shift_tag", "unknown"))
        model = _model_label(r)
        hr = _metric(r, "hr@10")
        nd = _metric(r, "ndcg@10")
        if hr is not None:
            bucket[shift_tag][model]["hr@10"].append(float(hr))
        if nd is not None:
            bucket[shift_tag][model]["ndcg@10"].append(float(nd))

    out_json = {}
    for shift_tag, by_model in bucket.items():
        out_json[shift_tag] = {}
        for model, mm in by_model.items():
            out_json[shift_tag][model] = {
                k: (sum(v) / len(v) if v else None) for k, v in mm.items()
            }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "temporal_shift_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)

    shift_tags = sorted(out_json.keys())
    models = sorted({m for s in out_json.values() for m in s.keys()})
    lines = [
        "# Temporal shift robustness",
        "",
        "Metrics under different time-based splits (train in the past, test in the future).",
        "",
    ]
    for metric in ["ndcg@10", "hr@10"]:
        lines.append(f"## {metric.upper()}")
        header = "| shift | " + " | ".join(models) + " |"
        sep = "|---|" + "|".join(["---"] * len(models)) + "|"
        lines.append(header)
        lines.append(sep)
        for tag in shift_tags:
            row = [tag]
            for m in models:
                v = out_json.get(tag, {}).get(m, {}).get(metric)
                row.append(f"{v:.4f}" if isinstance(v, (float, int)) and v is not None else "—")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    with open(out_dir / "temporal_shift_metrics.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")


def generate_robustness_report(project_root: Path) -> None:
    out_dir = project_root / "experiments" / "robustness"
    noise_runs = _read_jsonl(out_dir / "runs_noise.jsonl")
    shift_runs = _read_jsonl(out_dir / "runs_temporal_shift.jsonl")
    main_runs = _read_jsonl(project_root / "experiments" / "runs.jsonl")

    plot_noise_curve(noise_runs, out_dir)
    plot_sensitivity_from_main_runs(main_runs, out_dir)
    temporal_shift_table(shift_runs, out_dir)

