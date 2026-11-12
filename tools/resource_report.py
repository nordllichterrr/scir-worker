import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

RUNS_LOG_PATH = Path("experiments") / "runs.jsonl"
OUTPUT_DIR = Path("experiments") / "resources"
ROOT = Path(__file__).resolve().parents[1]


def load_runs(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    runs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    runs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return runs


def has_timing(r: Dict) -> bool:
    diag = r.get("diagnostics", {})
    return (
        diag.get("retrieval_time_mean") is not None
        or diag.get("rerank_time_mean") is not None
        or diag.get("total_time_per_user_mean") is not None
    )


def latency_table(runs: List[Dict]) -> List[Dict[str, Any]]:
    rows = []
    for r in runs:
        if not has_timing(r):
            continue
        cfg = r.get("config", {})
        if cfg.get("baseline"):
            continue
        diag = r.get("diagnostics", {})
        metrics = r.get("metrics") or {}
        hr10 = metrics.get("hr@10")
        if isinstance(hr10, dict):
            hr10 = hr10.get("mean")
        row = {
            "run_id": r.get("run_id", ""),
            "reranker_model": cfg.get("reranker_model", "N/A"),
            "pool_size": cfg.get("candidate_pool_size"),
            "retrieval_mean_s": diag.get("retrieval_time_mean"),
            "retrieval_p50_s": diag.get("retrieval_time_p50"),
            "retrieval_p95_s": diag.get("retrieval_time_p95"),
            "rerank_mean_s": diag.get("rerank_time_mean"),
            "rerank_p50_s": diag.get("rerank_time_p50"),
            "rerank_p95_s": diag.get("rerank_time_p95"),
            "total_mean_s": diag.get("total_time_per_user_mean"),
            "total_p50_s": diag.get("total_time_per_user_p50"),
            "total_p95_s": diag.get("total_time_per_user_p95"),
            "throughput_users_per_sec": diag.get("throughput_users_per_sec"),
            "hr10": hr10,
        }
        rows.append(row)
    return rows


def aggregate_by_pool_size(runs: List[Dict]) -> Dict[int, Dict[str, Any]]:
    by_pool = defaultdict(lambda: {"retrieval": [], "rerank": [], "total": [], "throughput": [], "hr10": []})
    for r in runs:
        cfg = r.get("config", {})
        if cfg.get("baseline") or not cfg.get("use_reranker", True):
            continue
        pool = cfg.get("candidate_pool_size")
        if pool is None:
            continue
        diag = r.get("diagnostics", {})
        if diag.get("total_time_per_user_mean") is None and diag.get("rerank_time_mean") is None:
            continue
        m = r.get("metrics") or {}
        hr10 = m.get("hr@10")
        if isinstance(hr10, dict):
            hr10 = hr10.get("mean")
        by_pool[pool]["retrieval"].append(diag.get("retrieval_time_mean"))
        by_pool[pool]["rerank"].append(diag.get("rerank_time_mean"))
        by_pool[pool]["total"].append(diag.get("total_time_per_user_mean"))
        by_pool[pool]["throughput"].append(diag.get("throughput_users_per_sec"))
        if hr10 is not None:
            by_pool[pool]["hr10"].append(float(hr10))
    def _mean(lst):
        vals = [x for x in lst if x is not None]
        return sum(vals) / len(vals) if vals else None

    result = {}
    for p, v in by_pool.items():
        result[p] = {
            "pool_size": p,
            "retrieval_mean_s": _mean(v["retrieval"]),
            "rerank_mean_s": _mean(v["rerank"]),
            "total_mean_s": _mean(v["total"]),
            "throughput_users_per_sec": _mean(v["throughput"]),
            "hr10_mean": _mean(v["hr10"]),
        }
    return result


def aggregate_by_reranker_model(runs: List[Dict]) -> Dict[str, Dict[str, Any]]:
    by_model = defaultdict(lambda: {"retrieval": [], "rerank": [], "total": [], "throughput": [], "hr10": []})
    for r in runs:
        cfg = r.get("config", {})
        if cfg.get("baseline") or not cfg.get("use_reranker", True):
            continue
        model = cfg.get("reranker_model") or "default"
        if hasattr(model, "__fspath__"):
            model = str(model)
        model_short = model.split("/")[-1] if "/" in model else model
        diag = r.get("diagnostics", {})
        if diag.get("total_time_per_user_mean") is None and diag.get("rerank_time_mean") is None:
            continue
        m = r.get("metrics") or {}
        hr10 = m.get("hr@10")
        if isinstance(hr10, dict):
            hr10 = hr10.get("mean")
        by_model[model_short]["retrieval"].append(diag.get("retrieval_time_mean"))
        by_model[model_short]["rerank"].append(diag.get("rerank_time_mean"))
        by_model[model_short]["total"].append(diag.get("total_time_per_user_mean"))
        by_model[model_short]["throughput"].append(diag.get("throughput_users_per_sec"))
        if hr10 is not None:
            by_model[model_short]["hr10"].append(float(hr10))
    def _mean(lst):
        vals = [x for x in lst if x is not None]
        return sum(vals) / len(vals) if vals else None

    result = {}
    for name, v in by_model.items():
        result[name] = {
            "reranker_model": name,
            "retrieval_mean_s": _mean(v["retrieval"]),
            "rerank_mean_s": _mean(v["rerank"]),
            "total_mean_s": _mean(v["total"]),
            "throughput_users_per_sec": _mean(v["throughput"]),
            "hr10_mean": _mean(v["hr10"]),
        }
    return result


def format_latency_row(row: Dict) -> str:
    parts = []
    if row.get("retrieval_mean_s") is not None:
        parts.append(f"retrieval={row['retrieval_mean_s']:.3f}s")
    if row.get("rerank_mean_s") is not None:
        parts.append(f"rerank={row['rerank_mean_s']:.3f}s")
    if row.get("total_mean_s") is not None:
        parts.append(f"total={row['total_mean_s']:.3f}s")
    if row.get("total_p50_s") is not None:
        parts.append(f"p50={row['total_p50_s']:.3f}s")
    if row.get("total_p95_s") is not None:
        parts.append(f"p95={row['total_p95_s']:.3f}s")
    if row.get("throughput_users_per_sec") is not None:
        parts.append(f"throughput={row['throughput_users_per_sec']:.4f} users/s")
    return ", ".join(parts)


def write_markdown_report(
    latency_rows: List[Dict],
    by_pool: Dict[int, Dict],
    by_model: Dict[str, Dict],
    out_path: Path,
) -> None:
    lines = [
        "# Resource report (for paper)",
        "",
        "## Latency per user (retrieval + rerank)",
        "",
        "| run_id | retrieval (mean) | rerank (mean) | total (mean) | p50 | p95 | throughput (users/s) | HR@10 |",
        "|--------|------------------|----------------|--------------|-----|-----|---------------------|-------|",
    ]
    for row in latency_rows[:30]:
        run_id = (row.get("run_id") or "")[:36]
        ret = f"{row['retrieval_mean_s']:.3f}" if row.get("retrieval_mean_s") is not None else "—"
        rr = f"{row['rerank_mean_s']:.3f}" if row.get("rerank_mean_s") is not None else "—"
        tot = f"{row['total_mean_s']:.3f}" if row.get("total_mean_s") is not None else "—"
        p50 = f"{row['total_p50_s']:.3f}" if row.get("total_p50_s") is not None else "—"
        p95 = f"{row['total_p95_s']:.3f}" if row.get("total_p95_s") is not None else "—"
        thr = f"{row['throughput_users_per_sec']:.4f}" if row.get("throughput_users_per_sec") is not None else "—"
        hr = f"{row['hr10']:.4f}" if row.get("hr10") is not None else "—"
        lines.append(f"| {run_id} | {ret} | {rr} | {tot} | {p50} | {p95} | {thr} | {hr} |")
    lines.extend([
        "",
        "## Quality vs compute: pool_size ablation",
        "",
        "| pool_size | retrieval (s) | rerank (s) | total (s) | throughput (users/s) | HR@10 |",
        "|-----------|---------------|------------|-----------|---------------------|-------|",
    ])
    for p in sorted(by_pool.keys()):
        v = by_pool[p]
        ret = f"{v['retrieval_mean_s']:.3f}" if v.get("retrieval_mean_s") is not None else "—"
        rr = f"{v['rerank_mean_s']:.3f}" if v.get("rerank_mean_s") is not None else "—"
        tot = f"{v['total_mean_s']:.3f}" if v.get("total_mean_s") is not None else "—"
        thr = f"{v['throughput_users_per_sec']:.4f}" if v.get("throughput_users_per_sec") is not None else "—"
        hr = f"{v['hr10_mean']:.4f}" if v.get("hr10_mean") is not None else "—"
        lines.append(f"| {p} | {ret} | {rr} | {tot} | {thr} | {hr} |")
    lines.extend([
        "",
        "## Quality vs compute: cross-encoder size (reranker model)",
        "",
        "| model | retrieval (s) | rerank (s) | total (s) | throughput (users/s) | HR@10 |",
        "|-------|---------------|------------|-----------|---------------------|-------|",
    ])
    for name in sorted(by_model.keys()):
        v = by_model[name]
        ret = f"{v['retrieval_mean_s']:.3f}" if v.get("retrieval_mean_s") is not None else "—"
        rr = f"{v['rerank_mean_s']:.3f}" if v.get("rerank_mean_s") is not None else "—"
        tot = f"{v['total_mean_s']:.3f}" if v.get("total_mean_s") is not None else "—"
        thr = f"{v['throughput_users_per_sec']:.4f}" if v.get("throughput_users_per_sec") is not None else "—"
        hr = f"{v['hr10_mean']:.4f}" if v.get("hr10_mean") is not None else "—"
        lines.append(f"| {name} | {ret} | {rr} | {tot} | {thr} | {hr} |")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Resource report: latency, throughput, quality vs compute.")
    parser.add_argument("--runs", type=Path, default=RUNS_LOG_PATH, help="Path to runs.jsonl")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    runs = load_runs(args.runs)
    if not runs:
        print(f"No runs found in {args.runs}")
        return

    latency_rows = latency_table(runs)
    by_pool = aggregate_by_pool_size(runs)
    by_model = aggregate_by_reranker_model(runs)

    print("\n" + "=" * 60)
    print("RESOURCE REPORT (for paper)")
    print("=" * 60)
    print("\nLatency per user (retrieval + rerank) and throughput")
    if latency_rows:
        r = latency_rows[0]
        print(f"  Example run: {r.get('run_id', '')[:50]}")
        print(f"  {format_latency_row(r)}")
        if r.get("hr10") is not None:
            print(f"  HR@10 = {r['hr10']:.4f}")
    else:
        print("  No runs with timing. Run experiments with reranker to get retrieval_times and rerank_times.")
    print("\nPool_size ablation (quality vs compute):")
    for p in sorted(by_pool.keys()):
        v = by_pool[p]
        t = v.get("total_mean_s") or 0
        hr = v.get("hr10_mean") or 0
        thr = v.get("throughput_users_per_sec") or 0
        print(f"  pool={p}: total={t:.3f}s, throughput={thr:.4f} users/s, HR@10={hr:.4f}")
    print("\nCross-encoder size ablation:")
    for name in sorted(by_model.keys()):
        v = by_model[name]
        t = v.get("total_mean_s") or 0
        hr = v.get("hr10_mean") or 0
        thr = v.get("throughput_users_per_sec") or 0
        print(f"  {name}: total={t:.3f}s, throughput={thr:.4f} users/s, HR@10={hr:.4f}")
    print("=" * 60)

    summary = {
        "latency_table": latency_rows,
        "by_pool_size": by_pool,
        "by_reranker_model": by_model,
    }
    json_path = args.out / "resource_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {json_path}")

    md_path = args.out / "resource_report.md"
    write_markdown_report(latency_rows, by_pool, by_model, md_path)
    print(f"Report: {md_path}")


if __name__ == "__main__":
    main()
