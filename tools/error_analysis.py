
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter, defaultdict
import random

MASTER_JSON = Path("experiments") / "master_results.json"
GT_PATH = Path("experiments") / "ground_truth.json"
OUTPUT_DIR = Path("experiments") / "error_analysis"
ITEMS_CSV = Path("data") / "processed" / "items_serendipity.csv"
ROOT = Path(__file__).resolve().parents[1]
ITEMS_CSV_ABS = ROOT / ITEMS_CSV

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


MIN_TITLE_LEN = 3
MIN_DESC_LEN = 10
HEAD_FRACTION = 0.2  
HEAD_COUNT_IN_TOP10 = 8  


def load_master() -> Dict[str, Any]:
    if not MASTER_JSON.exists():
        raise FileNotFoundError(f"Master results not found. Run tools/build_master_results.py first.")
    with open(MASTER_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def load_gt() -> Dict[str, List[str]]:
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


def load_items_metadata(path: Path = None) -> Dict[str, Dict[str, Any]]:
    path = path or ITEMS_CSV_ABS
    if not path.exists():
        return {}
    by_id = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iid = str(row.get("item_id") or row.get("movieId") or row.get("movie_id") or "")
            if not iid:
                continue
            title = str(row.get("title", "") or "").strip()
            desc = str(row.get("description", "") or row.get("text", "") or row.get("genres", "") or "").strip()
            by_id[iid] = {
                "title": title,
                "description": desc,
                "title_len": len(title),
                "desc_len": len(desc),
            }
    return by_id


def _exposure_count(per_item_exposure: Dict, item_id: str) -> int:
    v = (per_item_exposure or {}).get(str(item_id), {})
    if isinstance(v, dict):
        return v.get("count", v.get("exposure_count", 0))
    return 0


def classify_error_case(
    uid: str,
    gt_ids: List[str],
    rec_top10: List[str],
    candidate_pool: List[str],
    per_item_exposure: Dict[str, Dict[str, Any]],
    items_meta: Dict[str, Dict[str, Any]],
    head_exposure_threshold: Optional[float] = None,
) -> Tuple[str, Dict[str, bool]]:

    gt_set = set(str(i) for i in gt_ids)
    rec_set = list(rec_top10)[:10]
    cand_set = set(str(x) for x in (candidate_pool or []))

    gt_in_pool = bool(gt_set & cand_set)
    if not gt_in_pool:
        primary = "retrieval_failure"
    else:
        primary = "ranking_failure"

    bias_failure = False
    if head_exposure_threshold is not None and rec_set:
        exposure = per_item_exposure or {}
        head_in_top10 = sum(
            1 for i in rec_set
            if _exposure_count(exposure, i) >= head_exposure_threshold
        )
        bias_failure = head_in_top10 >= HEAD_COUNT_IN_TOP10

    metadata_failure = False
    if items_meta:
        for iid in gt_set:
            m = items_meta.get(str(iid), {})
            tl = m.get("title_len", 0)
            dl = m.get("desc_len", 0)
            if tl < MIN_TITLE_LEN or dl < MIN_DESC_LEN:
                metadata_failure = True
                break

    flags = {"bias_failure": bias_failure, "metadata_failure": metadata_failure}
    return primary, flags


def run_taxonomy(
    master: Dict[str, Any],
    gt: Dict[str, List[str]],
    run_id: Optional[str] = None,
    n_examples_per_type: int = 3,
) -> Dict[str, Any]:
    runs = master.get("runs", [])
    if run_id:
        runs = [r for r in runs if r.get("run_id") == run_id]
    our_runs = [r for r in runs if r.get("config", {}).get("use_reranker") and not r.get("config", {}).get("baseline")]
    if not our_runs:
        our_runs = runs
    run = our_runs[0]
    rid = run.get("run_id")
    per_user = run.get("per_user_detail", {})
    candidate_pools = load_candidate_pools(rid)
    per_item_exposure = master.get("per_item_exposure", {})
    items_meta = load_items_metadata()

    exp_counts = {}
    for iid, v in (per_item_exposure or {}).items():
        exp_counts[iid] = v.get("count", v.get("exposure_count", 0)) if isinstance(v, dict) else 0
    all_counts = sorted(exp_counts.values(), reverse=True)
    n_total = len(all_counts)
    head_threshold = float(all_counts[int(n_total * HEAD_FRACTION)]) if n_total else None

    errors_by_primary: Dict[str, List[Dict]] = defaultdict(list)
    bias_count = 0
    metadata_count = 0

    for uid, user_data in per_user.items():
        hr = user_data.get("hr@10", 0.0)
        if hr > 0:
            continue
        rec_ids_str = user_data.get("rec_ids", "")
        rec_top10 = rec_ids_str.split() if rec_ids_str and rec_ids_str != "None" else []
        gt_items = gt.get(str(uid), [])
        if not gt_items:
            continue
        gt_ids = [str(x) for x in gt_items]
        cand = candidate_pools.get(str(uid), [])
        primary, flags = classify_error_case(
            uid, gt_ids, rec_top10, cand, per_item_exposure, items_meta,
            head_exposure_threshold=head_threshold,
        )
        case = {
            "user_id": uid,
            "gt_items": gt_ids,
            "recommended_top10": rec_top10,
            "primary_type": primary,
            "bias_failure": flags["bias_failure"],
            "metadata_failure": flags["metadata_failure"],
        }
        errors_by_primary[primary].append(case)
        if flags["bias_failure"]:
            bias_count += 1
        if flags["metadata_failure"]:
            metadata_count += 1

    total_errors = sum(len(v) for v in errors_by_primary.values())
    retrieval_n = len(errors_by_primary.get("retrieval_failure", []))
    ranking_n = len(errors_by_primary.get("ranking_failure", []))

    examples = {}
    for p in ("retrieval_failure", "ranking_failure"):
        lst = errors_by_primary.get(p, [])
        if lst:
            examples[p] = random.sample(lst, min(n_examples_per_type, len(lst)))

    bias_cases = [c for lst in errors_by_primary.values() for c in lst if c.get("bias_failure")]
    if bias_cases:
        examples["bias_failure"] = random.sample(bias_cases, min(n_examples_per_type, len(bias_cases)))
    meta_cases = [c for lst in errors_by_primary.values() for c in lst if c.get("metadata_failure")]
    if meta_cases:
        examples["metadata_failure"] = random.sample(meta_cases, min(n_examples_per_type, len(meta_cases)))

    summary = {
        "total_error_cases": total_errors,
        "taxonomy": {
            "retrieval_failure": {
                "count": retrieval_n,
                "share_pct": round(100.0 * retrieval_n / total_errors, 1) if total_errors else 0,
                "description": "GT not in candidate pool (retrieval failure).",
                "example": examples.get("retrieval_failure", [{}])[0] if examples.get("retrieval_failure") else None,
            },
            "ranking_failure": {
                "count": ranking_n,
                "share_pct": round(100.0 * ranking_n / total_errors, 1) if total_errors else 0,
                "description": "GT in pool but reranker did not put it in top-K (ranking failure).",
                "example": examples.get("ranking_failure", [{}])[0] if examples.get("ranking_failure") else None,
            },
            "bias_failure": {
                "count": bias_count,
                "share_pct": round(100.0 * bias_count / total_errors, 1) if total_errors else 0,
                "description": "Top-K collapsed to head items (high exposure).",
                "example": examples.get("bias_failure", [{}])[0] if examples.get("bias_failure") else None,
            },
            "metadata_failure": {
                "count": metadata_count,
                "share_pct": round(100.0 * metadata_count / total_errors, 1) if total_errors else 0,
                "description": "GT or key items have missing/short title or description.",
                "example": examples.get("metadata_failure", [{}])[0] if examples.get("metadata_failure") else None,
            },
        },
        "run_id": rid,
        "n_examples_per_type": n_examples_per_type,
        "all_examples": examples,
    }
    return summary


def write_markdown_report(summary: Dict[str, Any], path: Path) -> None:

    total = summary.get("total_error_cases", 0)
    lines = [
        "# Error Analysis — Taxonomy",
        "",
        f"Total error cases (HR@10 = 0): **{total}**",
        f"Run: `{summary.get('run_id', 'N/A')}`",
        "",
        "## Taxonomy",
        "",
    ]
    if total == 0:
        lines.append("No error cases.")
        path.write_text("\n".join(lines), encoding="utf-8")
        return
    for typ, data in summary.get("taxonomy", {}).items():
        cnt = data.get("count", 0)
        pct = data.get("share_pct", 0)
        desc = data.get("description", "")
        lines.append(f"### {typ.replace('_', ' ').title()}")
        lines.append(f"- **Count:** {cnt} ({pct}%)")
        lines.append(f"- {desc}")
        ex = data.get("example")
        if ex:
            lines.append(f"- **Example:** user_id `{ex.get('user_id')}`")
            lines.append(f"  - GT items: `{ex.get('gt_items', [])[:5]}`")
            lines.append(f"  - Recommended top-10: `{ex.get('recommended_top10', [])[:5]}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def print_report(summary: Dict[str, Any]) -> None:
    total = summary.get("total_error_cases", 0)
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS — Taxonomy (for the paper)")
    print("=" * 60)
    print(f"\nTotal error cases (HR@10=0): {total}")
    if total == 0:
        print("No errors to analyze.")
        return
    tax = summary.get("taxonomy", {})
    for typ, data in tax.items():
        cnt = data.get("count", 0)
        pct = data.get("share_pct", 0)
        desc = data.get("description", "")
        print(f"\n--- {typ.replace('_', ' ').title()} ---")
        print(f"  Count: {cnt} ({pct}%)")
        print(f"  {desc}")
        ex = data.get("example")
        if ex:
            print(f"  Example: user_id={ex.get('user_id')}")
            print(f"    GT items: {ex.get('gt_items', [])[:5]}{'...' if len(ex.get('gt_items', [])) > 5 else ''}")
            print(f"    Recommended top-10: {ex.get('recommended_top10', [])[:5]}{'...' if len(ex.get('recommended_top10', [])) > 5 else ''}")
    print("\n" + "=" * 60)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Error analysis with taxonomy (retrieval / ranking / bias / metadata).")
    parser.add_argument("--run-id", type=str, default=None, help="Use specific run (default: first ours_with_reranker)")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--n-examples", type=int, default=3, help="Examples per type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for example sampling")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    try:
        master = load_master()
        gt = load_gt()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    summary = run_taxonomy(master, gt, run_id=args.run_id, n_examples_per_type=args.n_examples)
    print_report(summary)

    out_path = args.out / "error_analysis_taxonomy.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")

    md_path = args.out / "error_analysis_report.md"
    write_markdown_report(summary, md_path)
    print(f"Report: {md_path}")


if __name__ == "__main__":
    main()
