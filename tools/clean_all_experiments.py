
import argparse
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
EXPERIMENTS_DIR = ROOT / "experiments"


def collect_paths_to_remove(include_gt_and_splits: bool = False) -> Tuple[List[Path], List[Path]]:

    files: List[Path] = []
    dirs: List[Path] = []

    if RESULTS_DIR.exists():
        for f in RESULTS_DIR.glob("*.json"):
            files.append(f)

    if EXPERIMENTS_DIR.exists():
        for name in [
            "runs.jsonl",
            "master_results.json",
            "aggregated_results.json",
            "aggregated_results.csv",
        ]:
            p = EXPERIMENTS_DIR / name
            if p.exists():
                files.append(p)

        for f in EXPERIMENTS_DIR.glob("*_per_user.csv"):
            files.append(f)

        qv = EXPERIMENTS_DIR / "plots" / "quality_vs_time_summary.json"
        if qv.exists():
            files.append(qv)

    tables_dir = EXPERIMENTS_DIR / "tables"
    if tables_dir.exists():
        for f in tables_dir.iterdir():
            if f.is_file():
                files.append(f)


    plots_dir = EXPERIMENTS_DIR / "plots"
    if plots_dir.exists():
        for f in plots_dir.iterdir():
            if f.is_file():
                files.append(f)


    resources_dir = EXPERIMENTS_DIR / "resources"
    if resources_dir.exists():
        for f in resources_dir.iterdir():
            if f.is_file():
                files.append(f)

    ha_dir = EXPERIMENTS_DIR / "hypothesis_analysis"
    if ha_dir.exists():
        for f in ha_dir.iterdir():
            if f.is_file():
                files.append(f)

    score_dir = EXPERIMENTS_DIR / "score_analysis"
    if score_dir.exists():
        for f in score_dir.iterdir():
            if f.is_file():
                files.append(f)


    err_dir = EXPERIMENTS_DIR / "error_analysis"
    if err_dir.exists():
        for f in err_dir.iterdir():
            if f.is_file():
                files.append(f)

    stat_dir = EXPERIMENTS_DIR / "stat_tests"
    if stat_dir.exists():
        for f in stat_dir.iterdir():
            if f.is_file():
                files.append(f)


    cal_dir = EXPERIMENTS_DIR / "calibration"
    if cal_dir.exists():
        for f in cal_dir.iterdir():
            if f.is_file():
                files.append(f)

    if include_gt_and_splits:
        for name in [
            "ground_truth.json",
            "ground_truth_fixed.json",
            "ground_truth_original_backup.json",
            "split_metadata.json",
            "test_interactions.csv",
            "val_interactions.csv",
        ]:
            p = EXPERIMENTS_DIR / name
            if p.exists():
                files.append(p)

 
    for d in [plots_dir, resources_dir, ha_dir, score_dir, err_dir, stat_dir, cal_dir, tables_dir]:
        if d.exists() and d.is_dir():
            dirs.append(d)
    if RESULTS_DIR.exists() and RESULTS_DIR.is_dir():
        dirs.append(RESULTS_DIR)

    return files, dirs


def remove_all(dry_run: bool = False, include_gt_and_splits: bool = False) -> int:

    files, dirs = collect_paths_to_remove(include_gt_and_splits=include_gt_and_splits)
    removed = 0

    for f in files:
        if dry_run:
            print(f"  [would remove] {f}")
        else:
            try:
                f.unlink()
                print(f"  removed {f}")
                removed += 1
            except OSError as e:
                print(f"  warning: could not remove {f}: {e}")

    for d in dirs:
        if not d.exists():
            continue
        try:
            remaining = list(d.iterdir())
            if not remaining:
                if dry_run:
                    print(f"  [would rmdir] {d}")
                else:
                    d.rmdir()
                    print(f"  rmdir {d}")
                    removed += 1
        except OSError as e:
            print(f"  warning: could not rmdir {d}: {e}")

    return removed


def main():
    parser = argparse.ArgumentParser(
        description="Remove all experiment artifacts and metrics so you can run the pipeline from a clean state."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list what would be removed, do not delete.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Also remove ground_truth.json, split_metadata.json, test/val_interactions (full reset).",
    )
    args = parser.parse_args()

    print("Clean all experiment outputs")
    print("Root:", ROOT)
    if args.dry_run:
        print("DRY RUN — no files will be deleted.\n")
    if args.all:
        print("Including GT and split files (--all).\n")

    files, dirs = collect_paths_to_remove(include_gt_and_splits=args.all)
    print(f"Files to remove: {len(files)}")
    print(f"Dirs to remove (if empty): {len(dirs)}\n")

    removed = remove_all(dry_run=args.dry_run, include_gt_and_splits=args.all)

    if args.dry_run:
        print(f"\nDry run complete. Would remove {len(files)} files (and empty dirs).")
    else:
        print(f"\nCleanup complete. Removed {removed} items.")


if __name__ == "__main__":
    main()
