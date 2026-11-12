import json
import csv
from pathlib import Path
from typing import Dict, List, Set

GT_PATH = Path("experiments") / "ground_truth.json"
ITEMS_CSV = Path("data/processed/items_serendipity.csv")
OUTPUT_GT = Path("experiments") / "ground_truth_fixed.json"
OUTPUT_REPORT = Path("experiments") / "gt_fix_report.json"


def load_catalog_items() -> Set[str]:
    if not ITEMS_CSV.exists():
        raise FileNotFoundError(f"Items CSV not found: {ITEMS_CSV}")
    
    catalog_items = set()
    with open(ITEMS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_id = row.get("item_id", "")
            if item_id:
                catalog_items.add(str(item_id))
    
    return catalog_items


def fix_ground_truth():
    print("Loading ground truth and catalog...")
    
    with open(GT_PATH, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    
    gt_dict = gt_data.get("data", gt_data)
    catalog_items = load_catalog_items()
    
    print(f"Catalog items: {len(catalog_items)}")
    print(f"GT users: {len(gt_dict)}")
    
    fixed_gt = {}
    removed_users = []
    affected_users = []
    missing_items = set()
    
    for uid, gt_items in gt_dict.items():
        if not gt_items:
            continue
        
        gt_ids = [str(x) for x in gt_items]
        missing = [item_id for item_id in gt_ids if item_id not in catalog_items]
        
        if missing:
            missing_items.update(missing)
            affected_users.append(uid)
            
            valid_items = [item_id for item_id in gt_ids if item_id in catalog_items]
            
            if not valid_items:
                removed_users.append(uid)
            else:
                fixed_gt[uid] = valid_items
        else:
            fixed_gt[uid] = gt_ids
    
    report = {
        "original_users": len(gt_dict),
        "fixed_users": len(fixed_gt),
        "removed_users": len(removed_users),
        "affected_users": len(affected_users),
        "missing_items_count": len(missing_items),
        "missing_items": sorted(list(missing_items)),
        "removed_user_ids": sorted(removed_users)
    }
    
    output_data = {
        "data": fixed_gt,
        "_fixed": True,
        "_report": report
    }
    
    with open(OUTPUT_GT, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("GT Fix Report")
    print("="*60)
    print(f"Original users: {report['original_users']}")
    print(f"Fixed users: {report['fixed_users']}")
    print(f"Removed users: {report['removed_users']}")
    print(f"Affected users: {report['affected_users']}")
    print(f"Missing items: {report['missing_items_count']}")
    print(f"\nFixed GT saved to: {OUTPUT_GT}")
    print(f"Report saved to: {OUTPUT_REPORT}")
    
    if removed_users:
        print(f"\nRemoved user IDs (first 20): {removed_users[:20]}")
    
    return fixed_gt, report


def replace_gt_file(backup=True):
    import shutil
    
    if not OUTPUT_GT.exists():
        print("Error: ground_truth_fixed.json not found. Run fix_ground_truth() first.")
        return False
    
    if backup and GT_PATH.exists():
        backup_path = Path("experiments") / "ground_truth_original_backup.json"
        shutil.copy2(GT_PATH, backup_path)
        print(f"Backed up original GT to: {backup_path}")
    
    shutil.copy2(OUTPUT_GT, GT_PATH)
    print(f"Replaced {GT_PATH} with fixed version")
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix GT catalog issues")
    parser.add_argument("--replace", action="store_true", help="Automatically replace ground_truth.json with fixed version")
    parser.add_argument("--no-backup", action="store_true", help="Don't create backup of original GT")
    
    args = parser.parse_args()
    
    try:
        fix_ground_truth()
        
        if args.replace:
            replace_gt_file(backup=not args.no_backup)
            print("\n✅ GT file replaced! You can now re-run experiments.")
        else:
            print("\nNext steps:")
            print("1. Review gt_fix_report.json to see what was changed")
            print("2. Run with --replace flag to automatically replace:")
            print("   python -m tools.fix_gt_catalog --replace")
            print("   OR manually:")
            print("   - Backup: copy experiments/ground_truth.json to experiments/ground_truth_original_backup.json")
            print("   - Replace: copy experiments/ground_truth_fixed.json to experiments/ground_truth.json")
            print("3. Re-run experiments with fixed GT")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
