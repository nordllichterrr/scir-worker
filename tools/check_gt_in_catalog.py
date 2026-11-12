
import csv, json
from pathlib import Path

ITEMS_CSV = Path("data/processed/items_serendipity.csv")
GT_PATH = Path("experiments") / "ground_truth.json"

def load_items():
    ids = set()
    with open(ITEMS_CSV, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            # try common column names
            if 'item_id' in row and row['item_id']:
                ids.add(str(row['item_id']))
            elif 'id' in row and row['id']:
                ids.add(str(row['id']))
    return ids

def load_gt():
    raw = json.load(open(GT_PATH, encoding='utf-8'))
    if isinstance(raw, dict) and "data" in raw:
        raw = raw["data"]
    all_gt = set()
    for uid, lst in raw.items():
        if not lst: continue
        if isinstance(lst[0], dict):
            for d in lst:
                v = d.get('item_id') or d.get('id') or d.get('doc_id')
                if v: all_gt.add(str(v))
        else:
            for v in lst:
                all_gt.add(str(v))
    return all_gt

def main():
    items = load_items()
    gt = load_gt()
    missing = sorted(list(gt - items))
    print("GT total ids:", len(gt))
    print("Items total ids:", len(items))
    print("GT ids missing from items:", len(missing))
    print("Examples missing (up to 30):", missing[:30])

if __name__ == "__main__":
    main()
