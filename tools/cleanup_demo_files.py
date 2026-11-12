
import json
from pathlib import Path

RESULTS_DIR = Path("results")
EXPERIMENTS_DIR = Path("experiments")

def cleanup():
  
    removed = []
    
    demo_patterns = [
        "demo_rankings.json",
        "demo_sanity.json",
        "last_run_metadata.json",
        "candidates_*.json",
        "popularity_demo_*.json"
    ]
    
    if RESULTS_DIR.exists():
        for pattern in demo_patterns:
            if "*" in pattern:
        
                for f in RESULTS_DIR.glob(pattern):
                    f.unlink()
                    removed.append(str(f))
            else:

                f = RESULTS_DIR / pattern
                if f.exists():
                    f.unlink()
                    removed.append(str(f))

    demo_eval_patterns = [
        "eval_demo_*.csv",
        "eval_demo_*.json",
        "eval_last_run_metadata_*.csv",
        "eval_last_run_metadata_*.json",
        "config_demo.json"
    ]
    
    if EXPERIMENTS_DIR.exists():
        for pattern in demo_eval_patterns:
            for f in EXPERIMENTS_DIR.glob(pattern):
                f.unlink()
                removed.append(str(f))
    
    print(f"Removed {len(removed)} files:")
    for f in removed[:20]:  
        print(f"  - {f}")
    if len(removed) > 20:
        print(f"  ... and {len(removed) - 20} more")
    
    return removed

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--dry-run":
        print("DRY RUN - would remove:")
        removed = []
        RESULTS_DIR = Path("results")
        EXPERIMENTS_DIR = Path("experiments")
        
        for pattern in ["demo_rankings.json", "demo_sanity.json", "last_run_metadata.json", "candidates_*.json", "popularity_demo_*.json"]:
            if "*" in pattern:
                for f in RESULTS_DIR.glob(pattern):
                    removed.append(str(f))
            else:
                f = RESULTS_DIR / pattern
                if f.exists():
                    removed.append(str(f))
        
        for pattern in ["eval_demo_*.csv", "eval_demo_*.json", "eval_last_run_metadata_*.csv", "eval_last_run_metadata_*.json", "config_demo.json"]:
            for f in EXPERIMENTS_DIR.glob(pattern):
                removed.append(str(f))
        
        for f in removed:
            print(f"  - {f}")
        print(f"\nTotal: {len(removed)} files")
    else:
        removed = cleanup()
        print(f"\nCleanup complete. Removed {len(removed)} files.")
