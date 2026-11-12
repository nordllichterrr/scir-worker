from pathlib import Path

# =========================
# Global evaluation settings
# =========================
N_TEST_USERS = 500
EVAL_SEEDS = [42, 7, 123, 2024, 2025, 0, 13, 999, 1234, 777]
N_BOOTSTRAP = 1000

N_QUICK_TEST_USERS = 50
N_SANITY_USERS = 5
N_FAST_USERS = 30

FAST_SEEDS = EVAL_SEEDS[:5]
FAST_POOL_SIZES = [200, 500]

ABLATION_POOL_SIZES = [100, 300, 1000, 5000]
ABLATION_RETRIEVAL_MODES = ["ann", "bm25", "hybrid"]

SPLIT_METHOD = "time_based"  # "time_based" or "random"
SPLIT_RATIOS = (0.7, 0.1, 0.2)

COLD_START_SCENARIOS = ("new_users", "new_items", "both")
NEW_USER_MAX_TRAIN = 0

INTERACTION_BUCKETS = [
    (1, 2),
    (3, 5),
    (6, 10),
    (11, 20),
    (21, None),
]

NEGATIVE_SAMPLING_PROTOCOL = "full_ranking"
N_NEGATIVES_PER_POSITIVE = 100
NEGATIVE_SAMPLING_SEED = 42


# =========================
# Dataset-aware paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


def normalize_dataset_key(dataset: str = "serendipity") -> str:
    """Normalize supported dataset names to canonical slugs."""
    d = str(dataset or "serendipity").strip().lower()

    if d.startswith("serendipity"):
        return "serendipity"
    if d.startswith("taobao"):
        return "taobao"
    if d.startswith("movielens") or d.startswith("ml") or "movie" in d:
        return "movielens"

    raise ValueError(
        f"Unsupported dataset '{dataset}'. "
        "Expected Serendipity, Taobao, or MovieLens."
    )


# Backward-compatible alias for existing internal callers.
_ds_key = normalize_dataset_key


def get_eval_paths(dataset: str = "serendipity") -> dict:
    """
    Returns dataset-specific paths for splits + GT so datasets never overwrite each other.
    """
    ds = normalize_dataset_key(dataset)
    return {
        "gt": EXPERIMENTS_DIR / f"ground_truth_{ds}.json",
        "split_meta": EXPERIMENTS_DIR / f"split_metadata_{ds}.json",
        "train": EXPERIMENTS_DIR / f"training_interactions_{ds}.csv",
        "val": EXPERIMENTS_DIR / f"val_interactions_{ds}.csv",
        "test": EXPERIMENTS_DIR / f"test_interactions_{ds}.csv",
    }


def get_default_raw_interactions_csv(dataset: str = "serendipity") -> Path:
    """
    Where to read RAW interactions from (before standard split files exist).
    """
    ds = normalize_dataset_key(dataset)
    if ds == "taobao":
        # Produced by preprocess.py
        return PROCESSED_DIR / "interactions_taobao.csv"
    if ds == "movielens":
        # Produced by preprocess.py
        return PROCESSED_DIR / "interactions_movielens.csv"

    return DATA_DIR / "serendipity-sac2018" / "training.csv"