import argparse
from pathlib import Path

from .create_splits import build_splits
from .evaluation_config import get_default_raw_interactions_csv, _ds_key
from .utils import logger


def run_cv_for_dataset(dataset: str, cv_modes, n_folds: int, few_shot_n: int, split_seed: int = 42, csv_path_override: str = None):
    csv_path = Path(csv_path_override) if csv_path_override else get_default_raw_interactions_csv(dataset)
    if not Path(csv_path).exists():
        logger.warning("Skip dataset=%s, interactions csv not found: %s", dataset, csv_path)
        return
    for cv_mode in cv_modes:
        for fold_id in range(n_folds):
            logger.info("Build CV split: dataset=%s mode=%s fold=%d/%d", dataset, cv_mode, fold_id, n_folds)
            build_splits(
                interactions_path=Path(csv_path),
                dataset=dataset,
                cv_mode=cv_mode,
                n_folds=n_folds,
                fold_id=fold_id,
                few_shot_n=few_shot_n,
                split_seed=split_seed,
            )


def main():
    p = argparse.ArgumentParser(description="Build all cold-start CV folds for datasets.")
    p.add_argument("--datasets", nargs="+", default=["serendipity"], choices=["serendipity", "taobao", "movielens"])
    p.add_argument("--cv-modes", nargs="+", default=["user_kfold"], choices=["user_kfold", "item_kfold", "both_kfold"])
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--few-shot-n", type=int, default=0, help="Time-aware test-user few-shot observed interactions count.")
    p.add_argument("--split-seed", type=int, default=42, help="Random seed for fold assignment.")
    p.add_argument("--csv-path", type=str, default=None, help="Optional interactions CSV override (useful for smoke-tests).")
    args = p.parse_args()

    for ds in args.datasets:
        run_cv_for_dataset(
            ds,
            cv_modes=args.cv_modes,
            n_folds=args.n_folds,
            few_shot_n=args.few_shot_n,
            split_seed=args.split_seed,
            csv_path_override=args.csv_path,
        )

    logger.info("Done building CV folds for datasets=%s", args.datasets)


if __name__ == "__main__":
    main()
