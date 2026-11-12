
import argparse
from .run_experiment import run_with_logging
from .utils import logger
from .evaluation_config import N_TEST_USERS, EVAL_SEEDS


def run_filter_seen_ablation(
    n_users: int = None,
    seeds=None,
    dataset: str = "serendipity",
):
    variants = [
        ("no_filter", False),
        ("filter_already_seen", True),
    ]
    if n_users is None:
        n_users = N_TEST_USERS
    if seeds is None:
        seeds = EVAL_SEEDS

    logger.info("=" * 60)
    logger.info("Running FILTER ALREADY SEEN ablation (dataset=%s)", dataset)
    logger.info("=" * 60)

    for name, filter_seen in variants:
        for seed in seeds:
            run_id = f"ablation_filter_seen_{name}_{dataset}_seed{seed}_n{n_users}"
            logger.info("Running: %s", run_id)
            run_with_logging(
                run_id=run_id,
                n_users=n_users,
                seed=seed,
                config={
                    "baseline": None,
                    "use_reranker": True,
                    "topk": 10,
                    "filter_already_seen": filter_seen,
                    "dataset": dataset,
                },
                dataset=dataset,
            )


def main():
    parser = argparse.ArgumentParser(description="Ablation: with vs without filter already seen.")
    parser.add_argument("--n-users", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--dataset", type=str, default="serendipity", choices=["serendipity", "taobao"])
    args = parser.parse_args()
    run_filter_seen_ablation(
        n_users=args.n_users or N_TEST_USERS,
        seeds=args.seeds or EVAL_SEEDS,
        dataset=args.dataset,
    )
    logger.info("Filter-already-seen ablation completed.")


if __name__ == "__main__":
    main()
