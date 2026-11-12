
import argparse
from .run_experiment import run_with_logging
from .utils import logger
from .evaluation_config import N_TEST_USERS, EVAL_SEEDS


def run_profile_ablation(
    n_users: int = None,
    seeds=None,
    dataset: str = "serendipity",
):
    profile_types = ["last_k_items", "summary", "centroid"]
    if n_users is None:
        n_users = N_TEST_USERS
    if seeds is None:
        seeds = EVAL_SEEDS

    logger.info("=" * 60)
    logger.info("Running PROFILE ABLATION (types=%s, dataset=%s)", profile_types, dataset)
    logger.info("=" * 60)

    for profile_type in profile_types:
        for seed in seeds:
            run_id = f"ablation_profile_{profile_type}_{dataset}_seed{seed}_n{n_users}"
            logger.info("Running: %s", run_id)
            run_with_logging(
                run_id=run_id,
                n_users=n_users,
                seed=seed,
                config={
                    "baseline": None,
                    "use_reranker": True,
                    "topk": 10,
                    "dataset": dataset,
                    "profile_type": profile_type,
                    "use_viewed_items_profile": True,
                },
                dataset=dataset,
            )


def main():
    parser = argparse.ArgumentParser(description="Ablation over user profile type (centroid / summary / last_k_items).")
    parser.add_argument("--n-users", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--dataset", type=str, default="serendipity", choices=["serendipity", "taobao"])
    args = parser.parse_args()
    run_profile_ablation(
        n_users=args.n_users or N_TEST_USERS,
        seeds=args.seeds or EVAL_SEEDS,
        dataset=args.dataset,
    )
    logger.info("Profile ablation completed.")


if __name__ == "__main__":
    main()
