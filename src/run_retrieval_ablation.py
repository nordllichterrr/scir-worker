import argparse
from .run_experiment import run_with_logging
from .utils import logger
from .evaluation_config import N_TEST_USERS, EVAL_SEEDS, ABLATION_POOL_SIZES, ABLATION_RETRIEVAL_MODES


def run_retrieval_ablation(
    n_users: int = None,
    seeds=None,
    pool_sizes=None,
    dataset: str = "serendipity",
):
 
    if n_users is None:
        n_users = N_TEST_USERS
    if seeds is None:
        seeds = EVAL_SEEDS
    if pool_sizes is None:
        pool_sizes = ABLATION_POOL_SIZES  

    retrieval_modes = ABLATION_RETRIEVAL_MODES  

    logger.info("=" * 60)
    logger.info("Running RETRIEVAL ABLATION (modes=%s, pools=%s, dataset=%s)",
                retrieval_modes, pool_sizes, dataset)
    logger.info("=" * 60)

    for mode in retrieval_modes:
        for pool in pool_sizes:
            for seed in seeds:

                run_id = f"retrieval_{mode}_candidates_only_{dataset}_pool{pool}_seed{seed}_n{n_users}"
                logger.info("Running: %s", run_id)
                run_with_logging(
                    run_id=run_id,
                    n_users=n_users,
                    seed=seed,
                    config={
                        "baseline": None,
                        "use_reranker": False,
                        "topk": 10,
                        "candidate_pool_size": pool,
                        "retrieval_mode": mode,
                        "dataset": dataset,
                    },
                    dataset=dataset,
                )


                run_id = f"retrieval_{mode}_with_reranker_{dataset}_pool{pool}_seed{seed}_n{n_users}"
                logger.info("Running: %s", run_id)
                run_with_logging(
                    run_id=run_id,
                    n_users=n_users,
                    seed=seed,
                    config={
                        "baseline": None,
                        "use_reranker": True,
                        "topk": 10,
                        "candidate_pool_size": pool,
                        "rerank_pool_size": pool,
                        "retrieval_mode": mode,
                        "dataset": dataset,
                    },
                    dataset=dataset,
                )


def main():
    parser = argparse.ArgumentParser(description="Run retrieval + pool-size ablation.")
    parser.add_argument("--n-users", type=int, default=None, help="Number of test users (default: %d)" % N_TEST_USERS)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Random seeds (default: from evaluation_config)",
    )
    parser.add_argument(
        "--pool-sizes",
        nargs="+",
        type=int,
        default=None,
        help="Candidate pool sizes (default: 100,300,1000,5000)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="serendipity",
        choices=["serendipity", "taobao"],
        help="Dataset key (for logging and GT selection).",
    )

    args = parser.parse_args()
    run_retrieval_ablation(
        n_users=args.n_users if args.n_users is not None else N_TEST_USERS,
        seeds=args.seeds if args.seeds is not None else EVAL_SEEDS,
        pool_sizes=args.pool_sizes if args.pool_sizes is not None else ABLATION_POOL_SIZES,
        dataset=args.dataset,
    )

    logger.info("=" * 60)
    logger.info("Retrieval ablation completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

