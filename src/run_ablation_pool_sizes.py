import argparse
from .run_experiment import run_with_logging
from .utils import logger
from .evaluation_config import N_TEST_USERS, EVAL_SEEDS, ABLATION_POOL_SIZES


def run_pool_size_ablation(n_users=None, seeds=None, pool_sizes=None, dataset: str = "serendipity"):
    if n_users is None:
        n_users = N_TEST_USERS
    if seeds is None:
        seeds = EVAL_SEEDS
    if pool_sizes is None:
        pool_sizes = ABLATION_POOL_SIZES  
    logger.info("=" * 60)
    logger.info("Running POOL SIZE ABLATION (n_users=%d, pools=%s, dataset=%s)", n_users, pool_sizes, dataset)
    logger.info("=" * 60)

    for pool_size in pool_sizes:
        for seed in seeds:
            run_id = f"ablation_rerank_pool{pool_size}_{dataset}_seed{seed}_n{n_users}"
            logger.info("Running: %s", run_id)
            run_with_logging(
                run_id=run_id,
                n_users=n_users,
                seed=seed,
                config={
                    "baseline": None,
                    "use_reranker": True,
                    "topk": 10,
                    "candidate_pool_size": pool_size,
                    "rerank_pool_size": pool_size,
                    "dataset": dataset,
                },
                dataset=dataset,
            )


def main():
    parser = argparse.ArgumentParser(description="Run pool size ablation study")
    parser.add_argument("--n-users", type=int, default=None, help="Number of test users (default: %d)" % N_TEST_USERS)
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Random seeds (default: from evaluation_config)")
    parser.add_argument("--pool-sizes", nargs="+", type=int, default=None, help="Candidate pool sizes (default: 100,300,1000,5000)")
    parser.add_argument("--dataset", type=str, default="serendipity", choices=["serendipity", "taobao"])
    args = parser.parse_args()
    n_users = args.n_users if args.n_users is not None else N_TEST_USERS
    seeds = args.seeds if args.seeds is not None else EVAL_SEEDS
    pool_sizes = args.pool_sizes if args.pool_sizes is not None else ABLATION_POOL_SIZES
    run_pool_size_ablation(n_users=n_users, seeds=seeds, pool_sizes=pool_sizes, dataset=args.dataset)
    
    logger.info("=" * 60)
    logger.info("Pool size ablation completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
