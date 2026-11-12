
import argparse
from .run_experiment import run_with_logging
from .utils import logger
from .evaluation_config import N_TEST_USERS, EVAL_SEEDS


def run_debias_sweep(
    n_users: int = None,
    seeds=None,
    dataset: str = "serendipity",
):
    if n_users is None:
        n_users = N_TEST_USERS
    if seeds is None:

        seeds = EVAL_SEEDS[:3] if len(EVAL_SEEDS) >= 3 else EVAL_SEEDS

    popularity_alphas = [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]
  
    exposure_betas = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
    
    logger.info("=" * 60)
    logger.info("Running DEBIAS SWEEP (dataset=%s, n_users=%d)", dataset, n_users)
    logger.info("Popularity alphas: %s", popularity_alphas)
    logger.info("Exposure betas: %s", exposure_betas)
    logger.info("=" * 60)
    
    variants = []
    

    for alpha in popularity_alphas:
        variants.append((
            f"popularity_alpha_{alpha:.1f}".replace(".", "_"),
            {"popularity_penalty_alpha": alpha}
        ))
    
    for beta in exposure_betas:
        variants.append((
            f"exposure_beta_{beta:.2f}".replace(".", "_"),
            {"exposure_beta": beta}
        ))
    
    for name, diversify_config in variants:
        for seed in seeds:
            run_id = f"sweep_debias_{name}_{dataset}_seed{seed}_n{n_users}"
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
                    "diversify_config": diversify_config,
                },
                dataset=dataset,
            )


def main():
    parser = argparse.ArgumentParser(description="Sweep debias coefficients")
    parser.add_argument("--n-users", type=int, default=None, help="Number of test users")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Random seeds")
    parser.add_argument("--dataset", type=str, default="serendipity", choices=["serendipity", "taobao"])
    
    args = parser.parse_args()
    run_debias_sweep(
        n_users=args.n_users,
        seeds=args.seeds,
        dataset=args.dataset,
    )


if __name__ == "__main__":
    main()
