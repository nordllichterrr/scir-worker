
import argparse
from .run_experiment import run_with_logging
from .utils import logger
from .evaluation_config import N_TEST_USERS, EVAL_SEEDS


def run_debias_ablation(
    n_users: int = None,
    seeds=None,
    dataset: str = "serendipity",
):
   
    variants = [
        ("none", None),
        ("popularity_penalty", {"popularity_penalty_alpha": 0.3}),
        ("popularity_penalty_strong", {"popularity_penalty_alpha": 0.8}),
        ("exposure_penalty", {"exposure_beta": 0.1}),
        ("exposure_penalty_strong", {"exposure_beta": 0.5}),
        ("mmr", {"mmr_lambda": 0.4}),
        ("xquad", {"xquad_lambda": 0.5}),  
        ("xquad_strong", {"xquad_lambda": 0.7}),
    ]
    if n_users is None:
        n_users = N_TEST_USERS
    if seeds is None:
        seeds = EVAL_SEEDS

    logger.info("=" * 60)
    logger.info("Running DEBIAS ABLATION (variants=%s, dataset=%s)", [v[0] for v in variants], dataset)
    logger.info("=" * 60)

    for name, diversify_config in variants:
        for seed in seeds:
            run_id = f"ablation_debias_{name}_{dataset}_seed{seed}_n{n_users}"
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
    parser = argparse.ArgumentParser(description="Ablation over anti-bias (none / popularity / exposure / MMR).")
    parser.add_argument("--n-users", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--dataset", type=str, default="serendipity", choices=["serendipity", "taobao"])
    args = parser.parse_args()
    run_debias_ablation(
        n_users=args.n_users or N_TEST_USERS,
        seeds=args.seeds or EVAL_SEEDS,
        dataset=args.dataset,
    )
    logger.info("Debias ablation completed.")


if __name__ == "__main__":
    main()
