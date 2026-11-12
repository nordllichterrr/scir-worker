import argparse

from .evaluation_config import N_TEST_USERS, EVAL_SEEDS
from .run_experiment import run_with_logging
from .utils import logger


def run_preference_prior_ablation(n_users: int, seeds, dataset: str = "serendipity", dry_run: bool = False):
    modes = ["no_prior", "prior_only", "prior_plus_context"]
    logger.info("Running PREFERENCE PRIOR ABLATION modes=%s dataset=%s", modes, dataset)
    for mode in modes:
        for seed in seeds:
            run_id = f"ablation_prefprior_{mode}_{dataset}_seed{seed}_n{n_users}"
            if dry_run:
                logger.info("Dry-run: would execute %s mode=%s", run_id, mode)
                continue
            run_with_logging(
                run_id=run_id,
                n_users=n_users,
                seed=seed,
                split_seed=42,
                init_seed=42,
                config={
                    "baseline": None,
                    "use_reranker": True,
                    "topk": 10,
                    "dataset": dataset,
                    "use_viewed_items_profile": True,
                    "profile_type": "summary",
                    "retrieval_mode": "hybrid",
                    "preference_prior_mode": mode,
                    "trainable_component": False,
                },
                dataset=dataset,
            )


def main():
    p = argparse.ArgumentParser(
        description="Ablation over preference prior mode (no_prior / prior_only / prior_plus_context)."
    )
    p.add_argument("--n-users", type=int, default=N_TEST_USERS)
    p.add_argument("--seeds", nargs="+", type=int, default=EVAL_SEEDS)
    p.add_argument("--dataset", type=str, default="serendipity", choices=["serendipity", "taobao", "movielens"])
    p.add_argument("--dry-run", action="store_true", help="Validate generated runs without executing retrieval/reranking.")
    args = p.parse_args()
    run_preference_prior_ablation(n_users=args.n_users, seeds=args.seeds, dataset=args.dataset, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
