import argparse
from .run_experiment import run_with_logging
from .utils import logger
from .evaluation_config import N_TEST_USERS, EVAL_SEEDS


TRAINABLE_METHODS = {"mf", "finetuned_reranker", "two_tower_trainable"}


def run_baselines(n_users=None, seeds=None, split_seeds=None, init_seeds=None, dataset: str = "serendipity"):
    if n_users is None:
        n_users = N_TEST_USERS
    if seeds is None:
        seeds = EVAL_SEEDS
    if split_seeds is None:
        split_seeds = [42]
    if init_seeds is None:
        init_seeds = [42]
    logger.info("=" * 60)
    logger.info("Running BASELINE experiments (n_users=%d, seeds=%s, split_seeds=%s, init_seeds=%s)", n_users, seeds, split_seeds, init_seeds)
    logger.info("=" * 60)
    

    baselines = [
        "random",
        "popularity",
        "content_bm25",       
        "two_tower",         
        "embedding_cosine", 
        "itemknn",
        "ease",
        "mf",
    ]
    
    for baseline in baselines:
        baseline_init_seeds = init_seeds if baseline in TRAINABLE_METHODS else [42]
        for split_seed in split_seeds:
            for seed in seeds:
                for init_seed in baseline_init_seeds:
                    run_id = f"baseline_{baseline}_{dataset}_split{split_seed}_seed{seed}_init{init_seed}_n{n_users}"
                    logger.info("Running: %s", run_id)
                    run_with_logging(
                        run_id=run_id,
                        n_users=n_users,
                        seed=seed,
                        split_seed=split_seed,
                        init_seed=init_seed,
                        config={
                            "baseline": baseline,
                            "use_reranker": False,
                            "topk": 10,
                            "dataset": dataset,
                            "trainable_component": baseline in TRAINABLE_METHODS,
                        },
                        dataset=dataset,
                    )


def run_sanity_check_baselines(n_users=None, seeds=None, split_seeds=None, dataset: str = "serendipity"):

    if n_users is None:
        n_users = N_TEST_USERS
    if seeds is None:
        seeds = EVAL_SEEDS
    if split_seeds is None:
        split_seeds = [42]
    logger.info("=" * 60)
    logger.info("Running SANITY CHECK baselines (n_users=%d, seeds=%s)", n_users, seeds)
    logger.info("=" * 60)
    
    sanity_baselines = ["oracle_upper_bound", "random_in_candidate_pool"]
    
    for baseline in sanity_baselines:
        for split_seed in split_seeds:
            for seed in seeds:
                run_id = f"sanity_{baseline}_{dataset}_split{split_seed}_seed{seed}_init42_n{n_users}"
                logger.info("Running: %s", run_id)
                if baseline == "random_in_candidate_pool":
                    retrieval_mode = "bm25"
                else:
                    retrieval_mode = "ann"
                run_with_logging(
                    run_id=run_id,
                    n_users=n_users,
                    seed=seed,
                    split_seed=split_seed,
                    init_seed=42,
                    config={
                        "baseline": baseline,
                        "use_reranker": False,
                        "topk": 10,
                        "candidate_pool_size": 1000,  
                        "retrieval_mode": retrieval_mode,
                        "dataset": dataset,
                        "trainable_component": False,
                    },
                    dataset=dataset,
                )


def run_ablation_study(n_users=None, seeds=None, split_seeds=None, init_seeds=None, dataset: str = "serendipity"):
    if n_users is None:
        n_users = N_TEST_USERS
    if seeds is None:
        seeds = EVAL_SEEDS
    if split_seeds is None:
        split_seeds = [42]
    if init_seeds is None:
        init_seeds = [42]
    logger.info("=" * 60)
    logger.info("Running ABLATION study (n_users=%d, seeds=%s)", n_users, seeds)
    logger.info("=" * 60)
    
    configs = [
        {
            "name": f"candidates_only_{dataset}",
            "config": {
                "baseline": None,
                "use_reranker": False,
                "topk": 10,
                "retrieval_mode": "ann",
                "dataset": dataset,
            },
        },
        {
            "name": f"with_reranker_{dataset}",
            "config": {
                "baseline": None,
                "use_reranker": True,
                "topk": 10,
                "retrieval_mode": "ann",
                "dataset": dataset,
            },
        },
        {
            "name": f"candidates_only_hybrid_{dataset}",
            "config": {
                "baseline": None,
                "use_reranker": False,
                "topk": 10,
                "retrieval_mode": "hybrid",
                "dataset": dataset,
            },
        },
        {
            "name": f"with_reranker_hybrid_{dataset}",
            "config": {
                "baseline": None,
                "use_reranker": True,
                "topk": 10,
                "retrieval_mode": "hybrid",
                "dataset": dataset,
            },
        },
    ]
    
    for cfg in configs:
        trainable = bool((cfg["config"] or {}).get("trainable_component", False))
        cfg_init_seeds = init_seeds if trainable else [42]
        for split_seed in split_seeds:
            for seed in seeds:
                for init_seed in cfg_init_seeds:
                    run_id = f"ablation_{cfg['name']}_split{split_seed}_seed{seed}_init{init_seed}_n{n_users}"
                    logger.info("Running: %s", run_id)
                    run_with_logging(
                        run_id=run_id,
                        n_users=n_users,
                        seed=seed,
                        split_seed=split_seed,
                        init_seed=init_seed,
                        config=cfg["config"],
                        dataset=dataset,
                    )


def main():
    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument("--n-users", type=int, default=None, help="Number of test users (default: from evaluation_config, %d)" % N_TEST_USERS)
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Random seeds (default: from evaluation_config)")
    parser.add_argument("--split-seeds", nargs="+", type=int, default=[42], help="Data split seeds for fold assignment protocol.")
    parser.add_argument("--init-seeds", nargs="+", type=int, default=[42], help="Model initialization seeds (used only for trainable methods).")
    parser.add_argument("--baselines-only", action="store_true", help="Run only baselines")
    parser.add_argument("--ablation-only", action="store_true", help="Run only ablation study")
    parser.add_argument("--sanity-only", action="store_true", help="Run only sanity check baselines (oracle_upper_bound, random_in_candidate_pool)")
    parser.add_argument(
        "--dataset",
        type=str,
        default="serendipity",
        choices=["serendipity", "taobao", "movielens"],
        help="Dataset key (controls ground-truth selection and tagging).",
    )
    
    args = parser.parse_args()
    n_users = args.n_users if args.n_users is not None else N_TEST_USERS
    seeds = args.seeds if args.seeds is not None else EVAL_SEEDS
    split_seeds = args.split_seeds if args.split_seeds is not None else [42]
    init_seeds = args.init_seeds if args.init_seeds is not None else [42]

    if args.baselines_only:
        run_baselines(n_users=n_users, seeds=seeds, split_seeds=split_seeds, init_seeds=init_seeds, dataset=args.dataset)
    elif args.ablation_only:
        run_ablation_study(n_users=n_users, seeds=seeds, split_seeds=split_seeds, init_seeds=init_seeds, dataset=args.dataset)
    elif args.sanity_only:
        run_sanity_check_baselines(n_users=n_users, seeds=seeds, split_seeds=split_seeds, dataset=args.dataset)
    else:
        run_baselines(n_users=n_users, seeds=seeds, split_seeds=split_seeds, init_seeds=init_seeds, dataset=args.dataset)
        run_ablation_study(n_users=n_users, seeds=seeds, split_seeds=split_seeds, init_seeds=init_seeds, dataset=args.dataset)
    
    logger.info("=" * 60)
    logger.info("All experiments completed!")
    logger.info("Results logged to: experiments/runs.jsonl")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
