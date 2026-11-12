import argparse
from pathlib import Path

from .run_experiment import run_with_logging
from .utils import logger
from .evaluation_config import N_TEST_USERS, EVAL_SEEDS


def run_reranker_variants(
    n_users: int = None,
    seeds=None,
    dataset: str = "serendipity",
):
    
    if n_users is None:
        n_users = N_TEST_USERS
    if seeds is None:
        seeds = EVAL_SEEDS

    configs = [
        {
            "name": "ce_zeroshot",
            "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "reranker_ablation": "zero_shot",
        },
        {
            "name": "ce_light",
            "reranker_model": "cross-encoder/ms-marco-TinyBERT-L-6",
            "reranker_ablation": "zero_shot",
        },
    ]

    finetuned_path = Path("models") / "crossencoder_finetuned"
    if finetuned_path.exists():
        configs.append(
            {
                "name": "ce_finetuned",
                "reranker_model": str(finetuned_path),
                "reranker_ablation": "finetuned",
            }
        )
    else:
        logger.warning(
            "Fine-tuned cross-encoder not found at %s. Run: python -m tools.fine_tune_crossencoder --use-hard-negatives",
            finetuned_path,
        )

    logger.info("=" * 60)
    logger.info(
        "Running RERANKER VARIANTS (dataset=%s, configs=%s)", dataset, [c["name"] for c in configs]
    )
    logger.info("=" * 60)

    for cfg in configs:
        for seed in seeds:
            run_id = (
                f"reranker_{cfg['name']}_{dataset}_seed{seed}_n{n_users}"
            )
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
                    "reranker_model": cfg["reranker_model"],
                    "reranker_ablation": cfg.get("reranker_ablation", "zero_shot"),
                   
                    "reranker_device": "cpu",  
                    "reranker_fp16": False, 
                    "reranker_max_length": None, 
                },
                dataset=dataset,
            )


def main():
    parser = argparse.ArgumentParser(
        description="Run ablation over CrossEncoder reranker variants."
    )
    parser.add_argument("--n-users", type=int, default=None, help="Number of test users (default: %d)" % N_TEST_USERS)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Random seeds (default: from evaluation_config)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="serendipity",
        choices=["serendipity", "taobao"],
        help="Dataset key (for logging and GT selection).",
    )

    args = parser.parse_args()
    run_reranker_variants(
        n_users=args.n_users if args.n_users is not None else N_TEST_USERS,
        seeds=args.seeds if args.seeds is not None else EVAL_SEEDS,
        dataset=args.dataset,
    )

    logger.info("=" * 60)
    logger.info("Reranker variants ablation completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

