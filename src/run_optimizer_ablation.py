import argparse
import subprocess
import sys
from pathlib import Path

from .evaluation_config import N_TEST_USERS, N_FAST_USERS, EVAL_SEEDS, FAST_SEEDS
from .run_experiment import run_with_logging
from .utils import logger


def main():
    p = argparse.ArgumentParser(description="Train/evaluate finetuned reranker with different optimizers.")
    p.add_argument("--dataset", type=str, default="serendipity", choices=["serendipity", "taobao", "movielens"])
    p.add_argument("--n-users", type=int, default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--init-seeds", nargs="+", type=int, default=[42, 7, 2024])
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-samples", type=int, default=5000)
    p.add_argument("--dry-run", action="store_true", help="Skip fit/eval and only validate commands.")
    args = p.parse_args()

    n_users = args.n_users if args.n_users is not None else N_TEST_USERS
    seeds = args.seeds if args.seeds is not None else EVAL_SEEDS
    optimizers = ["adamw", "sgd", "adafactor"]
    root = Path(__file__).resolve().parents[1]

    for opt in optimizers:
        for init_seed in args.init_seeds:
            out_dir = root / "models" / f"crossencoder_finetuned_{opt}_init{init_seed}"
            cmd = [
                sys.executable,
                "-m",
                "tools.fine_tune_crossencoder",
                "--optimizer",
                opt,
                "--init-seed",
                str(init_seed),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--max-samples",
                str(args.max_samples),
                "--output-dir",
                str(out_dir),
            ]
            if args.dry_run:
                cmd.append("--dry-run")
            logger.info("Optimizer ablation train step: %s", " ".join(cmd))
            rc = subprocess.run(cmd, cwd=root).returncode
            if rc != 0:
                raise SystemExit(f"Failed optimizer training step: optimizer={opt}, init_seed={init_seed}, rc={rc}")

            if args.dry_run:
                continue
            for seed in seeds:
                run_id = f"optablation_{opt}_{args.dataset}_seed{seed}_init{init_seed}_n{n_users}"
                run_with_logging(
                    run_id=run_id,
                    n_users=n_users,
                    seed=seed,
                    split_seed=42,
                    init_seed=init_seed,
                    config={
                        "baseline": None,
                        "use_reranker": True,
                        "topk": 10,
                        "dataset": args.dataset,
                        "reranker_model": str(out_dir),
                        "reranker_ablation": f"finetuned_{opt}",
                        "trainable_component": True,
                        "optimizer_name": opt,
                    },
                    dataset=args.dataset,
                )

    logger.info("Optimizer ablation completed.")


if __name__ == "__main__":
    main()
