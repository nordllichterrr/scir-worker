import hashlib
import json
import mimetypes
import os
import platform
import sys
import traceback
from pathlib import Path
from urllib.request import Request, urlopen

from .celery_app import app

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = Path(
    os.environ.get("SCIR_ARTIFACTS_DIR", ROOT / "worker_artifacts")
)
DJANGO_URL = os.environ.get("DJANGO_INTERNAL_API_URL", "").rstrip("/")
WORKER_TOKEN = os.environ.get("WORKER_API_TOKEN", "")


def api_request(run_id, method="GET", payload=None):
    if not DJANGO_URL or not WORKER_TOKEN:
        raise RuntimeError("Worker API environment is not configured.")

    url = (
        f"{DJANGO_URL}/api/lab/v1/"
        f"internal/worker/runs/{run_id}/"
    )

    body = (
        None
        if payload is None
        else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    )

    request = Request(
        url,
        data=body,
        method=method,
        headers={
            "X-Worker-Token": WORKER_TOKEN,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )

    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def artifact_metadata(path, kind):
    path = Path(path)

    if not path.is_file():
        return None

    try:
        storage_key = str(
            path.resolve().relative_to(ARTIFACTS_ROOT.resolve())
        )
    except ValueError:
        try:
            storage_key = "project/" + str(
                path.resolve().relative_to(ROOT.resolve())
            )
        except ValueError:
            storage_key = "external/" + path.name

    mime_type = (
        mimetypes.guess_type(path.name)[0]
        or "application/octet-stream"
    )

    return {
        "name": path.name,
        "kind": kind,
        "storage_key": storage_key,
        "mime_type": mime_type,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "metadata": {
            "worker_local_path": str(path.resolve()),
        },
    }


def extract_metrics(record):
    rows = []

    def add_metrics(
        values,
        stage="overall",
        dimension="global",
        dimension_key="",
    ):
        if not isinstance(values, dict):
            return

        for name, item in values.items():
            if not isinstance(item, dict):
                continue

            mean = item.get("mean", item.get("value"))

            if mean is None:
                continue

            rows.append({
                "name": str(name),
                "stage": stage,
                "dimension": dimension,
                "dimension_key": str(dimension_key),
                "value": mean,
                "std": item.get("std"),
                "ci95_low": item.get("ci95_low"),
                "ci95_high": item.get("ci95_high"),
                "sample_count": item.get(
                    "n_users",
                    item.get("sample_count"),
                ),
                "metadata": {
                    key: value
                    for key, value in item.items()
                    if key not in {
                        "mean",
                        "value",
                        "std",
                        "ci95_low",
                        "ci95_high",
                        "n_users",
                        "sample_count",
                    }
                },
            })

    add_metrics(record.get("metrics", {}), "overall")
    add_metrics(record.get("metrics_pre", {}), "pre")
    add_metrics(record.get("metrics_post", {}), "post")

    for field, dimension in (
        ("by_bucket", "bucket"),
        ("by_scenario", "scenario"),
    ):
        groups = record.get(field, {})

        if not isinstance(groups, dict):
            continue

        for key, values in groups.items():
            add_metrics(
                values,
                stage="overall",
                dimension=dimension,
                dimension_key=str(key),
            )

    return rows


def read_last_jsonl_record(path):
    path = Path(path)

    if not path.exists():
        return {}

    records = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not records:
        return {}

    return json.loads(records[-1])


def publish_log(message, level="info", stage="worker", sequence=1):
    return {
        "sequence": sequence,
        "level": level,
        "stage": stage,
        "message": str(message)[:10000],
    }


def resolve_fold_file(raw_path, field_name):
    """Resolve a fold file and prevent paths outside the Worker repository."""
    if not raw_path:
        raise ValueError(f"Missing required fold path: {field_name}")

    candidate = Path(str(raw_path)).expanduser()
    if not candidate.is_absolute():
        candidate = ROOT / candidate

    root = ROOT.resolve()
    resolved = candidate.resolve()

    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError(
            f"Fold path '{field_name}' escapes Worker repository: {raw_path}"
        )

    if not resolved.is_file():
        raise FileNotFoundError(
            f"Fold file '{field_name}' does not exist: {resolved}"
        )

    return resolved


@app.task(bind=True, name="scir_worker.tasks.execute_run")
def execute_run(self, run_id):
    run_dir = ARTIFACTS_ROOT / str(run_id)
    results_dir = run_dir / "results"

    run_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    logs = []

    def log(message, level="info", stage="worker"):
        logs.append(
            publish_log(
                message,
                level=level,
                stage=stage,
                sequence=len(logs) + 1,
            )
        )

    try:
        run = api_request(run_id)

        if run.get("status") == "cancelled":
            return {"status": "cancelled"}

        api_request(
            run_id,
            "POST",
            {
                "status": "running",
                "progress": 1,
                "stage": "initializing",
                "worker_version": "0.1.0",
                "environment_snapshot": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "worker": "scir_worker",
                },
                "logs": [
                    publish_log(
                        "Worker accepted the experiment.",
                        stage="initializing",
                    )
                ],
            },
        )

        log("Loading SCIR pipeline.", stage="initializing")

        sys.path.insert(0, str(ROOT))

        from src.run_experiment import run_with_logging

        config = dict(run.get("config_snapshot") or {})
        config["results_dir"] = str(results_dir)
        config["per_user_metrics_path"] = str(
            run_dir / f"{run_id}_per_user_metrics.csv"
        )
        config["runs_log_path"] = str(run_dir / "runs.jsonl")
        config.setdefault("topk", int(run.get("topk") or 10))

        dataset = run["dataset_slug"]
        if dataset not in {"serendipity", "taobao", "movielens"}:
            raise ValueError(
                f"Unsupported SCIR dataset slug: {dataset}"
            )

        fold_metadata = run.get("fold_metadata") or {}
        if not isinstance(fold_metadata, dict):
            raise ValueError("fold_metadata must be a JSON object.")

        split_type = str(fold_metadata.get("split_type") or "standard")
        allowed_split_types = {
            "standard", "user_kfold", "item_kfold", "both_kfold"
        }
        if split_type not in allowed_split_types:
            raise ValueError(f"Unsupported split_type: {split_type}")

        fold_index = run.get("fold_index")
        config["dataset"] = dataset
        config["split_type"] = split_type

        if fold_index is not None:
            config["fold_index"] = int(fold_index)

        fold_path_fields = {
            "train_interactions_path": "train_interactions_path",
            "validation_interactions_path": "validation_interactions_path",
            "test_interactions_path": "test_interactions_path",
            "ground_truth_path": "ground_truth_path",
            "split_metadata_path": "split_metadata_path",
        }

        if split_type != "standard":
            missing = [
                key for key in fold_path_fields
                if not fold_metadata.get(key)
            ]
            if missing:
                raise ValueError(
                    f"CV fold metadata is missing paths: {missing}"
                )

        resolved_fold_paths = {}
        for metadata_key, config_key in fold_path_fields.items():
            raw_path = fold_metadata.get(metadata_key)
            if raw_path:
                resolved_fold_paths[config_key] = str(
                    resolve_fold_file(raw_path, metadata_key)
                )

        config.update(resolved_fold_paths)

        log(
            f"Starting SCIR on dataset '{dataset}', "
            f"split_type='{split_type}', fold_index={fold_index}.",
            stage="execution",
        )

        for field_name, resolved_path in resolved_fold_paths.items():
            log(f"Using {field_name}: {resolved_path}", stage="initializing")


        if dataset not in {"serendipity", "taobao", "movielens"}:
            raise ValueError(
                f"Unsupported SCIR dataset slug: {dataset}"
            )


        api_request(
            run_id,
            "POST",
            {
                "status": "running",
                "progress": 5,
                "stage": "execution",
                "logs": logs,
            },
        )

        run_with_logging(
            run_id=str(run_id),
            n_users=run.get("n_users"),
            seed=int(run.get("seed", 42)),
            split_seed=int(run.get("split_seed", 42)),
            init_seed=int(run.get("init_seed", 42)),
            config=config,
            dataset=dataset,
        )

        result_json = results_dir / f"{run_id}.json"
        runs_log = run_dir / "runs.jsonl"

        if result_json.exists():
            with result_json.open("r", encoding="utf-8") as stream:
                json.load(stream)

        record = read_last_jsonl_record(runs_log)
        metrics = extract_metrics(record)

        config_path = run_dir / "config.json"
        config_path.write_text(
            json.dumps(
                config,
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        per_user = run_dir / f"{run_id}_per_user_metrics.csv"
        legacy_per_user = run_dir / f"{run_id}_per_user.csv"

        artifacts = []

        for path, kind in (
            (result_json, "results"),
            (per_user, "per_user"),
            (runs_log, "log"),
            (config_path, "config"),
        ):
            item = artifact_metadata(path, kind)

            if item:
                artifacts.append(item)

        if not per_user.exists() and legacy_per_user.exists():
            item = artifact_metadata(legacy_per_user, "per_user")

            if item:
                artifacts.append(item)

        log(
            f"SCIR finished. Collected {len(metrics)} metrics.",
            stage="completed",
        )

        api_request(
            run_id,
            "POST",
            {
                "status": "succeeded",
                "progress": 100,
                "stage": "completed",
                "worker_version": "0.1.0",
                "metrics": metrics,
                "artifacts": artifacts,
                "logs": logs,
            },
        )

        return {
            "run_id": str(run_id),
            "status": "succeeded",
            "metrics_count": len(metrics),
            "artifacts_count": len(artifacts),
        }

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

        log(error, level="error", stage="failed")
        log(
            traceback.format_exc(),
            level="error",
            stage="traceback",
        )

        try:
            api_request(
                run_id,
                "POST",
                {
                    "status": "failed",
                    "progress": 0,
                    "stage": "failed",
                    "error_code": "SCIR_EXECUTION_FAILED",
                    "error_message": error[:5000],
                    "logs": logs,
                },
            )
        except Exception:
            pass

        raise
