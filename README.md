# SCIR Worker

SCIR Worker is the background experiment execution service for **SCIR Lab**.

It provides the computational infrastructure required to execute recommendation-system experiments asynchronously, independently of the SCIR Lab web application.

The repository is part of the SCIR Lab platform architecture and is responsible for running experiment jobs submitted by the backend.

## Purpose

The main purpose of SCIR Worker is to execute computationally intensive recommendation-system experiments without blocking the web application.

It connects the SCIR Lab backend with the experiment execution environment and returns execution information to the platform.

## Main Responsibilities

- Receive experiment jobs from the Celery task queue.
- Execute recommendation-system experiments.
- Support different datasets and experiment configurations.
- Track experiment execution status and progress.
- Collect experiment logs and evaluation metrics.
- Report execution results to the SCIR Lab backend.
- Provide metadata about generated experiment artifacts.

## Architecture

SCIR Worker operates as an independent background service within the SCIR Lab architecture.

The execution workflow is:

1. A user configures and submits an experiment through SCIR Lab.
2. The Django backend creates an experiment record and places a task in the Redis-backed Celery queue.
3. SCIR Worker retrieves the task from the `scir` queue.
4. The worker executes the corresponding experiment.
5. Execution status, progress, logs, metrics, and artifact metadata are sent to the Django backend through its internal API.
6. SCIR Lab makes the experiment results available through its web interface.

## Technology Stack

- **Python** — experiment execution environment.
- **Celery** — asynchronous task processing.
- **Redis** — message broker and task queue.
- **Django REST API** — communication with the SCIR Lab backend.
- **PyTorch** — machine learning computations.
- **Sentence Transformers** — semantic representations.
- **FAISS** — vector-based retrieval.
- **scikit-learn, NumPy, pandas, SciPy** — data processing and evaluation.

## Repository Structure

- `scir_worker/` — Celery application and background task definitions.
- `src/` — recommendation, retrieval, ranking, evaluation, and experiment execution modules.
- `tools/` — supporting utilities for experiments and research analysis.
- `requirements.txt` — Python dependencies.

Datasets, generated experiment results, model weights, and local environment files are not included in the repository.

## Configuration

The worker requires the following environment variables:

| Variable | Description |
|---|---|
| `REDIS_URL` | Connection URL of the Redis instance used as the Celery broker. |
| `DJANGO_INTERNAL_API_URL` | Base URL of the SCIR Lab Django backend. |
| `WORKER_API_TOKEN` | Shared authentication token used for internal worker-to-backend requests. |

These variables must be configured in the deployment environment.

Secrets and credentials must never be committed to the repository.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Worker

Start the Celery worker with:

```bash
celery -A scir_worker.celery_app worker --loglevel=INFO -Q scir
```

The worker listens to the `scir` queue and processes experiment execution tasks submitted by SCIR Lab.

## Deployment

SCIR Worker is designed to run as a separate Railway service.

It must share access to the Redis instance configured for the SCIR Lab backend and have the required internal API environment variables.

The worker does not expose a public web interface. Experiment configuration, user interaction, and result visualization are handled by SCIR Lab.

## Scope

This repository focuses on background experiment execution and integration with the SCIR Lab platform.

It is not a standalone recommendation service or a replacement for the SCIR Lab web application.

Persistent artifact storage and artifact downloads are outside the scope of the initial public demonstration.

## Acknowledgements

The experiment codebase builds upon and adapts work from the [`cold-start-algorithm`](https://github.com/nikita-zmanovskiy/cold-start-algorithm) repository.

The SCIR Worker repository has a separate purpose: providing the background execution infrastructure and integration layer for SCIR Lab.

