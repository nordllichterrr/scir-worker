import os
from celery import Celery

app = Celery(
    "scir_worker",
    broker=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
    include=["scir_worker.tasks"],
)
app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_time_limit=43200,
    task_soft_time_limit=39600,
    timezone="UTC",
    enable_utc=True,
)
