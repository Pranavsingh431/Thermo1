import os
from celery import Celery
from celery import signals
import json
from datetime import datetime

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None


def _broker() -> str:
    return os.environ.get("CELERY_BROKER_URL", os.environ.get("REDIS_URL", "redis://localhost:6379/0"))


def _backend() -> str:
    return os.environ.get("CELERY_RESULT_BACKEND", os.environ.get("REDIS_URL", "redis://localhost:6379/0"))


celery_app = Celery(
    "thermal_inspection",
    broker=_broker(),
    backend=_backend(),
    include=[
        "app.workers.tasks",
    ],
)

# Sensible defaults
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    worker_max_tasks_per_child=100,
)

# Beat schedule for maintenance tasks
celery_app.conf.beat_schedule = {
    "mark-stale-tasks": {
        "task": "maintenance.mark_stale_tasks",
        "schedule": 300.0,  # every 5 minutes
    }
}

# Ensure tasks are discovered when worker starts
try:
    celery_app.autodiscover_tasks(packages=["app.workers"])  # looks for tasks.py
except Exception:
    pass


# Simple DLQ using Redis list for failures
def _redis_client():
    url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    if redis is None:
        return None
    try:
        return redis.from_url(url)
    except Exception:
        return None


@signals.task_failure.connect
def _on_task_failure(sender=None, task_id=None, exception=None, args=None, kwargs=None, traceback=None, einfo=None, **kw):
    client = _redis_client()
    if not client:
        return
    try:
        payload = {
            "event": "task_failure",
            "task": getattr(sender, "name", str(sender)),
            "task_id": task_id,
            "args": args,
            "kwargs": kwargs,
            "error": str(exception) if exception else None,
            "when": datetime.utcnow().isoformat() + "Z",
        }
        client.lpush("celery:dlq", json.dumps(payload))
        # keep DLQ bounded
        client.ltrim("celery:dlq", 0, 999)
    except Exception:
        pass

