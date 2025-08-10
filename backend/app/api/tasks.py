from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional, List, Dict, Any

from app.database import get_db
from app.models.task_run import TaskRun
from app.workers.celery_app import celery_app
import os
import json

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None


router = APIRouter()


@router.get("/task-runs")
def list_task_runs(
    limit: int = Query(50, ge=1, le=200),
    status: Optional[str] = Query(None),
    task_name: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    q = db.query(TaskRun).order_by(TaskRun.id.desc())
    if status:
        q = q.filter(TaskRun.status == status)
    if task_name:
        q = q.filter(TaskRun.task_name == task_name)
    rows = q.limit(limit).all()
    return [
        {
            "id": r.id,
            "task_name": r.task_name,
            "task_id": r.task_id,
            "status": r.status,
            "started_at": r.started_at,
            "finished_at": r.finished_at,
            "duration_seconds": r.duration_seconds,
            "worker_hostname": r.worker_hostname,
        }
        for r in rows
    ]


def _redis_client():
    url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    if redis is None:
        return None
    try:
        return redis.from_url(url)
    except Exception:
        return None


@router.get("/dlq")
def list_dlq(limit: int = Query(50, ge=1, le=500)) -> List[Dict[str, Any]]:
    client = _redis_client()
    if not client:
        return []
    raw_items = client.lrange("celery:dlq", 0, limit - 1)
    items: List[Dict[str, Any]] = []
    for raw in raw_items:
        try:
            s = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
            items.append(json.loads(s))
        except Exception:
            items.append({"raw": str(raw)})
    return items


@router.post("/dlq/requeue")
def requeue_dlq(task_id: Optional[str] = None, index: Optional[int] = None) -> Dict[str, Any]:
    client = _redis_client()
    if not client:
        raise HTTPException(status_code=503, detail="Redis not available")

    # Find target payload
    target_raw = None
    payload = None

    if index is not None:
        raw = client.lindex("celery:dlq", index)
        if raw is None:
            raise HTTPException(status_code=404, detail="Index out of range")
        s = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
        try:
            payload = json.loads(s)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid DLQ payload at index")
        target_raw = s
    else:
        # scan first 500 for matching task_id
        raw_items = client.lrange("celery:dlq", 0, 499)
        for raw in raw_items:
            s = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
            try:
                p = json.loads(s)
                if p.get("task_id") == task_id:
                    payload = p
                    target_raw = s
                    break
            except Exception:
                continue
        if payload is None:
            raise HTTPException(status_code=404, detail="task_id not found in DLQ")

    # Requeue
    task_name = payload.get("task")
    args = payload.get("args") or []
    kwargs = payload.get("kwargs") or {}
    if not task_name:
        raise HTTPException(status_code=400, detail="DLQ payload missing task name")

    # Remove from DLQ by value
    client.lrem("celery:dlq", 1, target_raw)

    # Send task
    result = celery_app.send_task(task_name, args=args, kwargs=kwargs)
    return {"enqueued_task_id": result.id, "task": task_name}

