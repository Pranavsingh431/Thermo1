from typing import Optional
from sqlalchemy.orm import Session
from app.workers.celery_app import celery_app
from app.database import SessionLocal
from app.models.thermal_scan import ThermalScan
from app.services.thermal_analysis import process_single_thermal_image_full_ai
from app.models.task_run import TaskRun
import asyncio
import logging
import os
from datetime import datetime, timedelta

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None

logger = logging.getLogger(__name__)


def _redis_client():
    url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    if redis is None:
        return None
    try:
        return redis.from_url(url)
    except Exception:
        return None


@celery_app.task(
    name="process_single_image",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=300,  # max 5min between retries
    retry_jitter=True,
    retry_kwargs={"max_retries": 5},
    acks_late=True,
    time_limit=900,  # hard 15min
    soft_time_limit=600,  # soft 10min
)
def process_single_image(self, scan_id: int) -> Optional[int]:
    db: Session = SessionLocal()
    client = _redis_client()
    lock_key = f"locks:scan:{scan_id}"
    have_lock = False
    try:
        # create task_run row
        task_run = TaskRun(
            task_name="process_single_image",
            task_id=self.request.id if hasattr(self, "request") else None,
            args=[scan_id],
            kwargs={},
            status="started",
            worker_hostname=getattr(self.request, "hostname", None) if hasattr(self, "request") else None,
        )
        db.add(task_run)
        db.commit()
        db.refresh(task_run)

        # idempotency lock (15 min TTL)
        if client is not None:
            try:
                have_lock = bool(client.set(lock_key, task_run.task_id or "1", nx=True, ex=900))
            except Exception:
                have_lock = False
        if client is not None and not have_lock:
            logger.info("process_single_image: duplicate in-flight, skipping id=%s", scan_id)
            task_run.status = "skipped_duplicate"
            db.commit()
            return None

        scan = db.query(ThermalScan).filter(ThermalScan.id == scan_id).first()
        if not scan:
            logger.warning("process_single_image: scan not found id=%s", scan_id)
            task_run.status = "not_found"
            db.commit()
            return None
        logger.info("process_single_image: start id=%s", scan_id)
        # Await the async analysis function inside Celery task
        asyncio.run(process_single_thermal_image_full_ai(scan, db))
        logger.info("process_single_image: done id=%s", scan_id)
        # mark success
        task_run.status = "succeeded"
        task_run.finished_at = datetime.utcnow()
        if task_run.started_at:
            task_run.duration_seconds = (task_run.finished_at - task_run.started_at).total_seconds()
        db.commit()
        return scan.id
    except Exception as exc:
        logger.exception("process_single_image: error id=%s", scan_id)
        db.rollback()
        try:
            # update error status
            task_run = db.query(TaskRun).filter(TaskRun.task_id == (self.request.id if hasattr(self, "request") else None)).first()
            if task_run:
                task_run.status = "failed"
                task_run.error_message = str(exc)
                task_run.finished_at = datetime.utcnow()
                if task_run.started_at:
                    task_run.duration_seconds = (task_run.finished_at - task_run.started_at).total_seconds()
                db.commit()
        except Exception:
            db.rollback()
        raise exc
    finally:
        if client is not None and have_lock:
            try:
                client.delete(lock_key)
            except Exception:
                pass
        db.close()


@celery_app.task(
    name="process_batch",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True,
    retry_kwargs={"max_retries": 3},
    acks_late=True,
    time_limit=1800,
    soft_time_limit=1200,
)
def process_batch(self, batch_id: str) -> int:
    db: Session = SessionLocal()
    try:
        # create task_run row
        task_run = TaskRun(
            task_name="process_batch",
            task_id=self.request.id if hasattr(self, "request") else None,
            args=[batch_id],
            kwargs={},
            status="started",
            worker_hostname=getattr(self.request, "hostname", None) if hasattr(self, "request") else None,
        )
        db.add(task_run)
        db.commit()

        logger.info("process_batch: enqueue scans for batch_id=%s", batch_id)
        scans = db.query(ThermalScan).filter(ThermalScan.batch_id == batch_id).all()
        count = 0
        for s in scans:
            process_single_image.delay(s.id)
            count += 1
        logger.info("process_batch: enqueued=%s batch_id=%s", count, batch_id)
        from datetime import datetime
        task_run.status = "succeeded"
        task_run.finished_at = datetime.utcnow()
        if task_run.started_at:
            task_run.duration_seconds = (task_run.finished_at - task_run.started_at).total_seconds()
        db.commit()
        return count
    finally:
        db.close()


@celery_app.task(name="maintenance.mark_stale_tasks")
def mark_stale_tasks() -> int:
    """Mark long-running TaskRun rows as stale if exceeded threshold."""
    db: Session = SessionLocal()
    try:
        threshold = datetime.utcnow() - timedelta(minutes=30)
        q = db.query(TaskRun).filter(TaskRun.status == "started", TaskRun.started_at < threshold)
        stale = 0
        for row in q.all():
            row.status = "stale"
            stale += 1
        if stale:
            db.commit()
        return stale
    finally:
        db.close()

