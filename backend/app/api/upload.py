"""
Thermal image upload API routes
"""

import asyncio
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
import json
from sqlalchemy.orm import Session
from typing import List, Optional, AsyncGenerator
from pydantic import BaseModel
import logging
from datetime import datetime
import traceback
from pathlib import Path

from app.database import get_db, SessionLocal
from app.models.user import User
from app.models.substation import Substation
from app.models.thermal_scan import ThermalScan
from app.utils.auth import get_current_user, require_permission
from app.utils.auth import verify_token
from app.utils.thermal_processing import thermal_processor
from app.services.thermal_analysis import process_thermal_batch
from app.utils.storage import storage_client
from app.config import settings
from app.workers.tasks import process_batch

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models
class FileUploadStatus(BaseModel):
    filename: str
    status: str
    message: str
    thermal_scan_id: Optional[int] = None

class UploadResponse(BaseModel):
    batch_id: str
    total_files: int
    successful_uploads: int
    failed_uploads: int
    processing_status: str
    message: str
    details: List[FileUploadStatus] = []

class BatchStatus(BaseModel):
    batch_id: str
    total_images: int
    processed_images: int
    failed_images: int
    status: str
    created_at: str
    processing_duration: Optional[str] = None

class PresignRequest(BaseModel):
    batch_id: str
    filename: str
    content_type: Optional[str] = "image/jpeg"

class PresignResponse(BaseModel):
    provider: str
    url: str
    headers: dict
    key: str
class ConfirmFile(BaseModel):
    filename: str
    key: str
    content_type: Optional[str] = None
    size_bytes: Optional[int] = None

class ConfirmRequest(BaseModel):
    batch_id: str
    files: List[ConfirmFile]
    ambient_temperature: Optional[float] = 34.0
    notes: Optional[str] = None


@router.post("/thermal-images", response_model=UploadResponse)
async def upload_thermal_images(
    files: List[UploadFile] = File(...),
    ambient_temperature: Optional[float] = Form(34.0),
    notes: Optional[str] = Form(None),
    current_user: User = Depends(require_permission("upload")),
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = None
):
    """Upload multiple thermal images for processing"""
    
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided"
        )
    
    # Validate file count
    if len(files) > 5000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 5000 files allowed per batch"
        )
    
    # Create batch ID
    batch_id = thermal_processor.create_batch_id()
    successful_uploads = 0
    failed_uploads = 0
    upload_results = []
    
    # Get all substations for location matching
    substations = db.query(Substation).filter(Substation.is_active == True).all()
    
    logger.info(f"Starting batch upload {batch_id} with {len(files)} files by user {current_user.username}")
    
    for sequence, file in enumerate(files, 1):
        try:
            # Read file content
            file_content = await file.read()
            file_size = len(file_content)
            
            # Validate file
            is_valid, validation_message = thermal_processor.validate_image_file(file.filename, file_size)
            if not is_valid:
                failed_uploads += 1
                upload_results.append(FileUploadStatus(
                    filename=file.filename,
                    status="failed",
                    message=validation_message
                ))
                continue
            
            # Calculate file hash for deduplication
            file_hash = thermal_processor.calculate_file_hash(file_content)
            
            # Check for duplicate
            existing_scan = db.query(ThermalScan).filter(ThermalScan.file_hash == file_hash).first()
            if existing_scan:
                logger.warning(f"Duplicate file detected: {file.filename} (hash: {file_hash})")
                upload_results.append(FileUploadStatus(
                    filename=file.filename,
                    status="skipped",
                    message="Duplicate file already processed",
                    thermal_scan_id=existing_scan.id
                ))
                continue
            
            # Save file temporarily
            file_path = thermal_processor.save_uploaded_file(file_content, file.filename, batch_id)
            # Optionally upload to object storage
            storage_uri = None
            if settings.USE_OBJECT_STORAGE and settings.OBJECT_STORAGE_PROVIDER.lower() != 'none':
                remote_key = f"uploads/{batch_id}/{Path(file_path).name}"
                try:
                    storage_uri = storage_client.upload_file(file_path, remote_key)
                except Exception:
                    storage_uri = None
            
            # Extract metadata
            metadata = thermal_processor.extract_image_metadata(file_path)
            
            # Find matching substation
            substation_id = None
            if metadata.get('latitude') and metadata.get('longitude'):
                substation_id = thermal_processor.find_matching_substation(
                    metadata['latitude'], 
                    metadata['longitude'], 
                    substations
                )
            
            # Create thermal scan record
            thermal_scan = ThermalScan(
                original_filename=file.filename,
                file_path=storage_uri or file_path,
                file_size_bytes=file_size,
                file_hash=file_hash,
                camera_model=metadata.get('camera_model'),
                camera_software_version=metadata.get('camera_software_version'),
                image_width=metadata.get('image_width'),
                image_height=metadata.get('image_height'),
                latitude=metadata.get('latitude'),
                longitude=metadata.get('longitude'),
                altitude=metadata.get('altitude'),
                gps_timestamp=metadata.get('gps_timestamp'),
                capture_timestamp=metadata.get('capture_timestamp') or datetime.now(),
                ambient_temperature=ambient_temperature,
                camera_settings=metadata.get('camera_settings'),
                batch_id=batch_id,
                batch_sequence=sequence,
                substation_id=substation_id,
                uploaded_by=current_user.id,
                notes=notes
            )
            
            db.add(thermal_scan)
            db.commit()
            db.refresh(thermal_scan)
            
            successful_uploads += 1
            upload_results.append(FileUploadStatus(
                filename=file.filename,
                status="uploaded",
                message="Successfully uploaded and queued for processing",
                thermal_scan_id=thermal_scan.id
            ))
            
        except Exception as e:
            # Log full traceback for debugging
            logger.error(f"Failed to process file {file.filename}: {e}", exc_info=True)
            try:
                Path('logs').mkdir(exist_ok=True)
                with open('logs/upload_debug.log', 'a') as dbg:
                    dbg.write(f"[{datetime.utcnow().isoformat()}] File: {file.filename} Error: {e}\n")
                    dbg.write(traceback.format_exc())
                    dbg.write("\n")
            except Exception:
                pass
            failed_uploads += 1
            upload_results.append(FileUploadStatus(
                filename=file.filename,
                status="failed",
                message=f"Processing error: {str(e)}"
            ))
    
    # Start background processing if any files were uploaded successfully
    processing_status = "idle"
    if successful_uploads > 0:
        # Enqueue Celery batch task (scales beyond a single process)
        try:
            process_batch.delay(batch_id)
        except Exception:
            # Fallback to in-process task with its own DB session
            async def _run_batch():
                db2: Session = SessionLocal()
                try:
                    await process_thermal_batch(batch_id, db2, current_user.id)
                finally:
                    db2.close()
            asyncio.create_task(_run_batch())
        processing_status = "queued"
        
        message = f"Batch {batch_id}: {successful_uploads} files uploaded successfully, {failed_uploads} failed. Processing started."
    else:
        message = f"Batch {batch_id}: No files processed successfully. {failed_uploads} failed."
    
    logger.info(f"Batch upload {batch_id} completed: {successful_uploads} success, {failed_uploads} failed")
    
    return UploadResponse(
        batch_id=batch_id,
        total_files=len(files),
        successful_uploads=successful_uploads,
        failed_uploads=failed_uploads,
        processing_status=processing_status,
        message=message,
        details=upload_results
    )

@router.post("/thermal-images/presign", response_model=PresignResponse)
async def presign_upload(
    req: PresignRequest,
    current_user: User = Depends(require_permission("upload"))
):
    if not settings.USE_OBJECT_STORAGE or settings.OBJECT_STORAGE_PROVIDER.lower() == 'none':
        raise HTTPException(status_code=400, detail="Object storage not enabled")
    key = f"uploads/{req.batch_id}/{req.filename}"
    data = storage_client.presign_upload(key, req.content_type or "application/octet-stream")
    if not data:
        raise HTTPException(status_code=500, detail="Failed to generate presigned upload")
    return PresignResponse(provider=data["provider"], url=data["url"], headers=data["headers"], key=data["key"])

@router.post("/thermal-images/confirm", response_model=UploadResponse)
async def confirm_presigned_upload(
    req: ConfirmRequest,
    current_user: User = Depends(require_permission("upload")),
    db: Session = Depends(get_db)
):
    if not req.files:
        raise HTTPException(status_code=400, detail="No files provided")
    if len(req.files) > 5000:
        raise HTTPException(status_code=400, detail="Maximum 5000 files allowed per batch")
    substations = db.query(Substation).filter(Substation.is_active == True).all()
    batch_id = req.batch_id
    successful_uploads = 0
    failed_uploads = 0
    upload_results: List[FileUploadStatus] = []
    provider = settings.OBJECT_STORAGE_PROVIDER.lower()
    bucket = settings.OBJECT_STORAGE_BUCKET
    for sequence, f in enumerate(req.files, 1):
        try:
            if not f.filename or not f.key:
                failed_uploads += 1
                upload_results.append(FileUploadStatus(filename=f.filename or "", status="failed", message="Missing filename or key"))
                continue
            storage_uri = None
            if settings.USE_OBJECT_STORAGE and provider != "none":
                if provider == "s3":
                    storage_uri = f"s3://{bucket}/{f.key}"
                elif provider == "gcs":
                    storage_uri = f"gs://{bucket}/{f.key}"
            metadata = {}
            substation_id = None
            thermal_scan = ThermalScan(
                original_filename=f.filename,
                file_path=storage_uri or f.key,
                file_size_bytes=f.size_bytes,
                file_hash=None,
                camera_model=metadata.get("camera_model") if metadata else None,
                camera_software_version=metadata.get("camera_software_version") if metadata else None,
                image_width=metadata.get("image_width") if metadata else None,
                image_height=metadata.get("image_height") if metadata else None,
                latitude=metadata.get("latitude") if metadata else None,
                longitude=metadata.get("longitude") if metadata else None,
                altitude=metadata.get("altitude") if metadata else None,
                gps_timestamp=metadata.get("gps_timestamp") if metadata else None,
                capture_timestamp=datetime.now(),
                ambient_temperature=req.ambient_temperature,
                camera_settings=metadata.get("camera_settings") if metadata else None,
                batch_id=batch_id,
                batch_sequence=sequence,
                substation_id=substation_id,
                uploaded_by=current_user.id,
                notes=req.notes
            )
            db.add(thermal_scan)
            db.commit()
            db.refresh(thermal_scan)
            successful_uploads += 1
            upload_results.append(FileUploadStatus(filename=f.filename, status="uploaded", message="Queued for processing", thermal_scan_id=thermal_scan.id))
        except Exception as e:
            logger.error(f"Failed to confirm file {f.filename}: {e}", exc_info=True)
            failed_uploads += 1
            upload_results.append(FileUploadStatus(filename=f.filename, status="failed", message=f"Confirm error: {str(e)}"))
    processing_status = "idle"
    if successful_uploads > 0:
        try:
            process_batch.delay(batch_id)
        except Exception:
            async def _run_batch():
                db2: Session = SessionLocal()
                try:
                    await process_thermal_batch(batch_id, db2, current_user.id)
                finally:
                    db2.close()
            asyncio.create_task(_run_batch())
        processing_status = "queued"
    return UploadResponse(
        batch_id=batch_id,
        total_files=len(req.files),
        successful_uploads=successful_uploads,
        failed_uploads=failed_uploads,
        processing_status=processing_status,
        message=f"Batch {batch_id}: {successful_uploads} confirmed, {failed_uploads} failed.",
        details=upload_results
    )

@router.get("/batch/{batch_id}/status", response_model=BatchStatus)
async def get_batch_status(
    batch_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get processing status of a batch"""
    
    # Get all thermal scans in this batch
    scans = db.query(ThermalScan).filter(ThermalScan.batch_id == batch_id).all()
    
    if not scans:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Batch not found"
        )
    
    # Check if user has access to this batch
    if not current_user.can_view_all_data:
        # Users can only see their own batches
        if not any(scan.uploaded_by == current_user.id for scan in scans):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this batch"
            )
    
    # Calculate status
    total_images = len(scans)
    processed_images = len([s for s in scans if s.processing_status == "completed"])
    failed_images = len([s for s in scans if s.processing_status == "failed"])
    pending_images = len([s for s in scans if s.processing_status == "pending"])
    processing_images = len([s for s in scans if s.processing_status == "processing"])
    
    # Determine overall status
    if failed_images == total_images:
        overall_status = "failed"
    elif processed_images == total_images:
        overall_status = "completed"
    elif processing_images > 0:
        overall_status = "processing"
    elif pending_images > 0:
        overall_status = "pending"
    else:
        overall_status = "unknown"
    
    # Calculate processing duration (timezone-robust)
    processing_duration = None
    if scans:
        from datetime import timezone
        first_scan = min(scans, key=lambda x: x.created_at)
        completed_scans = [s for s in scans if s.processing_completed_at]
        if completed_scans:
            last_completed = max(completed_scans, key=lambda x: x.processing_completed_at)
            start = first_scan.created_at
            end = last_completed.processing_completed_at
            try:
                # Normalize both to UTC-aware datetimes
                if getattr(start, 'tzinfo', None) is None:
                    start_utc = start.replace(tzinfo=timezone.utc)
                else:
                    start_utc = start.astimezone(timezone.utc)
                if getattr(end, 'tzinfo', None) is None:
                    end_utc = end.replace(tzinfo=timezone.utc)
                else:
                    end_utc = end.astimezone(timezone.utc)
                duration = (end_utc - start_utc).total_seconds()
                processing_duration = f"{duration:.1f} seconds"
            except Exception:
                processing_duration = None
    
    return BatchStatus(
        batch_id=batch_id,
        total_images=total_images,
        processed_images=processed_images,
        failed_images=failed_images,
        status=overall_status,
        created_at=scans[0].created_at.isoformat() if scans else "",
        processing_duration=processing_duration
    )

@router.get("/batch/{batch_id}/files", response_model=List[FileUploadStatus])
async def get_batch_files(
    batch_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get detailed status of all files in a batch"""
    
    scans = db.query(ThermalScan).filter(ThermalScan.batch_id == batch_id).all()
    
    if not scans:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Batch not found"
        )
    
    # Check access permissions
    if not current_user.can_view_all_data:
        if not any(scan.uploaded_by == current_user.id for scan in scans):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this batch"
            )
    
    return [
        FileUploadStatus(
            filename=scan.original_filename,
            status=scan.processing_status,
            message=f"Processing time: {scan.processing_time_str}",
            thermal_scan_id=scan.id
        )
        for scan in sorted(scans, key=lambda x: x.batch_sequence)
    ]

@router.get("/batch/{batch_id}/stream")
async def stream_batch_progress(batch_id: str, request: Request, db: Session = Depends(get_db)):
    """Server-Sent Events (SSE) stream for live batch progress"""
    # Authenticate using Authorization header or token query param (EventSource cannot set headers reliably)
    username = None
    auth_header = request.headers.get('Authorization')
    if auth_header and auth_header.lower().startswith('bearer '):
        username = verify_token(auth_header.split(' ', 1)[1])
    if not username:
        token = request.query_params.get('token')
        if token:
            username = verify_token(token)
    if not username:
        raise HTTPException(status_code=403, detail="Unauthorized")
    current_user = db.query(User).filter(User.username == username).first()
    if not current_user or not current_user.is_active:
        raise HTTPException(status_code=403, detail="Unauthorized")

    # Access control
    scans = db.query(ThermalScan).filter(ThermalScan.batch_id == batch_id).all()
    if not scans:
        raise HTTPException(status_code=404, detail="Batch not found")
    if not current_user.can_view_all_data and not any(s.uploaded_by == current_user.id for s in scans):
        raise HTTPException(status_code=403, detail="Access denied to this batch")

    async def event_generator() -> AsyncGenerator[bytes, None]:
        import asyncio
        last_counts = (-1, -1, -1)
        while True:
            if await request.is_disconnected():
                break
            total = len(scans)
            done = len([s for s in scans if s.processing_status == "completed"])
            failed = len([s for s in scans if s.processing_status == "failed"])
            state = (total, done, failed)
            if state != last_counts:
                payload = {
                    "batch_id": batch_id,
                    "total": total,
                    "completed": done,
                    "failed": failed,
                }
                yield f"data: {json.dumps(payload)}\n\n".encode("utf-8")
                last_counts = state
            # refresh from DB periodically
            await asyncio.sleep(2)
            db.expire_all()
            scans[:] = db.query(ThermalScan).filter(ThermalScan.batch_id == batch_id).all()

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.delete("/batch/{batch_id}")
async def delete_batch(
    batch_id: str,
    current_user: User = Depends(require_permission("admin")),
    db: Session = Depends(get_db)
):
    """Delete a batch and all associated data (admin only)"""
    
    scans = db.query(ThermalScan).filter(ThermalScan.batch_id == batch_id).all()
    
    if not scans:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Batch not found"
        )
    
    # Delete all thermal scans (cascade will handle AI analyses)
    for scan in scans:
        db.delete(scan)
    
    # Clean up files
    thermal_processor.cleanup_batch_files(batch_id)
    
    db.commit()
    
    logger.info(f"Batch {batch_id} deleted by admin {current_user.username}")
    
    return {"message": f"Batch {batch_id} deleted successfully"}

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "upload"}                                                