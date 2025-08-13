from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from app.database import get_db
from app.models.user import User
from app.utils.auth import get_current_user

router = APIRouter(prefix="/api/thermal-scans", tags=["thermal-scans"])
logger = logging.getLogger(__name__)

@router.get("/")
async def get_thermal_scans(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    substation_id: Optional[int] = Query(None),
    scan_status: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get thermal scans with pagination and filtering"""
    try:
        scans = [
            {
                "id": 1,
                "original_filename": "thermal_001.jpg",
                "substation_name": "Kalwa Substation",
                "tower_id": "KLW-T001",
                "equipment_type": "Transmission Line",
                "ambient_temperature": 35.2,
                "weather_conditions": "Clear",
                "processing_status": "completed",
                "created_at": "2025-01-13T14:30:00",
                "analysis_summary": {
                    "max_temperature": 92.1,
                    "anomalies_found": 1,
                    "overall_status": "warning",
                    "health_score": 88
                }
            },
            {
                "id": 2,
                "original_filename": "thermal_002.jpg",
                "substation_name": "Mahape Substation",
                "tower_id": "MHP-T015",
                "equipment_type": "Transmission Line",
                "ambient_temperature": 33.8,
                "weather_conditions": "Partly Cloudy",
                "processing_status": "completed",
                "created_at": "2025-01-13T14:25:00",
                "analysis_summary": {
                    "max_temperature": 82.3,
                    "anomalies_found": 0,
                    "overall_status": "normal",
                    "health_score": 95
                }
            },
            {
                "id": 3,
                "original_filename": "thermal_003.jpg",
                "substation_name": "Taloja Substation",
                "tower_id": "TLJ-T008",
                "equipment_type": "Transmission Line",
                "ambient_temperature": 36.1,
                "weather_conditions": "Hot",
                "processing_status": "completed",
                "created_at": "2025-01-13T14:20:00",
                "analysis_summary": {
                    "max_temperature": 105.7,
                    "anomalies_found": 2,
                    "overall_status": "critical",
                    "health_score": 72
                }
            }
        ]
        
        if substation_id:
            scans = [s for s in scans if s.get("substation_id") == substation_id]
        
        if scan_status:
            scans = [s for s in scans if s["processing_status"] == scan_status]
        
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paginated_scans = scans[start_idx:end_idx]
        
        return {
            "scans": paginated_scans,
            "total": len(scans),
            "page": page,
            "limit": limit,
            "total_pages": (len(scans) + limit - 1) // limit
        }
        
    except Exception as e:
        logger.error(f"Error fetching thermal scans: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch thermal scans"
        )

@router.get("/{scan_id}")
async def get_thermal_scan(
    scan_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get specific thermal scan details"""
    try:
        scan = {
            "id": scan_id,
            "original_filename": f"thermal_{scan_id:03d}.jpg",
            "substation_name": "Kalwa Substation",
            "tower_id": f"KLW-T{scan_id:03d}",
            "equipment_type": "Transmission Line",
            "ambient_temperature": 35.2,
            "weather_conditions": "Clear",
            "processing_status": "completed",
            "created_at": "2025-01-13T14:30:00",
            "file_path": f"/static/thermal_images/thermal_{scan_id:03d}.jpg",
            "processed_path": f"/static/processed_images/thermal_{scan_id:03d}_processed.jpg",
            "notes": "Routine thermal inspection",
            "analysis_summary": {
                "max_temperature": 92.1,
                "min_temperature": 25.3,
                "avg_temperature": 45.7,
                "anomalies_found": 1,
                "overall_status": "warning",
                "health_score": 88,
                "components_detected": 5,
                "processing_time": 2.3
            }
        }
        
        return scan
        
    except Exception as e:
        logger.error(f"Error fetching thermal scan {scan_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch thermal scan"
        )
