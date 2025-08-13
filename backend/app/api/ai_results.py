from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from app.database import get_db
from app.models.user import User
from app.services.auth import get_current_user

router = APIRouter(prefix="/api/ai-results", tags=["ai-results"])
logger = logging.getLogger(__name__)

@router.get("/")
async def get_ai_results(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get AI analysis results with pagination and filtering"""
    try:
        results = [
            {
                "id": 1,
                "image_name": "thermal_001.jpg",
                "substation": "Kalwa Substation",
                "tower_id": "KLW-T001",
                "analysis_date": "2025-01-13 14:30:00",
                "model_version": "YOLO-NAS-v3.5.0",
                "processing_time": 2.3,
                "detections": [
                    {
                        "component": "nuts_bolts",
                        "confidence": 0.94,
                        "temperature": 85.2,
                        "status": "normal",
                        "bbox": {"x": 120, "y": 80, "width": 45, "height": 35}
                    },
                    {
                        "component": "polymer_insulator",
                        "confidence": 0.87,
                        "temperature": 92.1,
                        "status": "warning",
                        "bbox": {"x": 200, "y": 150, "width": 60, "height": 80}
                    }
                ],
                "overall_status": "warning",
                "max_temperature": 92.1,
                "anomalies_count": 1,
                "health_score": 88
            },
            {
                "id": 2,
                "image_name": "thermal_002.jpg",
                "substation": "Mahape Substation",
                "tower_id": "MHP-T015",
                "analysis_date": "2025-01-13 14:25:00",
                "model_version": "YOLO-NAS-v3.5.0",
                "processing_time": 1.8,
                "detections": [
                    {
                        "component": "mid_span_joint",
                        "confidence": 0.96,
                        "temperature": 78.5,
                        "status": "normal",
                        "bbox": {"x": 180, "y": 120, "width": 55, "height": 40}
                    },
                    {
                        "component": "conductor",
                        "confidence": 0.91,
                        "temperature": 82.3,
                        "status": "normal",
                        "bbox": {"x": 50, "y": 200, "width": 300, "height": 25}
                    }
                ],
                "overall_status": "normal",
                "max_temperature": 82.3,
                "anomalies_count": 0,
                "health_score": 95
            },
            {
                "id": 3,
                "image_name": "thermal_003.jpg",
                "substation": "Taloja Substation",
                "tower_id": "TLJ-T008",
                "analysis_date": "2025-01-13 14:20:00",
                "model_version": "YOLO-NAS-v3.5.0",
                "processing_time": 3.1,
                "detections": [
                    {
                        "component": "nuts_bolts",
                        "confidence": 0.89,
                        "temperature": 105.7,
                        "status": "critical",
                        "bbox": {"x": 160, "y": 90, "width": 40, "height": 30}
                    },
                    {
                        "component": "polymer_insulator",
                        "confidence": 0.92,
                        "temperature": 98.4,
                        "status": "warning",
                        "bbox": {"x": 220, "y": 160, "width": 65, "height": 85}
                    }
                ],
                "overall_status": "critical",
                "max_temperature": 105.7,
                "anomalies_count": 2,
                "health_score": 72
            }
        ]
        
        if status:
            results = [r for r in results if r["overall_status"] == status]
        
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paginated_results = results[start_idx:end_idx]
        
        return {
            "results": paginated_results,
            "total": len(results),
            "page": page,
            "limit": limit,
            "total_pages": (len(results) + limit - 1) // limit
        }
        
    except Exception as e:
        logger.error(f"Error fetching AI results: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch AI results"
        )

@router.get("/{result_id}")
async def get_ai_result(
    result_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get specific AI analysis result"""
    try:
        result = {
            "id": result_id,
            "image_name": f"thermal_{result_id:03d}.jpg",
            "substation": "Kalwa Substation",
            "tower_id": f"KLW-T{result_id:03d}",
            "analysis_date": "2025-01-13 14:30:00",
            "model_version": "YOLO-NAS-v3.5.0",
            "processing_time": 2.3,
            "detections": [
                {
                    "component": "nuts_bolts",
                    "confidence": 0.94,
                    "temperature": 85.2,
                    "status": "normal",
                    "bbox": {"x": 120, "y": 80, "width": 45, "height": 35}
                }
            ],
            "overall_status": "normal",
            "max_temperature": 85.2,
            "anomalies_count": 0,
            "health_score": 95
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error fetching AI result {result_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch AI result"
        )
