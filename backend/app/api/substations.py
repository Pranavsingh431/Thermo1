from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from app.database import get_db
from app.models.user import User
from app.services.auth import get_current_user

router = APIRouter(prefix="/api/substations", tags=["substations"])
logger = logging.getLogger(__name__)

@router.get("/")
async def get_substations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all substations"""
    try:
        substations = [
            {
                "id": 1,
                "name": "Kalwa Substation",
                "code": "KLW",
                "location": "Kalwa, Thane",
                "latitude": 19.2183,
                "longitude": 72.9781,
                "voltage_level": "220kV",
                "status": "active",
                "towers_count": 45,
                "last_inspection": "2025-01-10",
                "health_score": 95
            },
            {
                "id": 2,
                "name": "Mahape Substation",
                "code": "MHP",
                "location": "Mahape, Navi Mumbai",
                "latitude": 19.1136,
                "longitude": 73.0169,
                "voltage_level": "110kV",
                "status": "active",
                "towers_count": 32,
                "last_inspection": "2025-01-08",
                "health_score": 88
            },
            {
                "id": 3,
                "name": "Taloja Substation",
                "code": "TLJ",
                "location": "Taloja, Navi Mumbai",
                "latitude": 19.0176,
                "longitude": 73.0961,
                "voltage_level": "220kV",
                "status": "maintenance",
                "towers_count": 28,
                "last_inspection": "2025-01-05",
                "health_score": 76
            }
        ]
        
        return substations
        
    except Exception as e:
        logger.error(f"Error fetching substations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch substations"
        )

@router.get("/{substation_id}")
async def get_substation(
    substation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get specific substation details"""
    try:
        substations = {
            1: {
                "id": 1,
                "name": "Kalwa Substation",
                "code": "KLW",
                "location": "Kalwa, Thane",
                "latitude": 19.2183,
                "longitude": 72.9781,
                "voltage_level": "220kV",
                "status": "active",
                "towers_count": 45,
                "last_inspection": "2025-01-10",
                "health_score": 95
            },
            2: {
                "id": 2,
                "name": "Mahape Substation",
                "code": "MHP",
                "location": "Mahape, Navi Mumbai",
                "latitude": 19.1136,
                "longitude": 73.0169,
                "voltage_level": "110kV",
                "status": "active",
                "towers_count": 32,
                "last_inspection": "2025-01-08",
                "health_score": 88
            },
            3: {
                "id": 3,
                "name": "Taloja Substation",
                "code": "TLJ",
                "location": "Taloja, Navi Mumbai",
                "latitude": 19.0176,
                "longitude": 73.0961,
                "voltage_level": "220kV",
                "status": "maintenance",
                "towers_count": 28,
                "last_inspection": "2025-01-05",
                "health_score": 76
            }
        }
        
        if substation_id not in substations:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Substation not found"
            )
        
        return substations[substation_id]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching substation {substation_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch substation"
        )

@router.post("/")
async def create_substation(
    substation_data: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create new substation"""
    try:
        new_substation = {
            "id": 999,
            "name": substation_data.get("name"),
            "code": substation_data.get("code"),
            "location": substation_data.get("location"),
            "latitude": substation_data.get("latitude"),
            "longitude": substation_data.get("longitude"),
            "voltage_level": substation_data.get("voltage_level"),
            "status": "active",
            "towers_count": 0,
            "last_inspection": None,
            "health_score": 100,
            "created_at": datetime.utcnow().isoformat()
        }
        
        return {
            "message": "Substation created successfully",
            "substation": new_substation
        }
        
    except Exception as e:
        logger.error(f"Error creating substation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create substation"
        )

@router.put("/{substation_id}")
async def update_substation(
    substation_id: int,
    substation_data: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update existing substation"""
    try:
        return {
            "message": "Substation updated successfully",
            "substation_id": substation_id,
            "updated_data": substation_data
        }
        
    except Exception as e:
        logger.error(f"Error updating substation {substation_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update substation"
        )

@router.delete("/{substation_id}")
async def delete_substation(
    substation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete substation"""
    try:
        return {
            "message": "Substation deleted successfully",
            "substation_id": substation_id
        }
        
    except Exception as e:
        logger.error(f"Error deleting substation {substation_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete substation"
        )
