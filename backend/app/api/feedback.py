from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Dict, Any, List
import logging

from app.database import get_db
from app.models.user import User
from app.services.auth import get_current_user
from app.services.model_improvement import model_improvement_service

router = APIRouter(prefix="/api/feedback", tags=["feedback"])
logger = logging.getLogger(__name__)

@router.post("/analysis/{analysis_id}")
async def submit_analysis_feedback(
    analysis_id: str,
    feedback_data: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Submit user feedback on AI analysis results for model improvement"""
    try:
        success = model_improvement_service.collect_feedback(analysis_id, feedback_data)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analysis not found or feedback submission failed"
            )
        
        return {
            "message": "Feedback submitted successfully",
            "analysis_id": analysis_id,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit feedback"
        )

@router.get("/model-performance")
async def get_model_performance(
    current_user: User = Depends(get_current_user)
):
    """Get current model performance metrics"""
    try:
        metrics = model_improvement_service.get_model_performance_metrics()
        return {
            "performance_metrics": metrics,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve performance metrics"
        )

@router.post("/retrain-models")
async def trigger_model_retraining(
    current_user: User = Depends(get_current_user)
):
    """Trigger model retraining based on accumulated feedback"""
    try:
        if current_user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only administrators can trigger model retraining"
            )
        
        results = model_improvement_service.retrain_models()
        return {
            "retraining_results": results,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error during model retraining: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrain models"
        )

@router.get("/improvement-report")
async def get_improvement_report(
    current_user: User = Depends(get_current_user)
):
    """Generate comprehensive model improvement report"""
    try:
        report = model_improvement_service.generate_improvement_report()
        return {
            "improvement_report": report,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error generating improvement report: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate improvement report"
        )
