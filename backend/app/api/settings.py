from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any

from app.database import get_db
from app.models.app_setting import AppSetting


router = APIRouter()


@router.get("/settings")
def get_settings(db: Session = Depends(get_db)) -> Dict[str, Any]:
    rows = db.query(AppSetting).all()
    return {row.key: row.value for row in rows}


@router.post("/settings")
def upsert_settings(payload: Dict[str, Any], db: Session = Depends(get_db)) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid payload")
    for key, value in payload.items():
        row = db.query(AppSetting).filter(AppSetting.key == key).first()
        if row:
            row.value = value
        else:
            db.add(AppSetting(key=key, value=value))
    db.commit()
    rows = db.query(AppSetting).all()
    return {row.key: row.value for row in rows}

