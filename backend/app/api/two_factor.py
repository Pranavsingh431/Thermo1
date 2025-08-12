"""
Two-Factor Authentication API routes
"""

from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from datetime import timedelta
import pyotp
import qrcode
import io
import base64
import json
import secrets

from app.database import get_db
from app.models.user import User
from app.utils.auth import get_current_user, authenticate_user, create_access_token, create_refresh_token_pair
from app.config import settings

router = APIRouter()

class TwoFactorSetupResponse(BaseModel):
    secret: str
    qr_code: str
    backup_codes: list[str]

class TwoFactorVerifyRequest(BaseModel):
    token: str

class TwoFactorEnableRequest(BaseModel):
    token: str

class BackupCodeVerifyRequest(BaseModel):
    backup_code: str

class TwoFactorLoginRequest(BaseModel):
    username: str
    password: str
    token: str

@router.post("/setup", response_model=TwoFactorSetupResponse)
async def setup_two_factor(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Generate 2FA secret and QR code for user setup"""
    if current_user.two_factor_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Two-factor authentication is already enabled"
        )
    
    secret = pyotp.random_base32()
    
    backup_codes = [secrets.token_hex(4).upper() for _ in range(8)]
    
    totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
        name=current_user.email,
        issuer_name="Thermal Inspection System"
    )
    
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(totp_uri)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    qr_code_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    qr_code_data_uri = f"data:image/png;base64,{qr_code_base64}"
    
    current_user.two_factor_secret = secret
    current_user.backup_codes = json.dumps(backup_codes)
    db.commit()
    
    return TwoFactorSetupResponse(
        secret=secret,
        qr_code=qr_code_data_uri,
        backup_codes=backup_codes
    )

@router.post("/enable")
async def enable_two_factor(
    data: TwoFactorEnableRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Enable 2FA after verifying the setup token"""
    if current_user.two_factor_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Two-factor authentication is already enabled"
        )
    
    if not current_user.two_factor_secret:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Two-factor authentication setup not initiated"
        )
    
    totp = pyotp.TOTP(current_user.two_factor_secret)
    if not totp.verify(data.token, valid_window=1):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid verification token"
        )
    
    current_user.two_factor_enabled = True
    db.commit()
    
    return {"message": "Two-factor authentication enabled successfully"}

@router.post("/verify")
async def verify_two_factor(
    data: TwoFactorVerifyRequest,
    current_user: User = Depends(get_current_user)
):
    """Verify a 2FA token (for ongoing authentication)"""
    if not current_user.two_factor_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Two-factor authentication is not enabled"
        )
    
    totp = pyotp.TOTP(current_user.two_factor_secret)
    if not totp.verify(data.token, valid_window=1):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid verification token"
        )
    
    return {"message": "Token verified successfully"}

@router.post("/verify-backup")
async def verify_backup_code(
    data: BackupCodeVerifyRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Verify a backup code and mark it as used"""
    if not current_user.two_factor_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Two-factor authentication is not enabled"
        )
    
    if not current_user.backup_codes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No backup codes available"
        )
    
    try:
        backup_codes = json.loads(current_user.backup_codes)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Invalid backup codes format"
        )
    
    if data.backup_code.upper() not in backup_codes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid backup code"
        )
    
    backup_codes.remove(data.backup_code.upper())
    current_user.backup_codes = json.dumps(backup_codes)
    db.commit()
    
    return {"message": "Backup code verified successfully", "remaining_codes": len(backup_codes)}

@router.post("/disable")
async def disable_two_factor(
    data: TwoFactorVerifyRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Disable 2FA after verifying current token"""
    if not current_user.two_factor_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Two-factor authentication is not enabled"
        )
    
    totp = pyotp.TOTP(current_user.two_factor_secret)
    if not totp.verify(data.token, valid_window=1):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid verification token"
        )
    
    current_user.two_factor_enabled = False
    current_user.two_factor_secret = None
    current_user.backup_codes = None
    db.commit()
    
    return {"message": "Two-factor authentication disabled successfully"}

@router.get("/status")
async def get_two_factor_status(current_user: User = Depends(get_current_user)):
    """Get current 2FA status for user"""
    backup_codes_count = 0
    if current_user.backup_codes:
        try:
            backup_codes = json.loads(current_user.backup_codes)
            backup_codes_count = len(backup_codes)
        except json.JSONDecodeError:
            pass
    
    return {
        "enabled": current_user.two_factor_enabled,
        "backup_codes_remaining": backup_codes_count if current_user.two_factor_enabled else 0
    }

@router.post("/complete-login")
async def complete_two_factor_login(
    data: TwoFactorLoginRequest,
    db: Session = Depends(get_db)
):
    """Complete login process with 2FA verification"""
    user = authenticate_user(db, data.username, data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    if not user.two_factor_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Two-factor authentication is not enabled for this user"
        )
    
    totp = pyotp.TOTP(user.two_factor_secret)
    if not totp.verify(data.token, valid_window=1):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid verification token"
        )
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    refresh_token, _ = create_refresh_token_pair(db, user)
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "requires_2fa": False
    }
