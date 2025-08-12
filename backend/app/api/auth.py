"""
Authentication API routes
"""

from datetime import timedelta, datetime
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional

from app.database import get_db
from app.models.user import User
from app.utils.auth import authenticate_user, create_access_token, get_current_user, create_refresh_token_pair, rotate_refresh_token, verify_refresh_token
from app.utils.rate_limit import rate_limit
from app.config import settings

router = APIRouter()

# Pydantic models for request/response
class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    requires_2fa: bool = False

class TokenData(BaseModel):
    username: Optional[str] = None

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: str
    role: str
    department: Optional[str]
    designation: Optional[str]
    is_active: bool
    can_upload: bool
    is_engineer: bool
    is_admin: bool
    login_count: int
    
    class Config:
        from_attributes = True

class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    full_name: str
    employee_id: Optional[str] = None
    department: Optional[str] = None
    designation: Optional[str] = None
    role: str = "operator"

class RefreshTokenRequest(BaseModel):
    token: str

class ResetPasswordRequest(BaseModel):
    username: str
    new_password: str

@router.post("/login", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
    request: Request = None
):
    """Authenticate user and return access token"""
    # Rate limit login attempts per IP
    client_ip = request.client.host if request and request.client else "unknown"
    if not rate_limit(f"login:{client_ip}", limit=10, window_seconds=60):
        raise HTTPException(status_code=429, detail="Too many login attempts. Please try again later.")
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    # Check if 2FA is required
    if user.two_factor_enabled:
        return {
            "access_token": "",
            "refresh_token": "",
            "token_type": "bearer",
            "requires_2fa": True
        }
    
    device_info = f"{request.headers.get('user-agent', 'unknown')}|{client_ip}" if request else None
    refresh_token, _ = create_refresh_token_pair(db, user, device_info)
    
    return {
        "access_token": access_token, 
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "requires_2fa": False
    }

@router.post("/refresh", response_model=Token)
async def refresh_token(data: RefreshTokenRequest, request: Request = None, db: Session = Depends(get_db)):
    """Issue new access and refresh tokens using refresh token rotation."""
    try:
        user = verify_refresh_token(db, data.token)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid refresh token")
        
        client_ip = request.client.host if request and request.client else "unknown"
        device_info = f"{request.headers.get('user-agent', 'unknown')}|{client_ip}" if request else None
        
        new_refresh_token, _ = rotate_refresh_token(db, data.token, device_info)
        if not new_refresh_token:
            raise HTTPException(status_code=401, detail="Token rotation failed")
        
        # Create new access token
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username}, expires_delta=access_token_expires
        )
        
        return {
            "access_token": access_token,
            "refresh_token": new_refresh_token,
            "token_type": "bearer"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail="Token refresh failed")

@router.post("/reset-password")
async def reset_password(req: ResetPasswordRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Admin-only reset of a user's password."""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    user = db.query(User).filter((User.username == req.username) | (User.email == req.username)).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.set_password(req.new_password)
    db.commit()
    return {"message": "Password reset successful"}

@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return current_user

@router.post("/register", response_model=UserResponse)
async def register_user(
    user_data: UserCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Register a new user (admin only)"""
    # Check if current user is admin
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can register new users"
        )
    
    # Check if username or email already exists
    existing_user = db.query(User).filter(
        (User.username == user_data.username) | (User.email == user_data.email)
    ).first()
    
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email already registered"
        )
    
    # Create new user
    new_user = User(
        username=user_data.username,
        email=user_data.email,
        full_name=user_data.full_name,
        employee_id=user_data.employee_id,
        department=user_data.department,
        designation=user_data.designation,
        role=user_data.role
    )
    new_user.set_password(user_data.password)
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return new_user

@router.get("/users", response_model=list[UserResponse])
async def list_users(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    skip: int = 0,
    limit: int = 100
):
    """List all users (admin and engineer access)"""
    if not current_user.is_engineer:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Engineer access required"
        )
    
    users = db.query(User).offset(skip).limit(limit).all()
    return users

@router.put("/users/{user_id}/activate")
async def activate_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Activate/deactivate a user (admin only)"""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    user.is_active = not user.is_active
    db.commit()
    
    return {"message": f"User {'activated' if user.is_active else 'deactivated'} successfully"}

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "auth"}      