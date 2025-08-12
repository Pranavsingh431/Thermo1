"""
Authentication utilities
"""

from datetime import datetime, timedelta
from typing import Optional, Tuple
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.config import settings
from app.database import get_db
from app.models.user import User

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT token scheme
security = HTTPBearer()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def create_refresh_token_pair(db: Session, user: User, device_info: str = None) -> Tuple[str, str]:
    """Create a new refresh token with family rotation support"""
    from app.models.refresh_token import RefreshToken
    
    token = RefreshToken.generate_token()
    family_id = RefreshToken.generate_family_id()
    
    refresh_token = RefreshToken(
        token=token,
        family_id=family_id,
        user_id=user.id,
        expires_at=datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
        device_info=device_info
    )
    
    db.add(refresh_token)
    db.commit()
    
    return token, family_id

def rotate_refresh_token(db: Session, old_token: str, device_info: str = None) -> Optional[Tuple[str, str]]:
    """Rotate refresh token using family-based approach"""
    from app.models.refresh_token import RefreshToken
    
    old_refresh_token = db.query(RefreshToken).filter(
        RefreshToken.token == old_token,
        RefreshToken.is_revoked == False
    ).first()
    
    if not old_refresh_token or not old_refresh_token.is_valid():
        if old_refresh_token:
            revoke_token_family(db, old_refresh_token.family_id)
        return None
    
    # Update last used
    old_refresh_token.last_used = datetime.utcnow()
    
    new_token = RefreshToken.generate_token()
    
    new_refresh_token = RefreshToken(
        token=new_token,
        family_id=old_refresh_token.family_id,  # Same family
        user_id=old_refresh_token.user_id,
        expires_at=datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
        device_info=device_info
    )
    
    old_refresh_token.revoke()
    
    db.add(new_refresh_token)
    db.commit()
    
    return new_token, old_refresh_token.family_id

def revoke_token_family(db: Session, family_id: str) -> None:
    """Revoke all tokens in a family (security measure)"""
    from app.models.refresh_token import RefreshToken
    
    tokens = db.query(RefreshToken).filter(
        RefreshToken.family_id == family_id,
        RefreshToken.is_revoked == False
    ).all()
    
    for token in tokens:
        token.revoke()
    
    db.commit()

def verify_refresh_token(db: Session, token: str) -> Optional[User]:
    """Verify refresh token and return associated user"""
    from app.models.refresh_token import RefreshToken
    
    refresh_token = db.query(RefreshToken).filter(
        RefreshToken.token == token,
        RefreshToken.is_revoked == False
    ).first()
    
    if not refresh_token or not refresh_token.is_valid():
        return None
    
    return refresh_token.user

def verify_token(token: str) -> Optional[str]:
    """Verify a JWT token and return the username"""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
        return username
    except JWTError:
        return None

def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    """Authenticate a user with username/email and password"""
    # Try to find user by username or email
    user = db.query(User).filter(
        (User.username == username) | (User.email == username)
    ).first()
    
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    if not user.is_active:
        return None
    
    # Update login count and last login
    user.login_count += 1
    user.last_login = datetime.utcnow()
    db.commit()
    
    return user

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Get the current authenticated user"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        token = credentials.credentials
        username = verify_token(token)
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user (alias for consistency)"""
    return current_user

def require_permission(permission: str):
    """Create a dependency that requires specific permission"""
    def permission_checker(current_user: User = Depends(get_current_user)) -> User:
        if permission == "admin" and not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        elif permission == "engineer" and not current_user.is_engineer:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Engineer access required"
            )
        elif permission == "upload" and not current_user.can_upload:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Upload permission required"
            )
        elif permission == "view_all" and not current_user.can_view_all_data:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Full data access required"
            )
        return current_user
    return permission_checker  