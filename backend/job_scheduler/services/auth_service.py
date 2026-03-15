import hashlib
import os
import uuid as uuid_lib
from datetime import datetime, timedelta, timezone
from typing import Optional
import bcrypt
import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from job_scheduler.db.session import get_db
from job_scheduler.models.user import User
from job_scheduler.schemas.auth import TokenData

_SECRET_KEY = os.environ.get("JWT_SECRET_KEY")
SECRET_KEY = _SECRET_KEY.strip()
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(
    os.environ.get("JWT_EXPIRE_MINUTES", 60 * 24 * 7)
)  # 7 days default

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def get_token_from_request(request: Request) -> str:
    """Get JWT from Authorization header."""
    auth = request.headers.get("Authorization")
    if auth and auth.startswith("Bearer "):
        token = auth[7:].strip()
        if token:
            return token
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


def _password_digest(password: str) -> bytes:
    """SHA256 digest of password so bcrypt always receives <= 72 bytes."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest().encode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(
        _password_digest(plain_password),
        (
            hashed_password.encode("utf-8")
            if isinstance(hashed_password, str)
            else hashed_password
        ),
    )


def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(_password_digest(password), bcrypt.gensalt()).decode("utf-8")


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return str(encoded_jwt)


def get_current_user(request: Request, db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        token = get_token_from_request(request)
    except HTTPException:
        raise
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        username = payload.get("username")
        if user_id is None:
            raise credentials_exception
        token_data = TokenData(user_id=user_id, username=username)
    except jwt.InvalidTokenError:
        raise credentials_exception
    try:
        user_uuid = uuid_lib.UUID(token_data.user_id)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session invalid. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = db.query(User).filter(User.id == user_uuid).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session invalid. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user
