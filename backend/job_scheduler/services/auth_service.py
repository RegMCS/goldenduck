import hashlib
import os
import uuid as uuid_lib
from datetime import datetime, timedelta, timezone
from typing import Optional
import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from job_scheduler.db.session import get_db
from job_scheduler.models.user import User


SECRET_KEY = (os.environ.get("JWT_SECRET_KEY") or "").strip()
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("JWT_EXPIRE_MINUTES") or 60 * 24 * 7)

# auto_error=False so we receive None instead of an automatic 401 when no
# token is present — lets dev-mode bypass work cleanly.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


def _is_dev_no_auth() -> bool:
    """True when auth can be skipped (dev only). Set NODE_ENV=development or NODE_ENV=dev."""
    env = (os.environ.get("NODE_ENV") or "").lower()
    return env in ("development", "dev")


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
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return str(jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM))


def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
):
    # Dev mode: skip auth, return first user or auto-create admin.
    if _is_dev_no_auth() and not token:
        user = db.query(User).order_by(User.username).first()
        if not user:
            user = User(
                username="admin",
                hashed_password=get_password_hash("admin"),
                first_name=None,
                last_name=None,
            )
            db.add(user)
            db.commit()
            db.refresh(user)
        return user

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if not user_id:
            raise ValueError("missing sub")
        user_uuid = uuid_lib.UUID(user_id)
    except (jwt.InvalidTokenError, ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session invalid. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = db.query(User).filter(User.id == user_uuid).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session invalid. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user
