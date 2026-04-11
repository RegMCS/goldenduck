import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from goldenduck_core.db.session import get_db
from goldenduck_core.models.user import User
from goldenduck_core.schemas.auth import UserListResponse, UserResponse, UserAdminUpdate
from goldenduck_core.services.auth_service import require_admin

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/users", tags=["users"])


@router.get("", response_model=UserListResponse)
def list_users(
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """Return all users for admin management."""
    users = db.query(User).order_by(User.username.asc()).all()
    return UserListResponse(users=users)


@router.put("/{user_id}", response_model=UserResponse)
def update_user_admin(
    user_id: str,
    update_data: UserAdminUpdate,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """Allow an admin to toggle another user's admin status."""
    import uuid

    try:
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user_id format")

    if str(admin.id) == user_id:
        raise HTTPException(
            status_code=400, detail="You cannot modify your own admin status"
        )

    user = db.query(User).filter(User.id == user_uuid).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.is_admin = update_data.is_admin
    db.commit()
    db.refresh(user)

    logger.info(
        "Admin %s updated user %s is_admin=%s",
        admin.username,
        user.username,
        user.is_admin,
    )

    return user
