import uuid

import pytest
from fastapi import Depends
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session

from job_scheduler.db.base import Base
from job_scheduler.db.session import SessionLocal, engine, get_db
from job_scheduler.main import app
from job_scheduler.models.user import User
from job_scheduler.services.auth_service import get_current_user, get_password_hash


@pytest.fixture(scope="session", autouse=True)
def ensure_tables():
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture
def clean_users():
    """Delete dependent jobs, then users, before the test runs."""
    with SessionLocal() as db:
        db.execute(text("DELETE FROM training_jobs"))
        db.execute(text("DELETE FROM ai_model_jobs"))
        db.execute(text("DELETE FROM users"))
        db.commit()
    yield


@pytest.fixture
def test_db(clean_users):
    """Provide a DB session backed by a clean users table."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def test_admin_user(test_db):
    user = User(
        id=uuid.uuid4(),
        username="admin_user",
        hashed_password=get_password_hash("adminpass"),
        first_name=None,
        last_name=None,
        is_admin=True,
    )
    test_db.add(user)
    test_db.commit()
    test_db.refresh(user)
    return user


@pytest.fixture
def test_user(test_db):
    user = User(
        id=uuid.uuid4(),
        username="regular_user",
        hashed_password=get_password_hash("userpass"),
        first_name=None,
        last_name=None,
        is_admin=False,
    )
    test_db.add(user)
    test_db.commit()
    test_db.refresh(user)
    return user


@pytest.fixture
def admin_client(test_admin_user):
    """TestClient whose requests are authenticated as an admin user."""
    admin_id = test_admin_user.id

    def override_current_user(db: Session = Depends(get_db)):
        return db.query(User).filter(User.id == admin_id).first()

    app.dependency_overrides[get_current_user] = override_current_user
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture
def user_client(clean_users):
    """TestClient whose requests are authenticated as a non-admin user."""
    user_id = uuid.uuid4()
    with SessionLocal() as db:
        user = User(
            id=user_id,
            username="regular_client_user",
            hashed_password=get_password_hash("pass"),
            first_name=None,
            last_name=None,
            is_admin=False,
        )
        db.add(user)
        db.commit()

    def override_current_user(db: Session = Depends(get_db)):
        return db.query(User).filter(User.id == user_id).first()

    app.dependency_overrides[get_current_user] = override_current_user
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.pop(get_current_user, None)
