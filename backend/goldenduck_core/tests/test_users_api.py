import pytest
from fastapi.testclient import TestClient


def test_list_users_admin(admin_client, test_db, test_user):
    response = admin_client.get("/api/users")
    assert response.status_code == 200
    data = response.json()
    assert "users" in data
    assert len(data["users"]) >= 1


def test_list_users_non_admin(user_client):
    response = user_client.get("/api/users")
    assert response.status_code == 403


def test_toggle_admin_status(admin_client, test_db, test_user):
    user_id = str(test_user.id)
    # Check it can update
    response = admin_client.put(f"/api/users/{user_id}", json={"is_admin": True})
    assert response.status_code == 200
    data = response.json()
    assert data["is_admin"] is True


def test_cannot_toggle_own_admin_status(admin_client, test_admin_user):
    user_id = str(test_admin_user.id)
    response = admin_client.put(f"/api/users/{user_id}", json={"is_admin": False})
    assert response.status_code == 400
    assert "cannot modify your own admin status" in response.json()["detail"]
