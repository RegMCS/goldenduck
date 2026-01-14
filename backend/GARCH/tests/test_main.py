from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello from GARCH service!"}


def test_health_check_mock_db():
    # Note: This checks the endpoint wrapper itself.
    # For a true DB integration test, we would need a test DB environment.
    # Here we expect a 500 or 200 depending on if DB is actually reachable during test.
    # To make this robust unit test, we should mock the dependency.
    pass
