from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_check():
    """Проверка эндпоинта /health"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
