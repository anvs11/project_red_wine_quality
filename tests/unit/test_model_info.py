from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_model_info_structure():
    """Проверка структуры ответа /model-info"""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()

    assert "model_name" in data
    assert "model_type" in data
    assert "metrics" in data

    assert data["model_type"] in ["random_forest", "catboost"]
