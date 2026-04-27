from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_predict_missing_field():
    """Тест: отсутствует обязательное поле → 422"""
    payload = {
        "fixed_acidity": 7.4,
        # отсутствует volatile_acidity
        "citric_acid": 0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11,
        "total_sulfur_dioxide": 34,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_type():
    """Тест: неверный тип данных → 422"""
    payload = {
        "fixed_acidity": "invalid",  # должно быть float
        "volatile_acidity": 0.7,
        "citric_acid": 0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11,
        "total_sulfur_dioxide": 34,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_extra_field_ignored():
    """Тест: лишнее поле игнорируется (Pydantic) → 200 или 503"""
    payload = {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.7,
        "citric_acid": 0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11,
        "total_sulfur_dioxide": 34,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4,
        "extra_field": "should_be_ignored"  # Pydantic игнорирует лишние поля
    }

    response = client.post("/predict", json=payload)
    # Может быть 200 (если модель загружена) или 503 (если нет)
    assert response.status_code in [200, 503]
