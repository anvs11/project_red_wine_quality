from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_full_predict_flow():
    """
    Интеграционный тест: полный сценарий предсказания.

    1. Проверяем, что сервис здоров (/health)
    2. Отправляем валидный запрос на /predict
    3. Проверяем структуру ответа
    """
    health_response = client.get("/health")
    assert health_response.status_code == 200

    # валидный запрос на предсказание
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
        "alcohol": 9.4
    }

    predict_response = client.post("/predict", json=payload)

    # если модель загружена — ожидаем 200 и валидный ответ
    if predict_response.status_code == 200:
        data = predict_response.json()
        assert "prediction" in data
        assert data["prediction"] in ["good", "bad"]
        assert "probability" in data
        assert 0 <= data["probability"] <= 1
        assert "model_version" in data

    # если модель не загружена — ожидаем 503
    elif predict_response.status_code == 503:
        assert "Model not loaded" in predict_response.json()["detail"]

    else:
        # любой другой статус — ошибка теста
        assert False, f"Unexpected status code: {predict_response.status_code}"