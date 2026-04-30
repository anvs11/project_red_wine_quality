from fastapi import FastAPI, HTTPException, status
from contextlib import asynccontextmanager

from .schemas import (WineFeatures, PredictionResponse,
                      HealthResponse, ModelInfoResponse)
from .model_loader import ModelLoader

model_loader: ModelLoader = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация при старте приложения"""
    global model_loader
    print("Starting Wine Quality API...")
    model_loader = ModelLoader()
    if not model_loader.is_loaded():
        print("Model not loaded — /predict will return 503")
    yield
    print("Shutting down...")

app = FastAPI(
    title="Wine Quality Prediction API",
    description="ML model for predicting wine quality",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/", tags=["Root"])
def read_root():
    return {
        "service": "Wine Quality API",
        "docs": "/docs",
        "endpoints": ["/predict", "/health", "/model-info"]
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(data: WineFeatures, force_reload: bool = False):
    """
    Предсказание качества вина по 11 признакам.

    Возвращает:
    - prediction: 'good' или 'bad'
    - probability: уверенность модели
    - model_version: хэш версии из DVC

    force_reload: если True — проверит актуальность модели перед предсказанием
    """
    # Если модель не загружена или запрошена перезагрузка — пробуем обновить
    if not model_loader or not model_loader.is_loaded() or force_reload:
        if not model_loader or not model_loader.reload():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Check /health endpoint."
            )

    try:
        features = [
            data.fixed_acidity,
            data.volatile_acidity,
            data.citric_acid,
            data.residual_sugar,
            data.chlorides,
            data.free_sulfur_dioxide,
            data.total_sulfur_dioxide,
            data.density,
            data.pH,
            data.sulphates,
            data.alcohol
        ]

        label, probability = model_loader.predict(features)

        return PredictionResponse(
            prediction=label,
            probability=round(probability, 4),
            model_version=model_loader.version
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Проверка работоспособности сервиса"""
    if model_loader and model_loader.is_loaded():
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            message="Model loaded and ready"
        )
    else:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            message="Model not loaded — predictions unavailable"
        )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Metadata"])
async def model_info():
    """Метаданные модели: тип, метрики, версия
    (читаются из файла, созданного при обучении)"""
    if not model_loader or not model_loader.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded — metadata unavailable"
        )
    meta = model_loader.get_metadata()

    return ModelInfoResponse(
        model_name=meta.get("model_name", "unknown"),
        model_type=meta.get("model_type", "unknown"),
        metrics=meta.get("metrics", {}),
        dvc_version=model_loader.version,
        trained_on=meta.get("trained_on")
    )
