from pydantic import BaseModel, Field
from typing import Optional


class WineFeatures(BaseModel):
    """Входные данные: 11 признаков вина"""
    fixed_acidity: float = Field(..., description="Fixed acidity (g/dm³)")
    volatile_acidity: float = Field(..., description="Volatile acidity (g/dm³)")
    citric_acid: float = Field(..., description="Citric acid (g/dm³)")
    residual_sugar: float = Field(..., description="Residual sugar (g/dm³)")
    chlorides: float = Field(..., description="Chlorides (g/dm³)")
    free_sulfur_dioxide: float = Field(..., description="Free sulfur dioxide (mg/dm³)")
    total_sulfur_dioxide: float = Field(..., description="Total sulfur dioxide (mg/dm³)")
    density: float = Field(..., description="Density (g/cm³)")
    pH: float = Field(..., description="pH")
    sulphates: float = Field(..., description="Sulphates (g/dm³)")
    alcohol: float = Field(..., description="Alcohol (% by volume)")

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictionResponse(BaseModel):
    """Ответ эндпоинта /predict"""
    prediction: str = Field(..., description="Predicted quality: 'good' or 'bad'")
    probability: float = Field(..., description="Probability of 'good' class")
    model_version: Optional[str] = Field(None, description="DVC version hash")


class HealthResponse(BaseModel):
    """Ответ эндпоинта /health"""
    status: str
    model_loaded: bool
    message: Optional[str] = None


class ModelInfoResponse(BaseModel):
    """Ответ эндпоинта /model-info"""
    model_name: str
    model_type: str
    metrics: dict
    dvc_version: Optional[str]
    trained_on: Optional[str]
