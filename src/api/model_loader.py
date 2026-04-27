import subprocess
from pathlib import Path
from typing import Optional, Tuple


class ModelLoader:
    """Загружает модель из DVC и кэширует в памяти"""
    def __init__(self, model_path: str = "models/rf_model.pkl"):
        self.model_path = Path(model_path)
        self.metadata_path = Path("models/model_metadata.json")
        self.model = None
        self.metadata = {}
        self.version = None
        self._load()

    def _load_metadata(self) -> dict:
        """Загружает метаданные из файла"""
        if self.metadata_path.exists():
            try:
                import json
                with open(self.metadata_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Failed to load metadata: {e}")
        # Fallback: базовые значения
        return {
            "model_name": "wine-quality-classifier",
            "model_type": "RandomForestClassifier",
            "metrics": {},
            "trained_on": None
        }

    def _pull_from_dvc(self) -> bool:
        """Выполняет `dvc pull` для получения актуальной модели"""
        try:
            result = subprocess.run(
                ["dvc", "pull", str(self.model_path)],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent
            )
            return result.returncode == 0
        except Exception as e:
            print(f"DVC pull failed: {e}")
            return False

    def _get_dvc_version(self) -> Optional[str]:
        """Получает хэш версии модели из DVC-метафайла"""
        dvc_file = Path(str(self.model_path) + ".dvc")
        if dvc_file.exists():
            try:
                import yaml
                with open(dvc_file, 'r') as f:
                    meta = yaml.safe_load(f)
                    outs = meta.get('outs', [])
                    if outs and outs[0]:
                        return outs[0].get('md5') or outs[0].get('hash')
            except Exception:
                pass
        return None

    def _load(self) -> bool:
        """Загружает модель и метаданные"""
        if not self.model_path.exists():
            print("Model not found locally, trying DVC pull...")
            self._pull_from_dvc()
        if self.model_path.exists():
            try:
                import joblib
                self.model = joblib.load(self.model_path)
                self.version = self._get_dvc_version()
                print(f"Model loaded: {self.model_path} (version: {self.version})")
            except Exception as e:
                print(f"Failed to load model: {e}")
                return False
        else:
            print(f"Model file not found: {self.model_path}")
            return False

        self.metadata = self._load_metadata()
        print(f"Metadata loaded: {self.metadata.get('model_type', 'unknown')}")
        return True

    def predict(self, features: list) -> Tuple[str, float]:
        """Делает предсказание: возвращает (класс, вероятность)"""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        import numpy as np
        X = np.array([features])
        proba = self.model.predict_proba(X)[0]
        pred_class = int(self.model.predict(X)[0])
        label = "good" if pred_class == 1 else "bad"
        probability = float(proba[pred_class])
        return label, probability

    def is_loaded(self) -> bool:
        return self.model is not None

    def reload(self) -> bool:
        """Перезагружает модель (для hot-reload)"""
        self.model = None
        return self._load()

    def get_metadata(self) -> dict:
        """Возвращает метаданные модели"""
        return self.metadata.copy()
