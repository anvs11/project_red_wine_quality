import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import argparse
import os
import json

from pathlib import Path
from config import config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix
)


def save_confusion_matrix(y_true, y_pred, filename="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return filename


def load_and_preprocess(path):
    """Загрузка данных и бинарное представление quality"""
    label_quality = LabelEncoder()
    df = pd.read_csv(path, sep=',')
    df['quality'] = pd.cut(df['quality'], bins=config["quality_bins"], labels=config["quality_labels"])
    df['quality'] = label_quality.fit_transform(df['quality'])
    X = df.drop('quality', axis=1)
    y = df['quality']
    print(f"Class distribution: {y.value_counts().to_dict()}")
    return X, y


def split_dataset(X, y):
    # Подсчет дисбаланса
    class_counts = pd.Series(y).value_counts()
    imbalance_ratio = class_counts.max() / class_counts.min()
    use_stratify = imbalance_ratio > config["data"]["imbalance_threshold"]

    split_kwargs = {
        "test_size": config["data"]["test_size"],
        "random_state": config["random_state"]
    }

    if use_stratify:
        split_kwargs["stratify"] = y
        print(f"Данные несбалансированы (ratio: {imbalance_ratio:.2f}). Включена стратификация.")
    else:
        print(f"Данные сбалансированы (ratio: {imbalance_ratio:.2f}). Обычное разбиение.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, **split_kwargs)
    return X_train, X_test, y_train, y_test


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def train_and_log(model_name, X_train, y_train, X_test, y_test, params):
    with mlflow.start_run(run_name=f"{model_name}_exp"):
        mlflow.log_params(params)
        mlflow.log_param("model_type", model_name)

        if model_name == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(random_state=config["random_state"], **params)
            model_path = config["model_paths"][model_name]
        elif model_name == "catboost":
            from catboost import CatBoostClassifier
            model = CatBoostClassifier(random_state=config["random_state"], **params)
            model_path = config["model_paths"][model_name]
        else:
            raise ValueError(f"Unknown model: {model_name}")

        model = train_model(model, X_train, y_train)
        os.makedirs("models", exist_ok=True)
        if model_name == "random_forest":
            joblib.dump(model, model_path)
            # mlflow.sklearn.log_model(model, "model")
            print(f"Model saved: {model_path}")
        else:
            model.save_model(model_path)
            # mlflow.log_artifact(model_path, "model")
            print(f"Model saved: {model_path}")

        roc_auc, metrics = evaluate_and_log(model, X_test, y_test,
                                            run_name=f"{model_name}_exp",
                                            model_type=model_name)

        metadata = {
            "model_name": "wine-quality-classifier",
            "model_type": model_name,
            "metrics": {
                "accuracy": metrics["accuracy"],
                "roc_auc": metrics["roc_auc"],
                "f1_binary": metrics["f1_binary"],
                "precision": metrics["precision"],
                "recall": metrics["recall"]
            },
            "params": params,
            "trained_on": pd.Timestamp.now().isoformat(),
            "dvc_version": None
        }

        metadata_path = Path("models/model_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved: {metadata_path}")

        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"Model saved: {model_path}")
        return roc_auc


def evaluate_and_log(model, X_test, y_test, run_name, model_type):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_binary": f1_score(y_test, y_pred, average="binary"),
        "precision": precision_score(y_test, y_pred, average="binary"),
        "recall": recall_score(y_test, y_pred, average="binary"),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }

    for name, value in metrics.items():
        mlflow.log_metric(name, value)
        print(f"{name}: {value:.4f}")

    # cm_path = save_confusion_matrix(y_test, y_pred, f"cm_{run_name}.png")
    # mlflow.log_artifact(cm_path)
    return metrics["roc_auc"], metrics


if __name__ == "__main__":
    # === ОПРЕДЕЛЯЕМ СРЕДУ: Docker или локально ===
    if os.getenv("AIRFLOW_HOME"):
        # Внутри контейнера Airflow (Linux): файловый бэкенд в смонтированной папке
        mlflow.set_tracking_uri("file:///opt/airflow/project/mlruns")
        os.environ["GIT_PYTHON_REFRESH"] = "quiet"  # подавляем предупреждение о git
    else:
        # Локально на хосте (Windows)
        mlflow.set_tracking_uri("file:./mlruns")

    mlflow.set_experiment("wine-quality")
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # Парсинг аргументов
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["random_forest", "catboost"])
    args = parser.parse_args()

    # Загрузка данных
    X, y = load_and_preprocess(config["data_path"])
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # Запуск обучения одной модели
    params = config["models"][args.model]
    roc_auc = train_and_log(args.model, X_train, y_train, X_test, y_test, params)

    print(f"\n {args.model} finished with ROC-AUC: {roc_auc:.4f}")
