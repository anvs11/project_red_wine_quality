config = {
    "quality_bins": (2, 6.5, 8),
    "quality_labels": ['bad', 'good'],
    "random_state": 42,
    "model_paths": {
        "random_forest": "models/rf_model.pkl",
        "catboost": "models/catboost_model.cbm"
    },
    "active_model": "random_forest",
    "data_path": "data/winequality-red.csv",
    "data": {
        "test_size": 0.25,
        "imbalance_threshold": 1.5  # если соотношение классов > 1.5, включаем стратификацию
    },
    "models": {
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_leaf": 1,
            "class_weight": "balanced"
        },
        "catboost": {
            "iterations": 500,
            "learning_rate": 0.03,
            "depth": 6,
            "verbose": 0
        }
    }
}
