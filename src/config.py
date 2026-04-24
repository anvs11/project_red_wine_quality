config = {
    "random_state": 42,
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