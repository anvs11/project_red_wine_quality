"""
Интеграционный тест: проверка полного DVC-цикла
Запуск: pytest tests/integration/test_dvc_cycle.py -v
"""
import os
import subprocess
import pytest
from pathlib import Path


@pytest.mark.skipif(
    not os.getenv("AWS_ACCESS_KEY_ID"),
    reason="Requires DVC credentials (skip in local CI)"
)
def test_dvc_pull_train_add_cycle():
    """
    Проверяет, что:
    1. dvc pull данных работает
    2. обучение запускается без ошибок
    3. dvc add создаёт .dvc-файл
    """
    project_root = Path(__file__).parent.parent.parent

    result = subprocess.run(
        ["dvc", "pull", "data/winequality-red.csv"],
        cwd=project_root,
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"dvc pull failed: {result.stderr}"

    result = subprocess.run(
        ["python", "src/train.py", "--model", "random_forest"],
        cwd=project_root,
        capture_output=True,
        text=True,
        env={**os.environ, "MLFLOW_TRACKING_URI": "file:./mlruns_test"}
    )
    assert result.returncode == 0, f"train.py failed: {result.stderr}"

    dvc_file = project_root / "models" / "rf_model.pkl.dvc"
    assert dvc_file.exists(), "Model .dvc file not created"

    import yaml
    with open(dvc_file, 'r') as f:
        meta = yaml.safe_load(f)
    assert 'outs' in meta and len(meta['outs']) > 0
    assert 'md5' in meta['outs'][0] or 'hash' in meta['outs'][0]
