import os
from pathlib import Path

# Корневая директория проекта
BASE_DIR = Path(__file__).resolve().parent.parent

# Директории
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Убедимся, что директории существуют
for d in [DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# Настройки модели по умолчанию
RANDOM_STATE = 42
N_ESTIMATORS = 100

# MLflow
MLFLOW_EXPERIMENT_NAME = "hw5_ci_cd_experiment"