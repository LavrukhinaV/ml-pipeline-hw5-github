import os
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from config import (
    MODELS_DIR,
    RANDOM_STATE,
    N_ESTIMATORS,
    MLFLOW_EXPERIMENT_NAME,
)
from data import load_iris_data


def main():
    # Загрузка данных
    X_train, X_test, y_train, y_test = load_iris_data(
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    # Настройка MLflow
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run():
        # Параметры модели
        params = {
            "n_estimators": N_ESTIMATORS,
            "random_state": RANDOM_STATE,
        }

        # Логируем параметры
        for k, v in params.items():
            mlflow.log_param(k, v)

        # Обучаем модель
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Предсказания и метрика
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Логируем метрику
        mlflow.log_metric("accuracy", acc)

        # Сохраняем модель как артефакт
        os.makedirs(MODELS_DIR, exist_ok=True)
        model_path = Path(MODELS_DIR) / "rf_iris.pkl"
        mlflow.sklearn.save_model(model, path=str(model_path))

        # Сохраняем модель также в MLflow артефакты
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"Accuracy: {acc:.4f}")
        print(f"Model saved to: {model_path}")

        # TODO: на следующих этапах добавить:
        # - Deepchecks отчёт и mlflow.log_artifact(...)
        # - Evidently отчёт и mlflow.log_artifact(...)


if __name__ == "__main__":
    main()
