import os
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import train_test_validation

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from config import (
    MODELS_DIR,
    REPORTS_DIR,
    RANDOM_STATE,
    N_ESTIMATORS,
    MLFLOW_EXPERIMENT_NAME,
)
from data import load_iris_data

# Запуск Deepchecks train_test_validation и сохранение HTML-отчёта.
def run_deepchecks_suite(X_train, y_train, X_test, y_test) -> Path:
    train_ds = Dataset(X_train, label=y_train)
    test_ds = Dataset(X_test, label=y_test)

    suite = train_test_validation()
    result = suite.run(train_ds, test_ds)

    report_path = REPORTS_DIR / "deepchecks_train_test.html"
    result.save_as_html(str(report_path))
    return report_path

def run_evidently_report(X_train, X_test) -> Path:
    """Генерация отчёта Data Drift через Evidently."""
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=X_train, current_data=X_test)

    report_path = REPORTS_DIR / "evidently_data_drift.html"
    report.save_html(str(report_path))

    return report_path

def main():
    # Загрузка данных
    X_train, X_test, y_train, y_test = load_iris_data(
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    
    # Запуск Deepchecks train_test_validation и сохранение HTML-отчёта.
    dc_report_path = run_deepchecks_suite(X_train, y_train, X_test, y_test)
    print(f"Deepchecks report saved to: {dc_report_path}")
    
    # Запуск Evidently report и сохранение HTML-отчёта.
    evidently_report_path = run_evidently_report(X_train, X_test)
    print(f"Evidently report saved to: {evidently_report_path}")

    # Настройка MLflow
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run():
        # Логируем путь к отчёту как артефакт
        if dc_report_path.exists():
            mlflow.log_artifact(str(dc_report_path), artifact_path="deepchecks")

        # Логируем путь к отчёту как артефакт
        if evidently_report_path.exists():
            mlflow.log_artifact(str(evidently_report_path), artifact_path="evidently")

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

if __name__ == "__main__":
    main()
