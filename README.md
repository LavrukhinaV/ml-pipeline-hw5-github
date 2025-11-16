# ML Pipeline HW5 – CI/CD + Monitoring

Учебный проект для домашнего задания №5 по модулю «CI/CD для ML».

## Структура проекта

- `src/` — исходный код:
  - `data.py` — загрузка и подготовка данных
  - `config.py` — базовые настройки и константы
  - `train.py` — обучение модели и логирование в MLflow
- `data/` — входные данные (не хранятся в Git)
- `models/` — сохранённые модели (крупные файлы, исключены из Git)
- `reports/` — отчёты Deepchecks, Evidently, артефакты экспериментов
- `.github/workflows/ci.yml` — конфигурация GitHub Actions
- `.gitlab-ci.yml` — конфигурация GitLab CI
- `Dockerfile` — описание контейнера
- `requirements.txt` — список зависимостей

## Быстрый старт (локально)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/train.py