FROM python:3.10-slim

WORKDIR /app

# Устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код
COPY src ./src

# Директории для моделей и отчётов
RUN mkdir -p models reports

# Точка входа: обучение модели
CMD ["python", "src/train.py"]
