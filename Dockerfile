# syntax=docker/dockerfile:1.7
# ================== Stage 1: trainer ==================
# Тут є важкі залежності для навчання, датасет і сам процес навчання.
# У фінальний образ нічого з цього не потрапляє, крім папки artifacts/.
FROM python:3.10-slim AS trainer

ARG EPOCHS=10
ARG SEED=42
ARG MAX_PER_CLASS=0
ARG MODEL_VERSION=1.0.0

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Залежності йдуть першими, щоб зміна коду не інвалідувала кеш pip
COPY requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt

# Завантаження датасету — окремий шар. Копіюємо лише download.py,
# тож правки в train.py/api.py не змушують завантажувати 2.3 ГБ повторно.
COPY src/download.py ./src/download.py
RUN python src/download.py --classes yes no up down --out /build/data

# Копіюємо лише файли, потрібні для навчання: правка api.py не інвалідує кеш і не запускає навчання знову
COPY src/__init__.py src/audio.py src/dataset.py src/evaluate.py src/model.py src/train.py ./src/
# Навчання + оцінка -> /build/artifacts (model.pt, labels.json, metrics.json, samples, testset)
RUN python -m src.train --classes yes no up down \
        --epochs ${EPOCHS} --seed ${SEED} --max-per-class ${MAX_PER_CLASS} \
        --version ${MODEL_VERSION} --data-root /build/data --out /build/artifacts

# ================== Stage 2: runtime ==================
FROM python:3.10-slim AS runtime

ARG MODEL_VERSION=1.0.0
ARG GIT_COMMIT=unknown
ARG BUILD_DATE=unknown

LABEL org.opencontainers.image.title="kws-api" \
      org.opencontainers.image.version="${MODEL_VERSION}" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.created="${BUILD_DATE}"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ ./src/
COPY --from=trainer /build/artifacts ./artifacts

ENV MODEL_PATH=/app/artifacts/model.pt \
    PORT=8000 \
    LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1

# Запуск не від root
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-m", "src.healthcheck"]

# shell-форма з exec, щоб підставити $PORT і при цьому передавати сигнали uvicorn
CMD ["sh", "-c", "exec uvicorn src.api:app --host 0.0.0.0 --port ${PORT}"]
