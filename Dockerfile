# syntax=docker/dockerfile:1.7
# ================= Target: trainer =================
# Образ із залежностями для навчання і датасетом. Саме навчання запускається
# під час `docker run` (ENTRYPOINT), а модель пишеться в змонтований /artifacts.
FROM python:3.10-slim AS trainer
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Залежності йдуть першими, щоб зміна коду не інвалідувала кеш pip
COPY requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt

# Датасет — окремий шар: копіюємо лише download.py, тож правки решти коду
# не змушують завантажувати архів 2.3 ГБ повторно (шар береться з кешу GHA).
COPY src/download.py ./src/download.py
RUN python src/download.py --classes yes no up down --out /build/data

COPY src/ ./src/
# Список дикторів еталонних зразків: їх записи не беруться в навчання
COPY samples/sources.json ./samples/sources.json

# Навчання запускається під час docker run, а не docker build
ENTRYPOINT ["python", "-m", "src.train", "--data-root", "/build/data"]
CMD ["--classes", "yes", "no", "up", "down", "--epochs", "10", \
     "--seed", "42", "--out", "/artifacts"]

# ================= Target: runtime =================
FROM python:3.10-slim AS runtime

ARG GIT_COMMIT=unknown
LABEL org.opencontainers.image.title="kws-api" \
      org.opencontainers.image.revision="${GIT_COMMIT}"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ ./src/
# Модель береться з каталогу artifacts/ (артефакт job train), а не з етапу trainer
COPY artifacts/ ./artifacts/

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
