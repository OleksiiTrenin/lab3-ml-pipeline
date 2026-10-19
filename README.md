# ЛР №2 — Контейнеризація ML-моделі (аудіокласифікатор yes / no / up / down)

FastAPI-сервіс інференсу CNN з ЛР №1, упакований у Docker (multi-stage build).

## Структура

```
lab2/
├── src/            audio.py (препроцесинг), dataset.py, model.py, train.py, evaluate.py,
│                   download.py (датасет), api.py (FastAPI), healthcheck.py
├── scripts/        benchmark.py (latency), measure.py (збирання метрик для таблиці)
├── .github/workflows/ci.yml
├── Dockerfile          multi-stage: trainer -> runtime
├── Dockerfile.single   single-stage, лише для порівняння розмірів
├── requirements-train.txt / requirements-api.txt / requirements-dev.txt
└── .dockerignore
```

`artifacts/` (model.pt, labels.json, metrics.json, samples/, testset/) генерується навчанням.

## Збирання і запуск

```bash
docker build -t kws-api:1.0.0 .                       # навчання відбувається під час збирання (етап trainer)
docker run -d --name kws -p 8000:8000 kws-api:1.0.0
docker ps                                             # статус має стати "healthy"
curl http://localhost:8000/health
curl http://localhost:8000/model-info
docker cp kws:/app/artifacts/samples/yes.wav .        # приклад для перевірки
curl -X POST -F "file=@yes.wav" http://localhost:8000/predict
```

Swagger UI: http://localhost:8000/docs

Параметри збирання: `--build-arg EPOCHS=10 --build-arg MAX_PER_CLASS=0 --build-arg MODEL_VERSION=1.0.0`.
Змінні середовища сервісу: `MODEL_PATH`, `PORT`, `LOG_LEVEL`, `MAX_UPLOAD_MB`, `TORCH_THREADS`.

## Локальний запуск (без Docker) для порівняння

```bash
python -m venv .venv && .venv\Scripts\activate        # Linux/macOS: source .venv/bin/activate
pip install -r requirements-train.txt -r requirements-dev.txt
python scripts/measure.py export --image kws-api:1.0.0   # ті самі ваги, що й у контейнері
```

(Альтернатива — навчити локально: `python src/download.py --out data && python -m src.train --data-root data`.)

## Вимірювання (п. 2.8)

```bash
docker build -f Dockerfile.single -t kws-api:1.0.0-single .
python scripts/measure.py local  --label "Локально (venv)"
python scripts/measure.py docker --image kws-api:1.0.0-single --label "Docker (single-stage)"
python scripts/measure.py docker --image kws-api:1.0.0       --label "Docker (multi-stage)"
python scripts/measure.py table                           # results/summary.md
docker history kws-api:1.0.0-single
docker history kws-api:1.0.0
```

Для чесного порівняння локальний запуск і контейнер використовують ті самі ваги й той самий тестовий піднабір.
Повна accuracy на всьому тестовому поділі лежить у `metrics.json` (поле `accuracy` у `/model-info`).

## CI

`.github/workflows/ci.yml`: lint (ruff) -> build -> run -> /health -> smoke test (+ перевірка коду 400 для
некоректного файлу) -> quality gate по accuracy -> evaluate у контейнері -> quality gate по p95 latency.
Поріг accuracy задається змінною `ACC_THRESHOLD` (80 %) або вручну під час `workflow_dispatch`.
