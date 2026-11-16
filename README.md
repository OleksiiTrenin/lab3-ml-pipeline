# ЛР №3 — CI/CD-пайплайн ML-сервісу в GitHub Actions

Аудіокласифікатор команд `yes / no / up / down` (Google Speech Commands, CNN з ЛР №1)
і FastAPI-сервіс з ЛР №2 (`/health`, `/model-info`, `/predict`). Пайплайн
[`ml-pipeline.yml`](.github/workflows/ml-pipeline.yml) замінює `ci.yml` з ЛР №2:

```
pull_request / push / workflow_dispatch
        │
        ▼
 lint ──► train ──► verify ──► build-push
            │          │           │
            ▼          ▼           ▼
       артефакт    артефакт    образ у GHCR
        model       report     (push у main або ручний запуск з push_image)
```

| Job | Що робить | Результат |
| --- | --- | --- |
| `lint` | `ruff check src/ scripts/` | — |
| `train` | збирає ціль `trainer`, навчає модель у контейнері (`docker run`) | артефакт `model`: model.pt, labels.json, metrics.json, train.log |
| `verify` | збирає ціль `runtime` з моделлю, піднімає сервіс, еталонні зразки, quality gate, latency | артефакт `report`: verify.json, latency.json, summary.json, report.md, service.log; Job Summary |
| `build-push` | публікує образ `ghcr.io/<owner>/<repo>/kws-api:sha-<commit>` і `:latest` | образ у GHCR |

| Тригер | Епох | Записів на клас | Поріг accuracy | Публікація |
| --- | --- | --- | --- | --- |
| `pull_request` у `main` | 3 | 2000 | 60 % | ні |
| `push` у `main` | 10 | усі | 80 % | так |
| `workflow_dispatch` | з форми | усі | 80 % | якщо `push_image` |

## Структура

```
.github/workflows/ml-pipeline.yml
src/        audio.py (log-Mel), dataset.py, model.py, train.py, evaluate.py, download.py, api.py, healthcheck.py
scripts/    benchmark.py (latency → JSON), verify_model.py (еталонні зразки), make_report.py (Markdown-звіт)
samples/    yes.wav, no.wav, up.wav, down.wav, expected.json, sources.json
Dockerfile  цілі trainer і runtime
```

Еталонні зразки взято з офіційного тестового набору Speech Commands v0.02
(`samples/sources.json`). Під час навчання записи цих дикторів виключаються з вибірки.

## Локально

```bash
docker build --target trainer -t kws-trainer .
mkdir -p artifacts
docker run --rm -v "$PWD/artifacts:/artifacts" kws-trainer \
  --classes yes no up down --epochs 10 --seed 42 --out /artifacts

docker build --target runtime -t kws-api .
docker run -d --name kws -p 8000:8000 kws-api
python scripts/verify_model.py --url http://localhost:8000 --expected samples/expected.json
```

Готовий образ: `docker pull ghcr.io/oleksiitrenin/lab3-ml-pipeline/kws-api:latest`.
