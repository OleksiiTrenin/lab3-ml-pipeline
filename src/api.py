"""FastAPI-сервіс інференсу: GET /health, GET /model-info, POST /predict.

Конфігурація тільки через змінні середовища:
    MODEL_PATH      шлях до model.pt (labels.json і metrics.json шукаються поруч)
    LOG_LEVEL       рівень логування (INFO за замовчуванням)
    MAX_UPLOAD_MB   максимальний розмір файлу (5 МБ)
    TORCH_THREADS   кількість потоків PyTorch (за замовчуванням — як вирішить torch)
    MODEL_VERSION   версія, якщо у metrics.json її немає
"""
import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile

from src.audio import decode, preprocess
from src.model import AudioCNN

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}',
)
log = logging.getLogger("kws-api")

MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/model.pt"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "5")) * 1024 * 1024

state = {}


def load_artifacts():
    """Завантаження моделі — один раз під час старту сервісу, а не на кожен запит."""
    labels_path = Path(os.getenv("LABELS_PATH", MODEL_PATH.parent / "labels.json"))
    metrics_path = Path(os.getenv("METRICS_PATH", MODEL_PATH.parent / "metrics.json"))

    if os.getenv("TORCH_THREADS"):
        torch.set_num_threads(int(os.environ["TORCH_THREADS"]))

    classes = json.loads(labels_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}

    model = AudioCNN(num_classes=len(classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()

    with torch.inference_mode():  # прогрів: перший виклик torch повільніший
        model(torch.zeros(1, 1, 64, 32))

    state.update(
        model=model,
        classes=classes,
        metrics=metrics,
        version=metrics.get("model_version", os.getenv("MODEL_VERSION", "unknown")),
        size_kb=round(MODEL_PATH.stat().st_size / 1024, 2),
        num_parameters=sum(p.numel() for p in model.parameters()),
    )
    log.info("model loaded path=%s version=%s classes=%s", MODEL_PATH, state["version"], classes)


@asynccontextmanager
async def lifespan(_app):
    load_artifacts()
    yield


app = FastAPI(title="Audio Classification API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    if "model" not in state:
        raise HTTPException(status_code=503, detail="модель ще не завантажена")
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {
        "model_version": state["version"],
        "architecture": "AudioCNN: 2 x [Conv2d 3x3 + BatchNorm + ReLU] -> AdaptiveAvgPool(4x4) -> FC(512,64) -> FC(64,N)",
        "input": "Mel-спектрограма 1x64x32 (16 кГц, 1 с, n_fft=1024, hop=512, n_mels=64)",
        "classes": state["classes"],
        "num_parameters": state["num_parameters"],
        "model_size_kb": state["size_kb"],
        "accuracy": state["metrics"].get("accuracy"),  # у відсотках, з metrics.json
        "torch_version": torch.__version__,
    }


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    raw = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not raw:
        raise HTTPException(status_code=400, detail="Порожній файл")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Файл більший за {MAX_UPLOAD_BYTES // 1024 // 1024} МБ")

    start = time.perf_counter()  # latency: лише препроцесинг + інференс
    try:
        waveform, sample_rate = decode(io.BytesIO(raw))
    except Exception as exc:  # некоректний/не-аудіо файл -> 400, а не 500
        raise HTTPException(status_code=400, detail=f"Не вдалося прочитати аудіо (очікується wav): {exc}") from exc

    spec = preprocess(waveform, sample_rate).unsqueeze(0)  # [1, 1, 64, 32]
    with torch.inference_mode():
        probs = torch.softmax(state["model"](spec), dim=1)[0]
    latency_ms = (time.perf_counter() - start) * 1000.0

    pred_idx = int(probs.argmax())
    classes = state["classes"]
    result = {
        "label": classes[pred_idx],
        "confidence": round(float(probs[pred_idx]), 4),
        "probabilities": {c: round(float(p), 4) for c, p in zip(classes, probs)},
        "latency_ms": round(latency_ms, 2),
        "model_version": state["version"],
    }
    log.info("predict label=%s confidence=%.4f latency_ms=%.2f", result["label"], result["confidence"], latency_ms)
    return result
