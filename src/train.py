"""Навчання аудіокласифікатора + оцінка -> artifacts/{model.pt, labels.json, metrics.json}.

Запускається і локально, і на етапі trainer у Dockerfile:
    python -m src.train --classes yes no up down --epochs 10 --seed 42 --out artifacts
"""
import argparse
import json
import os
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.dataset import SpeechCommandsSubset, split_indices
from src.evaluate import evaluate_model
from src.model import AudioCNN


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def export_test_files(dataset, test_idx, classes, out, per_class):
    """Копіює невеликий тестовий піднабір у artifacts/testset і по одному прикладу у artifacts/samples.

    Потрібно, щоб evaluate.py і curl-перевірки працювали всередині контейнера,
    де немає повного датасету.
    """
    testset_dir = Path(out) / "testset"
    samples_dir = Path(out) / "samples"
    counters = {c: 0 for c in classes}
    for idx in test_idx:
        path, target = dataset.items[idx]
        label = classes[target]
        if counters[label] >= per_class:
            continue
        (testset_dir / label).mkdir(parents=True, exist_ok=True)
        shutil.copy(path, testset_dir / label / os.path.basename(path))
        if counters[label] == 0:
            samples_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(path, samples_dir / f"{label}.wav")
        counters[label] += 1
    return counters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", nargs="+", default=["yes", "no", "up", "down"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--max-per-class", type=int, default=0, help="0 = без обмеження (для швидкого CI)")
    parser.add_argument("--version", default=os.getenv("MODEL_VERSION", "1.0.0"))
    parser.add_argument("--export-test-per-class", type=int, default=25)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Пристрій: {device}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    max_per_class = args.max_per_class or None
    dataset = SpeechCommandsSubset(root=args.data_root, classes=args.classes, max_per_class=max_per_class)
    if len(dataset) == 0:
        sys.exit(f"У {args.data_root} не знайдено wav-файлів для класів {args.classes}")

    train_idx, test_idx = split_indices(len(dataset), test_ratio=0.2, seed=args.seed)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False)
    print(f"Навчання: {len(train_idx)} зразків, тест: {len(test_idx)} зразків")

    model = AudioCNN(num_classes=len(args.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for specs, labels in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", disable=not sys.stderr.isatty()):
            specs, labels = specs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * specs.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        print(f"Епоха [{epoch}/{args.epochs}] loss={running_loss / total:.4f} train_acc={100 * correct / total:.2f}%", flush=True)

    accuracy, y_true, y_pred = evaluate_model(model, test_loader, device)
    print(f"Test accuracy: {accuracy:.2f}%")

    model = model.cpu()
    model_path = out / "model.pt"
    torch.save(model.state_dict(), model_path)
    (out / "labels.json").write_text(json.dumps(args.classes, ensure_ascii=False), encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(args.classes))))
    per_class = {
        c: round(100.0 * cm[i, i] / max(cm[i].sum(), 1), 2) for i, c in enumerate(args.classes)
    }
    n_params = sum(p.numel() for p in model.parameters())
    metrics = {
        "model_version": args.version,
        "architecture": "AudioCNN",
        "classes": args.classes,
        "accuracy": round(accuracy, 2),  # у відсотках
        "per_class_accuracy": per_class,
        "confusion_matrix": cm.tolist(),
        "train_samples": len(train_idx),
        "test_samples": len(test_idx),
        "epochs": args.epochs,
        "seed": args.seed,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "max_per_class": max_per_class,
        "num_parameters": n_params,
        "model_size_kb": round(os.path.getsize(model_path) / 1024, 2),
        "torch_version": torch.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    exported = export_test_files(dataset, test_idx, args.classes, out, args.export_test_per_class)
    print(f"Експортовано тестових файлів: {exported}")
    print(f"Артефакти збережено в {out}/: model.pt ({metrics['model_size_kb']} КБ), labels.json, metrics.json")


if __name__ == "__main__":
    main()
