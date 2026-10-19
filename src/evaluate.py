"""Оцінка accuracy збереженої моделі.

Локально (повний тестовий поділ, такий самий як під час навчання):
    python -m src.evaluate --data-root data --split test
У контейнері (підмножина тесту, експортована етапом trainer):
    docker exec kws python -m src.evaluate --data-root artifacts/testset --split all
Останній рядок виводу: accuracy=<відсотки>.
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.dataset import SpeechCommandsSubset, split_indices
from src.model import AudioCNN


def evaluate_model(model, loader, device="cpu"):
    """Повертає (accuracy у %, список істинних міток, список передбачень)."""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for specs, labels in loader:
            outputs = model(specs.to(device))
            y_pred += outputs.argmax(dim=1).cpu().tolist()
            y_true += labels.tolist()
    correct = sum(int(t == p) for t, p in zip(y_true, y_pred))
    return 100.0 * correct / max(len(y_true), 1), y_true, y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--split", choices=["test", "all"], default="test")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-per-class", type=int, default=None)
    args = parser.parse_args()

    art = Path(args.artifacts)
    classes = json.loads((art / "labels.json").read_text(encoding="utf-8"))
    metrics = {}
    if (art / "metrics.json").exists():
        metrics = json.loads((art / "metrics.json").read_text(encoding="utf-8"))
    # seed і max_per_class мають збігатися з навчанням, інакше тест «протече» в train
    seed = args.seed if args.seed is not None else metrics.get("seed", 42)
    max_per_class = args.max_per_class if args.max_per_class is not None else metrics.get("max_per_class")

    dataset = SpeechCommandsSubset(root=args.data_root, classes=classes, max_per_class=max_per_class)
    if args.split == "test":
        _, test_idx = split_indices(len(dataset), seed=seed)
        dataset = Subset(dataset, test_idx)
    if len(dataset) == 0:
        raise SystemExit("Немає зразків для оцінки: перевірте --data-root")

    model = AudioCNN(num_classes=len(classes))
    model.load_state_dict(torch.load(art / "model.pt", map_location="cpu", weights_only=True))

    accuracy, _, _ = evaluate_model(model, DataLoader(dataset, batch_size=64))
    print(f"samples={len(dataset)}")
    print(f"accuracy={accuracy:.2f}")


if __name__ == "__main__":
    main()
