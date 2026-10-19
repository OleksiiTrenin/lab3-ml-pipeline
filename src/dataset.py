import glob
import os

import torch
from torch.utils.data import Dataset

from src.audio import load_spectrogram

LABELS = ["yes", "no", "up", "down"]


class SpeechCommandsSubset(Dataset):
    """Підмножина Google Speech Commands: файли <root>/**/<клас>/*.wav."""

    def __init__(self, root="./data", classes=LABELS, max_per_class=None):
        self.classes = list(classes)
        class_to_idx = {c: i for i, c in enumerate(self.classes)}

        per_class = {c: [] for c in self.classes}
        pattern = os.path.join(root, "**", "*.wav")
        for path in glob.glob(pattern, recursive=True):
            label = os.path.basename(os.path.dirname(path))
            if label in per_class:
                per_class[label].append(path)

        self.items = []
        for label in self.classes:
            paths = sorted(per_class[label])  # sorted -> однаковий порядок на будь-якій машині
            if max_per_class:
                paths = paths[:max_per_class]
            self.items += [(p, class_to_idx[label]) for p in paths]

        print(f"Знайдено зразків для вибраних класів: {len(self.items)}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, target = self.items[idx]
        return load_spectrogram(path), target


def split_indices(n, test_ratio=0.2, seed=42):
    """Детермінований поділ train/test. Однакова функція у train.py та evaluate.py."""
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator).tolist()
    n_test = int(n * test_ratio)
    return perm[n_test:], perm[:n_test]
