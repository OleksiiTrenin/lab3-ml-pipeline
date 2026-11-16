"""Спільний препроцесинг аудіо для навчання та інференсу.

Один і той самий код використовують і src/dataset.py (навчання), і src/api.py
(сервіс), тому перетворення гарантовано однакові (немає train/serve skew).
Параметри збігаються з ЛР №1: mono, 16 кГц, рівно 1 с, Mel-спектрограма 64 смуги.
ЛР №3: спектрограма логарифмується (log-Mel). Без логарифма енергія мовлення
відрізняється на порядки між записами, і мала CNN навчалася до ~60 % accuracy.
"""
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.transforms as T

SAMPLE_RATE = 16000
NUM_SAMPLES = 16000  # 1 секунда

_mel = T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64)
_resamplers = {}


def decode(source):
    """Читає wav (шлях або file-like) -> (waveform [1, N], sample_rate). Моно усереднюється."""
    data, sample_rate = sf.read(source, dtype="float32", always_2d=True)  # (N, channels)
    if data.shape[0] == 0:
        raise ValueError("аудіофайл порожній")
    waveform = torch.from_numpy(np.ascontiguousarray(data.T)).mean(dim=0, keepdim=True)
    return waveform, sample_rate


def preprocess(waveform, sample_rate):
    """Ресемплінг до 16 кГц, доповнення/обрізання до 1 с, log-Mel-спектрограма [1, 64, 32]."""
    if sample_rate != SAMPLE_RATE:
        resampler = _resamplers.get(sample_rate)
        if resampler is None:
            resampler = T.Resample(sample_rate, SAMPLE_RATE)
            _resamplers[sample_rate] = resampler
        waveform = resampler(waveform)

    if waveform.shape[1] < NUM_SAMPLES:
        waveform = F.pad(waveform, (0, NUM_SAMPLES - waveform.shape[1]))
    else:
        waveform = waveform[:, :NUM_SAMPLES]

    return torch.log(_mel(waveform) + 1e-6)


def load_spectrogram(source):
    waveform, sample_rate = decode(source)
    return preprocess(waveform, sample_rate)
