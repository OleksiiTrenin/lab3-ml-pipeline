"""Завантаження Google Speech Commands v0.02 лише для потрібних класів.

Використовує тільки стандартну бібліотеку. Архів (~2.3 ГБ) читається потоково,
на диск розпаковуються лише wav-файли вибраних класів: <out>/<клас>/*.wav.
Окремий файл, щоб у Dockerfile його можна було скопіювати ДО решти коду і
закешувати важкий шар завантаження.
"""
import argparse
import os
import sys
import tarfile
import time
import urllib.request

URL = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"


def already_present(out, classes):
    for label in classes:
        folder = os.path.join(out, label)
        if not (os.path.isdir(folder) and any(f.endswith(".wav") for f in os.listdir(folder))):
            return False
    return True


def download(out, classes, url=URL, retries=3):
    for attempt in range(1, retries + 1):
        try:
            print(f"Завантаження {url} (спроба {attempt}/{retries})", flush=True)
            return _stream_extract(out, classes, url)
        except Exception as exc:  # мережеві збої
            print(f"Помилка: {exc}", file=sys.stderr, flush=True)
            if attempt == retries:
                raise
            time.sleep(5 * attempt)


def _stream_extract(out, classes, url):
    wanted = set(classes)
    count = 0
    with urllib.request.urlopen(url, timeout=60) as response:
        with tarfile.open(fileobj=response, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".wav"):
                    continue
                parts = member.name.replace("\\", "/").split("/")
                label = parts[-2] if len(parts) >= 2 else ""
                if label not in wanted:
                    continue
                target_dir = os.path.join(out, label)
                os.makedirs(target_dir, exist_ok=True)
                with open(os.path.join(target_dir, os.path.basename(member.name)), "wb") as f:
                    f.write(tar.extractfile(member).read())
                count += 1
                if count % 2000 == 0:
                    print(f"  розпаковано {count} файлів", flush=True)
    print(f"Готово: {count} файлів -> {out}", flush=True)
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", nargs="+", default=["yes", "no", "up", "down"])
    parser.add_argument("--out", default="data")
    parser.add_argument("--url", default=URL)
    args = parser.parse_args()

    if already_present(args.out, args.classes):
        print(f"Датасет уже є в {args.out}, пропускаю завантаження")
        return
    download(args.out, args.classes, args.url)


if __name__ == "__main__":
    main()
