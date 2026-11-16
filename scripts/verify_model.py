"""Перевірка моделі на стабільних еталонних зразках (golden samples).

    python scripts/verify_model.py --url http://localhost:8000 \
        --expected samples/expected.json --out metrics/verify.json

Кожен файл з expected.json надсилається на POST /predict, отриманий клас
порівнюється з очікуваним. Код виходу 1, якщо хоча б один зразок розпізнано
неправильно або сервіс повернув помилку.
"""
import argparse
import json
import sys
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--expected", default="samples/expected.json")
    parser.add_argument("--out", default="metrics/verify.json")
    args = parser.parse_args()

    expected_path = Path(args.expected)
    expected = json.loads(expected_path.read_text(encoding="utf-8"))

    results = []
    for name, label in expected.items():
        wav = expected_path.parent / name
        row = {"file": name, "expected": label, "predicted": None, "confidence": None, "ok": False}
        try:
            with open(wav, "rb") as f:
                response = requests.post(f"{args.url}/predict", files={"file": (name, f, "audio/wav")}, timeout=30)
            response.raise_for_status()
            body = response.json()
            row.update(predicted=body["label"], confidence=body["confidence"], ok=body["label"] == label)
        except Exception as exc:  # недоступний файл, 4xx/5xx, некоректна відповідь
            row["error"] = str(exc)
        results.append(row)
        mark = "OK  " if row["ok"] else "FAIL"
        print(f"{mark} {name:10s} expected={label:5s} predicted={row['predicted']} confidence={row['confidence']}")

    passed = sum(r["ok"] for r in results)
    summary = {"passed": passed, "total": len(results), "results": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"golden samples: {passed}/{len(results)}")
    if passed != len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
