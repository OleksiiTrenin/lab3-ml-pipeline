"""Звіт у Markdown і зведений файл метрик із результатів job verify.

    python scripts/make_report.py --metrics metrics/ --model artifacts/metrics.json \
        --out metrics/report.md

Читає (якщо є): artifacts/metrics.json (навчання), metrics/verify.json (еталонні
зразки), metrics/latency.json (benchmark). Відсутні файли не є помилкою: крок звіту
виконується з if: always(), тобто і тоді, коли перевірка впала посередині.
Поруч зі звітом пише metrics/summary.json.
"""
import argparse
import json
import os
from pathlib import Path


def load(path):
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def fmt(value, suffix=""):
    return "—" if value is None else f"{value}{suffix}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="metrics")
    parser.add_argument("--model", default="artifacts/metrics.json")
    parser.add_argument("--out", default="metrics/report.md")
    args = parser.parse_args()

    metrics_dir = Path(args.metrics)
    model = load(args.model) or {}
    verify = load(metrics_dir / "verify.json")
    latency = load(metrics_dir / "latency.json")
    server = (latency or {}).get("server_ms", {})
    min_acc = float(os.getenv("MIN_ACCURACY", "0"))
    accuracy = model.get("accuracy")

    summary = {
        "commit": os.getenv("GITHUB_SHA", "local")[:7],
        "event": os.getenv("GITHUB_EVENT_NAME", "local"),
        "run_id": os.getenv("GITHUB_RUN_ID"),
        "epochs": model.get("epochs"),
        "max_per_class": model.get("max_per_class"),
        "accuracy": accuracy,
        "min_accuracy": min_acc,
        "model_size_kb": model.get("model_size_kb"),
        "latency_ms": {k: server.get(k) for k in ("mean", "p50", "p95")},
        "golden_samples": {"passed": verify["passed"], "total": verify["total"]} if verify else None,
    }
    (metrics_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    gate_ok = accuracy is not None and accuracy >= min_acc
    golden_ok = bool(verify) and verify["passed"] == verify["total"]
    lines = [
        "## KWS model report",
        "",
        f"Commit `{summary['commit']}`, event `{summary['event']}`.",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Epochs | {fmt(model.get('epochs'))} |",
        f"| Samples per class (limit) | {fmt(model.get('max_per_class') or 'all')} |",
        f"| Train / test samples | {fmt(model.get('train_samples'))} / {fmt(model.get('test_samples'))} |",
        f"| Test accuracy | {fmt(accuracy, ' %')} (gate ≥ {min_acc:g} %: {'pass' if gate_ok else 'FAIL'}) |",
        f"| Model size | {fmt(model.get('model_size_kb'), ' KB')} |",
        f"| Parameters | {fmt(model.get('num_parameters'))} |",
        f"| Latency mean / p50 / p95 | {fmt(server.get('mean'))} / {fmt(server.get('p50'))} / {fmt(server.get('p95'))} ms |",
        f"| Golden samples | {fmt(summary['golden_samples'] and verify['passed'])} / "
        f"{fmt(summary['golden_samples'] and verify['total'])} ({'pass' if golden_ok else 'FAIL'}) |",
        "",
    ]
    if model.get("per_class_accuracy"):
        lines += ["### Per-class accuracy", "", "| Class | Accuracy, % |", "| --- | --- |"]
        lines += [f"| {c} | {a} |" for c, a in model["per_class_accuracy"].items()]
        lines += [""]
    if verify:
        lines += ["### Golden samples", "", "| File | Expected | Predicted | Confidence | Result |", "| --- | --- | --- | --- | --- |"]
        for r in verify["results"]:
            lines += [f"| {r['file']} | {r['expected']} | {fmt(r['predicted'])} | {fmt(r['confidence'])} | "
                      f"{'✅' if r['ok'] else '❌'} |"]
        lines += [""]
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
