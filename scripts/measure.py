"""Автоматичне збирання метрик для таблиці п. 2.8 (працює на Windows/Linux/macOS).

Порядок роботи:
    docker build -t kws-api:1.0.0 .
    docker build -f Dockerfile.single -t kws-api:1.0.0-single .
    python scripts/measure.py export --image kws-api:1.0.0          # ті самі ваги моделі -> ./artifacts
    python scripts/measure.py local  --label "Локально (venv)"       # з активованого venv
    python scripts/measure.py docker --image kws-api:1.0.0-single --label "Docker (single-stage)"
    python scripts/measure.py docker --image kws-api:1.0.0 --label "Docker (multi-stage)"
    python scripts/measure.py table                                  # зведена таблиця (Markdown)

Результати: results/*.json, results/*.history.txt, results/summary.md
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from benchmark import run_benchmark

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CONTAINER = "kws-bench"


def run(cmd, check=True, **kwargs):
    return subprocess.run(cmd, check=check, capture_output=True, text=True, **kwargs)


def get_json(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read())


def wait_health(port, started, timeout=180):
    """Секунди від моменту started до першої успішної відповіді /health."""
    while time.perf_counter() - started < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    return round(time.perf_counter() - started, 2)
        except Exception:
            time.sleep(0.05)
    raise TimeoutError("сервіс не став готовим за відведений час")


def parse_accuracy(text):
    matches = re.findall(r"accuracy=([\d.]+)", text)
    if not matches:
        raise RuntimeError(f"не знайдено accuracy у виводі evaluate.py:\n{text}")
    return float(matches[-1])


def dir_size_mb(path):
    total = 0
    for dirpath, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                pass
    return total / 1e6


def parse_mem(text):
    """'123.4MiB / 7.6GiB' -> МБ (MiB)."""
    value, unit = re.match(r"([\d.]+)\s*([kKMG]?i?B)", text.strip()).groups()
    factor = {"B": 1 / 1048576, "KiB": 1 / 1024, "MiB": 1, "GiB": 1024, "kB": 1 / 1000, "MB": 1.048576, "GB": 1048.576}
    return float(value) * factor.get(unit, 1)


def slug(label):
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "result"


def save(label, data):
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{slug(label)}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Збережено {path}")


def cmd_export(args):
    run(["docker", "rm", "-f", "kws-export"], check=False)
    run(["docker", "create", "--name", "kws-export", args.image])
    try:
        target = ROOT / "artifacts"
        target.mkdir(exist_ok=True)
        run(["docker", "cp", "kws-export:/app/artifacts/.", str(target)])
    finally:
        run(["docker", "rm", "-f", "kws-export"], check=False)
    print(f"Артефакти з образу {args.image} скопійовано в {ROOT / 'artifacts'}")


def cmd_local(args):
    import psutil  # лише на хості

    port = args.port
    env = {**os.environ, "MODEL_PATH": str(ROOT / "artifacts" / "model.pt"), "PORT": str(port)}
    venv = Path(args.venv) if args.venv else Path(sys.prefix)
    wav = Path(args.wav) if args.wav else ROOT / "artifacts" / "samples" / "yes.wav"

    started = time.perf_counter()
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        cold = wait_health(port, started)
        info = get_json(f"http://127.0.0.1:{port}/model-info")
        bench = run_benchmark(f"http://127.0.0.1:{port}", str(wav), args.n, args.warmup)
        p = psutil.Process(proc.pid)
        ram = (p.memory_info().rss + sum(c.memory_info().rss for c in p.children(recursive=True))) / 1048576
        evaluation = run(
            [sys.executable, "-m", "src.evaluate", "--artifacts", str(ROOT / "artifacts"),
             "--data-root", args.eval_root, "--split", args.eval_split],
            cwd=ROOT,
        )
        accuracy = parse_accuracy(evaluation.stdout)
    finally:
        proc.terminate()
        proc.wait(timeout=10)

    env_mb = dir_size_mb(venv) + info["model_size_kb"] / 1024 * 1.048576
    save(args.label, {
        "label": args.label, "kind": "local", "model": "CNN", "model_size_kb": info["model_size_kb"],
        "accuracy_eval": accuracy, "accuracy_metrics_json": info["accuracy"],
        "server_ms": bench["server_ms"], "client_ms": bench["client_ms"],
        "size_mb": round(env_mb, 1), "ram_mb": round(ram, 1), "cold_start_s": cold,
        "n": args.n, "warmup": args.warmup,
    })


def cmd_docker(args):
    port = args.port
    run(["docker", "rm", "-f", CONTAINER], check=False)
    size_mb = int(run(["docker", "image", "inspect", "-f", "{{.Size}}", args.image]).stdout) / 1e6  # як у docker images (SI)
    history = run(["docker", "history", "--no-trunc", "--format", "table {{.Size}}\t{{.CreatedBy}}", args.image]).stdout
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{slug(args.label)}.history.txt").write_text(history, encoding="utf-8")

    started = time.perf_counter()
    run(["docker", "run", "-d", "--name", CONTAINER, "-p", f"{port}:8000", args.image])
    try:
        cold = wait_health(port, started)
        info = get_json(f"http://127.0.0.1:{port}/model-info")
        wav = Path(args.wav) if args.wav else RESULTS / "_yes.wav"
        if not args.wav:
            run(["docker", "cp", f"{CONTAINER}:/app/artifacts/samples/yes.wav", str(wav)])
        bench = run_benchmark(f"http://127.0.0.1:{port}", str(wav), args.n, args.warmup)
        stats = run(["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", CONTAINER]).stdout
        ram = parse_mem(stats.split("/")[0])
        evaluation = run(["docker", "exec", CONTAINER, "python", "-m", "src.evaluate",
                          "--data-root", "artifacts/testset", "--split", "all"])
        accuracy = parse_accuracy(evaluation.stdout)
        health = run(["docker", "inspect", "-f", "{{if .State.Health}}{{.State.Health.Status}}{{end}}", CONTAINER]).stdout.strip()
    finally:
        run(["docker", "rm", "-f", CONTAINER], check=False)

    save(args.label, {
        "label": args.label, "kind": "docker", "image": args.image, "model": "CNN",
        "model_size_kb": info["model_size_kb"], "accuracy_eval": accuracy, "accuracy_metrics_json": info["accuracy"],
        "server_ms": bench["server_ms"], "client_ms": bench["client_ms"],
        "size_mb": round(size_mb, 1), "ram_mb": round(ram, 1), "cold_start_s": cold,
        "health_status_at_end": health, "n": args.n, "warmup": args.warmup,
    })


def cmd_table(_args):
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RESULTS.glob("*.json"))]
    rows.sort(key=lambda r: 0 if r["kind"] == "local" else (1 if "single" in r["label"].lower() else 2))
    lines = [
        "| Середовище | Модель | Розмір моделі, КБ | Accuracy, % | Latency (сервер) mean / p95, мс | "
        "Latency (клієнт) mean / p95, мс | Розмір образу / середовища, МБ | RAM, МБ | Холодний старт, с |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        s, c = r["server_ms"], r["client_ms"]
        lines.append(
            f"| {r['label']} | {r['model']} | {r['model_size_kb']} | {r['accuracy_eval']:.2f} | "
            f"{s['mean']:.2f} / {s['p95']:.2f} | {c['mean']:.2f} / {c['p95']:.2f} | "
            f"{r['size_mb']} | {r['ram_mb']} | {r['cold_start_s']} |"
        )
    text = "\n".join(lines)
    print(text)
    (RESULTS / "summary.md").write_text(text + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("export", help="скопіювати artifacts/ з образу на хост")
    p.add_argument("--image", required=True)
    p.set_defaults(func=cmd_export)

    for name, func in (("local", cmd_local), ("docker", cmd_docker)):
        p = sub.add_parser(name)
        p.add_argument("--label", required=True)
        p.add_argument("--wav")
        p.add_argument("-n", type=int, default=100)
        p.add_argument("--warmup", type=int, default=10)
        p.add_argument("--port", type=int, default=8000)
        if name == "docker":
            p.add_argument("--image", required=True)
        else:
            p.add_argument("--venv", help="папка venv для виміру розміру (за замовчуванням sys.prefix)")
            p.add_argument("--eval-root", default="artifacts/testset")
            p.add_argument("--eval-split", default="all")
        p.set_defaults(func=func)

    p = sub.add_parser("table", help="зібрати таблицю з results/*.json")
    p.set_defaults(func=cmd_table)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
