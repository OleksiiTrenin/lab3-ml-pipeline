"""Вимірювання latency POST /predict (лише стандартна бібліотека).

    python scripts/benchmark.py --url http://localhost:8000 --wav samples/yes.wav -n 100

Спочатку 10 «розігрівних» запитів (не враховуються), потім N вимірюваних.
Виводить server latency (поле latency_ms з відповіді) та end-to-end latency
клієнта (HTTP-запит від надсилання до отримання відповіді, keep-alive з'єднання).
Код виходу 1, якщо задано --max-p95-ms і p95 (сервер) його перевищує.
"""
import argparse
import http.client
import json
import os
import socket
import statistics
import sys
import time
import uuid
from urllib.parse import urlparse


def build_multipart(wav_path):
    boundary = uuid.uuid4().hex
    with open(wav_path, "rb") as f:
        data = f.read()
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{os.path.basename(wav_path)}"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode()
    body = head + data + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def summarize(values):
    ordered = sorted(values)
    p95 = statistics.quantiles(ordered, n=100, method="inclusive")[94] if len(ordered) > 1 else ordered[0]
    return {
        "mean": round(statistics.fmean(ordered), 3),
        "p50": round(statistics.median(ordered), 3),
        "p95": round(p95, 3),
    }


def run_benchmark(url, wav_path, n=100, warmup=10):
    parsed = urlparse(url)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=30)
    conn.connect()
    conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # без затримок Nagle
    body, content_type = build_multipart(wav_path)
    headers = {"Content-Type": content_type, "Content-Length": str(len(body))}

    server_ms, client_ms = [], []
    for i in range(warmup + n):
        t0 = time.perf_counter()
        conn.request("POST", "/predict", body=body, headers=headers)
        response = conn.getresponse()
        payload = response.read()
        elapsed = (time.perf_counter() - t0) * 1000.0
        if response.status != 200:
            raise RuntimeError(f"/predict повернув {response.status}: {payload[:200]!r}")
        if i >= warmup:
            server_ms.append(json.loads(payload)["latency_ms"])
            client_ms.append(elapsed)
    conn.close()
    return {"n": n, "warmup": warmup, "server_ms": summarize(server_ms), "client_ms": summarize(client_ms)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--wav", default="samples/yes.wav")
    parser.add_argument("-n", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--out", help="зберегти результат у JSON")
    parser.add_argument("--max-p95-ms", type=float, help="quality gate: p95 (сервер) не вище цього значення")
    args = parser.parse_args()

    result = run_benchmark(args.url, args.wav, args.n, args.warmup)
    print(f"Запитів: {args.n} (+{args.warmup} розігрівних)")
    for key, title in (("server_ms", "server latency"), ("client_ms", "end-to-end (клієнт)")):
        s = result[key]
        print(f"{title:22s} mean={s['mean']:.2f} мс  p50={s['p50']:.2f} мс  p95={s['p95']:.2f} мс")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    if args.max_p95_ms is not None and result["server_ms"]["p95"] > args.max_p95_ms:
        print(f"FAIL: p95 {result['server_ms']['p95']} мс > ліміт {args.max_p95_ms} мс")
        sys.exit(1)


if __name__ == "__main__":
    main()
