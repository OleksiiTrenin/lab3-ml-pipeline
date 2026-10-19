"""Скрипт для HEALTHCHECK: код 0, якщо /health відповідає 200, інакше 1."""
import os
import sys
import urllib.request

port = os.getenv("PORT", "8000")
try:
    urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3)
except Exception:
    sys.exit(1)
