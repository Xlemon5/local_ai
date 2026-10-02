"""Shared HTTP, readiness, CLI and JSON reporting helpers (standard library only)."""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class CheckError(Exception):
    """A service is unavailable or its response does not satisfy the contract."""


def check(condition, message):
    if not condition:
        raise CheckError(message)


def request(base, path, payload=None, timeout=300):
    url = base.rstrip("/") + path
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as response:
            raw = response.read()
    except HTTPError as exc:
        detail = exc.read(1000).decode("utf-8", errors="replace")
        raise CheckError(f"{url}: HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise CheckError(f"{url}: {exc}") from exc
    if not raw.strip():  # TEI /health may return an empty HTTP 200 response.
        return None
    try:
        return json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise CheckError(f"{url}: ответ не является JSON") from exc


def wait_ready(base, seconds):
    deadline = time.monotonic() + seconds
    next_log = 0
    while True:
        try:
            request(base, "/health", timeout=min(10, max(0.1, deadline - time.monotonic())))
            return
        except CheckError as exc:
            if time.monotonic() >= deadline:
                raise CheckError(f"Сервис не готов за {seconds:g} с. {exc}") from exc
            if time.monotonic() >= next_log:
                print(f"  Ожидание {base}/health…", flush=True)
                next_log = time.monotonic() + 30
            time.sleep(min(2, max(0, deadline - time.monotonic())))


def positive_number(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Ожидалось положительное конечное число")
    return number


def make_parser(description, default_report, service="all"):
    parser = argparse.ArgumentParser(description=description)
    if service in ("all", "llm"):
        parser.add_argument("--llm-url", default=os.getenv("LLM_URL", "http://127.0.0.1:8000"))
        parser.add_argument("--model", default="Qwen2.5-7B-Instruct")
    if service in ("all", "embeddings"):
        parser.add_argument("--tei-url", default=os.getenv("TEI_URL", "http://127.0.0.1:8080"))
    parser.add_argument("--timeout", type=positive_number, default=300,
                        help="Таймаут одного запроса в секундах (по умолчанию 300)")
    parser.add_argument("--wait", type=positive_number, default=1200,
                        help="Ожидание готовности каждого сервиса (по умолчанию 1200 с)")
    parser.add_argument("--report", type=Path, default=Path(default_report))
    return parser


def run_checks(args, checks):
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "checks": []}
    for key in ("llm_url", "tei_url"):
        if hasattr(args, key):
            report[key] = getattr(args, key)
    print(f"Дата проверки (UTC): {report['timestamp_utc']}", flush=True)
    for name, test, describe in checks:
        print(f"\n=== {name} ===", flush=True)
        started = time.monotonic()
        try:
            details = test(args)
            result = {"name": name, "status": "passed", "details": details}
            for line in describe(details):
                print(line, flush=True)
            print("PASS: все проверки этой модели пройдены.", flush=True)
        except (CheckError, ValueError, TypeError, KeyError, IndexError) as exc:
            result = {"name": name, "status": "failed", "error": str(exc)}
            print(f"FAIL: {exc}", file=sys.stderr, flush=True)
        result["duration_seconds"] = round(time.monotonic() - started, 3)
        print(f"Длительность группы: {result['duration_seconds']:.3f} с", flush=True)
        report["checks"].append(result)
    report["passed"] = all(result["status"] == "passed" for result in report["checks"])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nОтчёт: {args.report}")
    print("Все проверки пройдены." if report["passed"] else "Есть ошибки. См. docker compose logs.")
    return 0 if report["passed"] else 1
