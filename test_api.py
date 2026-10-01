#!/usr/bin/env python3
"""Integration smoke test for Qwen + BGE-M3. Python 3.10+, no dependencies."""

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


def validate_vectors(vectors, count):
    check(isinstance(vectors, list) and len(vectors) == count,
          f"Ожидалось {count} векторов")
    for index, vector in enumerate(vectors):
        check(isinstance(vector, list) and len(vector) == 1024,
              f"Вектор {index}: размерность должна быть 1024")
        check(all(type(x) in (int, float) and math.isfinite(x) for x in vector),
              f"Вектор {index}: есть нечисловые значения, NaN или Infinity")
        norm = math.sqrt(sum(x * x for x in vector))
        check(abs(norm - 1) < 0.01,
              f"Вектор {index}: ожидалась единичная норма, получено {norm:.6f}")


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b)) / math.sqrt(
        sum(x * x for x in a) * sum(y * y for y in b))


def test_llm(args):
    wait_ready(args.llm_url, args.wait)
    models = request(args.llm_url, "/v1/models", timeout=args.timeout)
    check(isinstance(models, dict) and isinstance(models.get("data"), list),
          "Некорректный ответ /v1/models")
    check(any(isinstance(m, dict) and m.get("id") == args.model for m in models["data"]),
          f"Модель {args.model} отсутствует в /v1/models")
    response = request(args.llm_url, "/v1/chat/completions", {
        "model": args.model,
        "messages": [{"role": "user", "content":
                      "Сколько будет 6 * 7? Ответь только числом, без пояснений."}],
        "temperature": 0,
        "max_tokens": 16,
        "stream": False,
    }, timeout=args.timeout)
    check(isinstance(response, dict) and response.get("object") == "chat.completion",
          "Некорректный формат chat.completion")
    choices = response.get("choices")
    check(isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict),
          "Ожидался один ответ в choices")
    choice = choices[0]
    message = choice.get("message")
    check(isinstance(message, dict) and message.get("role") == "assistant",
          "Отсутствует сообщение assistant")
    content = message.get("content")
    check(isinstance(content, str) and content.strip(), "Пустой ответ LLM")
    check(content.strip().rstrip(".! ") == "42", f"Неверный ответ на 6 * 7: {content!r}")
    check(choice.get("finish_reason") == "stop", "Генерация не завершилась штатно")
    usage = response.get("usage")
    check(isinstance(usage, dict) and type(usage.get("completion_tokens")) is int
          and usage["completion_tokens"] > 0, "Отсутствует счётчик токенов")
    return {"model": args.model, "answer": content, "usage": usage}


def test_embeddings(args):
    wait_ready(args.tei_url, args.wait)
    info = request(args.tei_url, "/info", timeout=args.timeout)
    check(isinstance(info, dict) and info.get("model_id") == "BAAI/bge-m3",
          "TEI обслуживает не BAAI/bge-m3")
    texts = [
        "Кошка спит на диване.",
        "A cat is sleeping on the sofa.",
        "Вулкан извергает лаву и пепел.",
        "Кошка спит на диване.",
    ]
    vectors = request(args.tei_url, "/embed", {
        "inputs": texts, "normalize": True, "truncate": False,
    }, timeout=args.timeout)
    validate_vectors(vectors, len(texts))
    translated = cosine(vectors[0], vectors[1])
    unrelated = cosine(vectors[0], vectors[2])
    duplicate = cosine(vectors[0], vectors[3])
    check(duplicate > 0.999, f"Одинаковые тексты дали разные векторы: {duplicate:.4f}")
    check(translated > unrelated + 0.05,
          f"Перевод должен быть ближе постороннего текста: {translated:.4f} vs {unrelated:.4f}")

    # Also verify TEI's OpenAI-compatible embeddings endpoint and input order.
    response = request(args.tei_url, "/v1/embeddings", {
        "model": "BAAI/bge-m3", "input": texts[:2], "encoding_format": "float",
    }, timeout=args.timeout)
    check(isinstance(response, dict) and response.get("object") == "list",
          "Некорректный формат /v1/embeddings")
    data = response.get("data")
    check(isinstance(data, list) and len(data) == 2
          and all(isinstance(item, dict) for item in data), "Ожидалось два embeddings")
    check([item.get("index") for item in data] == [0, 1], "Неверный порядок индексов")
    check(all(item.get("object") == "embedding" for item in data), "Неверный тип embedding")
    compatible_vectors = [item.get("embedding") for item in data]
    validate_vectors(compatible_vectors, 2)
    for native, compatible in zip(vectors, compatible_vectors):
        check(cosine(native, compatible) > 0.999, "API /embed и /v1/embeddings расходятся")
    return {"model": info["model_id"], "dimension": 1024, "texts": texts,
            "cosine_translation": round(translated, 6),
            "cosine_unrelated": round(unrelated, 6),
            "cosine_duplicate": round(duplicate, 6), "openai_endpoint": "passed"}


def positive_number(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Ожидалось положительное конечное число")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm-url", default=os.getenv("LLM_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--tei-url", default=os.getenv("TEI_URL", "http://127.0.0.1:8080"))
    parser.add_argument("--model", default="Qwen2.5-7B-Instruct")
    parser.add_argument("--timeout", type=positive_number, default=300,
                        help="Таймаут одного запроса в секундах (по умолчанию 300)")
    parser.add_argument("--wait", type=positive_number, default=1200,
                        help="Ожидание готовности каждого сервиса (по умолчанию 1200 с)")
    parser.add_argument("--report", type=Path, default=Path("artifacts/test-report.json"))
    args = parser.parse_args()
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "llm_url": args.llm_url, "tei_url": args.tei_url, "checks": []}
    for name, test in [("Qwen /v1/chat/completions", test_llm), ("BGE-M3 embeddings", test_embeddings)]:
        print(f"Проверка: {name}", flush=True)
        started = time.monotonic()
        try:
            details = test(args)
            result = {"name": name, "status": "passed", "details": details}
            print(f"  PASS: {json.dumps(details, ensure_ascii=False)}", flush=True)
        except (CheckError, ValueError, TypeError, KeyError, IndexError) as exc:
            result = {"name": name, "status": "failed", "error": str(exc)}
            print(f"  FAIL: {exc}", file=sys.stderr, flush=True)
        result["duration_seconds"] = round(time.monotonic() - started, 3)
        report["checks"].append(result)
    report["passed"] = all(result["status"] == "passed" for result in report["checks"])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Отчёт: {args.report}")
    print("Все проверки пройдены." if report["passed"] else "Есть ошибки. См. docker compose logs.")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
