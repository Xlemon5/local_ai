#!/usr/bin/env python3
"""Отдельная проверка BGE-M3 через TEI и сравнение смысла текстов."""

import math
import sys
from api_utils import check, make_parser, request, run_checks, wait_ready


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
        "На диване спит кот.",
    ]
    vectors = request(args.tei_url, "/embed", {
        "inputs": texts, "normalize": True, "truncate": False,
    }, timeout=args.timeout)
    validate_vectors(vectors, len(texts))
    translated = cosine(vectors[0], vectors[1])
    unrelated = cosine(vectors[0], vectors[2])
    duplicate = cosine(vectors[0], vectors[3])
    paraphrase = cosine(vectors[0], vectors[4])
    check(duplicate > 0.999, f"Одинаковые тексты дали разные векторы: {duplicate:.4f}")
    check(translated > unrelated + 0.05,
          f"Перевод должен быть ближе постороннего текста: {translated:.4f} vs {unrelated:.4f}")
    check(paraphrase > unrelated + 0.05,
          f"Перефразирование должно быть ближе постороннего текста: {paraphrase:.4f} vs {unrelated:.4f}")

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
            "count": len(vectors), "vector_preview": vectors[0][:5],
            "norms": [round(math.sqrt(sum(x * x for x in vector)), 6) for vector in vectors],
            "cosine_translation": round(translated, 6),
            "cosine_unrelated": round(unrelated, 6),
            "cosine_duplicate": round(duplicate, 6),
            "cosine_paraphrase": round(paraphrase, 6), "openai_endpoint": "passed"}


def describe_embeddings(details):
    lines = [
        f"Модель: {details['model']}",
        "API: /embed и /v1/embeddings",
        f"Получено векторов: {details['count']}",
        f"Размерность каждого вектора: {details['dimension']}",
        "Первые 5 чисел первого вектора: " + ", ".join(f"{x:.6f}" for x in details["vector_preview"]),
        "Нормы векторов: " + ", ".join(f"{x:.6f}" for x in details["norms"]),
        "\nСравнение смысла текстов:",
        f"Базовая фраза: {details['texts'][0]}",
    ]
    for index, key, label in [
        (4, "cosine_paraphrase", "Перефразирование"),
        (1, "cosine_translation", "Английский перевод"),
        (2, "cosine_unrelated", "Другая тема"),
        (3, "cosine_duplicate", "Точная копия"),
    ]:
        lines.append(f"  {details[key]:.6f} | {label}: {details['texts'][index]}")
    lines.extend([
        "\nПроверено: конечные числа, 1024 измерения, единичная норма.",
        "Перевод и перефразирование ближе, чем текст на другую тему.",
        "Одинаковые тексты и результаты двух API согласованы.",
    ])
    return lines


if __name__ == "__main__":
    parser = make_parser(__doc__, "artifacts/embeddings-test-report.json", service="embeddings")
    sys.exit(run_checks(parser.parse_args(), [("BGE-M3 embeddings", test_embeddings, describe_embeddings)]))
