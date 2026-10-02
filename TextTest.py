#!/usr/bin/env python3
"""Отдельная проверка Qwen2.5-7B-Instruct через llama.cpp."""

import sys
from api_utils import check, make_parser, request, run_checks, wait_ready


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
    return {"model": args.model, "answer": content, "usage": usage,
            "finish_reason": choice["finish_reason"]}


def describe_llm(details):
    return [
        f"Модель: {details['model']}",
        "API: /v1/models и /v1/chat/completions",
        "Вопрос: Сколько будет 6 * 7? Ответь только числом, без пояснений.",
        f"Ответ модели: {details['answer'].strip()}",
        "Ожидаемый ответ: 42",
        f"Завершение генерации: {details['finish_reason']}",
        f"Токены: вход {details['usage']['prompt_tokens']}, "
        f"ответ {details['usage']['completion_tokens']}",
    ]


if __name__ == "__main__":
    parser = make_parser(__doc__, "artifacts/text-test-report.json", service="llm")
    sys.exit(run_checks(parser.parse_args(), [("Qwen /v1/chat/completions", test_llm, describe_llm)]))
