#!/usr/bin/env python3
"""Общая проверка Qwen и BGE-M3; отдельные тесты: TextTest.py и EmbeddingsTest.py."""

import sys
from api_utils import make_parser, run_checks
from TextTest import describe_llm, test_llm
from EmbeddingsTest import describe_embeddings, test_embeddings


def main():
    parser = make_parser(__doc__, "artifacts/test-report.json")
    return run_checks(parser.parse_args(), [
        ("Qwen /v1/chat/completions", test_llm, describe_llm),
        ("BGE-M3 embeddings", test_embeddings, describe_embeddings),
    ])


if __name__ == "__main__":
    sys.exit(main())
