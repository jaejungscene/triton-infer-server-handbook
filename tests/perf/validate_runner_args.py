#!/usr/bin/env python3
"""Validate perf runner arguments and render normalized HTTP endpoints."""

from __future__ import annotations

import argparse
import re
from urllib.parse import urlsplit


_SAFE_MODEL = re.compile(r"[A-Za-z0-9._-]+")
_CONCURRENCY = re.compile(
    r"([1-9][0-9]*)(?::([1-9][0-9]*)(?::([1-9][0-9]*))?)?"
)


def validate_model(model: str) -> None:
    if model and not _SAFE_MODEL.fullmatch(model):
        raise ValueError("model must contain only letters, numbers, dot, underscore, or hyphen")


def validate_concurrency(value: str) -> None:
    match = _CONCURRENCY.fullmatch(value)
    if match is None:
        raise ValueError("concurrency must use start[:end[:step]] positive integer syntax")
    start = int(match.group(1))
    end = int(match.group(2)) if match.group(2) is not None else start
    step = int(match.group(3)) if match.group(3) is not None else 1
    if start > end:
        raise ValueError("concurrency start must not exceed end")
    if max(start, end, step) > 10_000:
        raise ValueError("concurrency values must not exceed 10000")


def normalize_endpoint(value: str) -> tuple[str, str, bool]:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError("url must be an HTTP(S) origin with an explicit port")
    has_scheme = "://" in value
    parsed = urlsplit(value if has_scheme else f"//{value}")
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("url contains an invalid port") from error
    if (
        (has_scheme and parsed.scheme not in {"http", "https"})
        or (not has_scheme and parsed.scheme)
        or not parsed.hostname
        or port is None
        or port <= 0
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("url must be an HTTP(S) origin with an explicit port")

    scheme = parsed.scheme if has_scheme else "http"
    origin = f"{scheme}://{parsed.netloc}".rstrip("/")
    perf_url = origin if scheme == "https" else parsed.netloc
    return origin, perf_url, scheme == "https"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="")
    parser.add_argument("--concurrency", required=True)
    parser.add_argument("--url", required=True)
    args = parser.parse_args()

    try:
        validate_model(args.model)
        validate_concurrency(args.concurrency)
        http_url, perf_url, use_tls = normalize_endpoint(args.url)
    except ValueError as error:
        parser.exit(2, f"ERROR: {error}\n")

    print(http_url)
    print(perf_url)
    print("true" if use_tls else "false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
