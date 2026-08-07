"""
conftest.py — pytest 공통 fixture

Triton 서버 URL, 공용 클라이언트 인스턴스 등을 제공합니다.
"""

import os
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--triton-url",
        action="store",
        default=os.getenv("TRITON_HTTP_URL", "http://localhost:8000"),
        help="Triton HTTP endpoint URL",
    )
    parser.addoption(
        "--triton-grpc-url",
        action="store",
        default=os.getenv("TRITON_GRPC_URL", "localhost:8001"),
        help="Triton gRPC endpoint URL",
    )
    parser.addoption(
        "--triton-metrics-url",
        action="store",
        default=os.getenv("TRITON_METRICS_URL"),
        help="Triton Prometheus metrics endpoint URL",
    )
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="Run smoke/integration tests against a live Triton server",
    )


def pytest_collection_modifyitems(config, items):
    tests_root = Path(__file__).resolve().parent
    run_live = config.getoption("--run-live")

    for item in items:
        try:
            suite = Path(item.path).resolve().relative_to(tests_root).parts[0]
        except (ValueError, IndexError):
            continue
        if suite not in {"smoke", "integration"}:
            continue

        item.add_marker(getattr(pytest.mark, suite))
        if not run_live:
            item.add_marker(
                pytest.mark.skip(
                    reason=f"{suite} test requires --run-live and a Triton endpoint"
                )
            )


def _derive_metrics_url(http_url):
    parsed = urlsplit(http_url)
    host = parsed.hostname or "localhost"
    netloc = f"{host}:8002"
    if parsed.username:
        auth = parsed.username
        if parsed.password:
            auth = f"{auth}:{parsed.password}"
        netloc = f"{auth}@{netloc}"
    return urlunsplit((parsed.scheme or "http", netloc, "", "", ""))


@pytest.fixture(scope="session")
def triton_url(request):
    return request.config.getoption("--triton-url")


@pytest.fixture(scope="session")
def triton_grpc_url(request):
    return request.config.getoption("--triton-grpc-url")


@pytest.fixture(scope="session")
def triton_metrics_url(request, triton_url):
    return request.config.getoption("--triton-metrics-url") or _derive_metrics_url(triton_url)


@pytest.fixture(scope="session")
def project_root():
    """프로젝트 루트 디렉토리 경로"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
