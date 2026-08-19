"""Static guards that keep required live gates from degrading into optional tests."""

import importlib.util
from pathlib import Path

import pytest


def test_required_cache_gate_uses_active_model_and_counter_deltas(project_root):
    source = Path(
        project_root, "tests", "integration", "test_cache.py"
    ).read_text(encoding="utf-8")

    assert 'model_name = "text_classifier"' in source
    assert "nv_cache_num_hits_per_model" in source
    assert "nv_cache_num_misses_per_model" in source
    assert "hits_after >= hits_before + 1" in source
    assert "misses_after >= misses_before + 1" in source
    assert 'pytest.skip("Cache metrics' not in source


def test_cache_metric_parser_sums_model_versions(project_root):
    path = Path(project_root, "tests", "integration", "test_cache.py")
    spec = importlib.util.spec_from_file_location("cache_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metrics = """
nv_cache_num_hits_per_model{model="text_classifier",version="1"} 2
nv_cache_num_hits_per_model{model="text_classifier",version="2"} 3
nv_cache_num_hits_per_model{model="another_model",version="1"} 100
"""

    assert module._model_metric_value(
        metrics, "nv_cache_num_hits_per_model", "text_classifier"
    ) == 5


def test_live_test_auth_headers_reject_injection(monkeypatch):
    import conftest

    monkeypatch.setenv("TRITON_AUTH_TOKEN", "token\nX-Forged: true")
    with pytest.raises(pytest.UsageError, match="line breaks"):
        conftest._authorization_headers("TRITON_AUTH_TOKEN")

    monkeypatch.setenv("TRITON_AUTH_TOKEN", "test-token")
    assert conftest._authorization_headers("TRITON_AUTH_TOKEN") == {
        "Authorization": "Bearer test-token"
    }


def test_metrics_auth_can_use_a_separate_token(monkeypatch):
    import conftest

    monkeypatch.setenv("TRITON_AUTH_TOKEN", "inference-token")
    monkeypatch.setenv("TRITON_METRICS_AUTH_TOKEN", "metrics-token")
    assert conftest._authorization_headers(
        "TRITON_METRICS_AUTH_TOKEN", fallback_variable="TRITON_AUTH_TOKEN"
    ) == {"Authorization": "Bearer metrics-token"}


def test_metrics_url_preserves_ipv6_origin_shape():
    import conftest

    assert conftest._derive_metrics_url("https://[2001:db8::1]:8000") == \
        "https://[2001:db8::1]:8002"
