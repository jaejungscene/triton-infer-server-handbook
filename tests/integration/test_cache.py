"""Required response-cache contract for the active text classifier."""

import re
import uuid

import numpy as np
import pytest
import requests


def _model_metric_value(metrics: str, metric_name: str, model_name: str) -> float:
    total = 0.0
    matched = False
    model_label = re.compile(rf'(?:^|,)model="{re.escape(model_name)}"(?:,|$)')
    prefix = f"{metric_name}{{"
    for line in metrics.splitlines():
        if not line.startswith(prefix):
            continue
        labels, separator, sample = line[len(prefix) :].partition("}")
        if not separator or model_label.search(labels) is None:
            continue
        total += float(sample.strip().split()[0])
        matched = True
    if not matched:
        raise AssertionError(f"Missing {metric_name} for model={model_name}")
    return total


def _read_cache_counters(
    metrics_url: str, model_name: str, headers: dict[str, str]
) -> tuple[float, float]:
    response = requests.get(
        f"{metrics_url.rstrip('/')}/metrics", headers=headers, timeout=10
    )
    response.raise_for_status()
    return (
        _model_metric_value(
            response.text, "nv_cache_num_hits_per_model", model_name
        ),
        _model_metric_value(
            response.text, "nv_cache_num_misses_per_model", model_name
        ),
    )


def test_required_model_records_cache_miss_then_hit(
    triton_url,
    triton_metrics_url,
    triton_headers,
    triton_metrics_headers,
):
    try:
        import tritonclient.http as httpclient
    except ImportError:
        pytest.fail("tritonclient is required: install requirements-integration.txt")

    model_name = "text_classifier"
    ssl_enabled = triton_url.startswith("https://")
    server_url = triton_url.split("://", 1)[-1].rstrip("/")
    client = httpclient.InferenceServerClient(
        url=server_url,
        connection_timeout=10,
        network_timeout=10,
        ssl=ssl_enabled,
    )
    try:
        assert client.is_model_ready(
            model_name, headers=triton_headers
        ), f"required model {model_name} is not ready"
        hits_before, misses_before = _read_cache_counters(
            triton_metrics_url, model_name, triton_metrics_headers
        )

        unique_text = np.array([[f"cache-contract-{uuid.uuid4().hex}"]], dtype=object)
        input_tensor = httpclient.InferInput(
            "INPUT_TEXT", list(unique_text.shape), "BYTES"
        )
        input_tensor.set_data_from_numpy(unique_text)
        outputs = [
            httpclient.InferRequestedOutput("LABEL"),
            httpclient.InferRequestedOutput("CONFIDENCE"),
        ]

        first = client.infer(
            model_name, [input_tensor], outputs, headers=triton_headers
        )
        second = client.infer(
            model_name, [input_tensor], outputs, headers=triton_headers
        )

        np.testing.assert_array_equal(
            first.as_numpy("LABEL"), second.as_numpy("LABEL")
        )
        np.testing.assert_array_equal(
            first.as_numpy("CONFIDENCE"), second.as_numpy("CONFIDENCE")
        )
        hits_after, misses_after = _read_cache_counters(
            triton_metrics_url, model_name, triton_metrics_headers
        )
        assert hits_after >= hits_before + 1
        assert misses_after >= misses_before + 1
    finally:
        client.close()
