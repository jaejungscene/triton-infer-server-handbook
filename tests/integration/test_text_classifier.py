"""기본 manifest에 항상 포함되는 text_classifier E2E 계약."""

import numpy as np
import pytest


def test_required_text_classifier_contract(triton_url, triton_headers):
    try:
        import tritonclient.http as httpclient
    except ImportError:
        pytest.fail(
            "tritonclient is required: install requirements-integration.txt"
        )

    ssl_enabled = triton_url.startswith("https://")
    server_url = triton_url.split("://", 1)[-1].rstrip("/")
    client = httpclient.InferenceServerClient(
        url=server_url,
        connection_timeout=10,
        network_timeout=10,
        ssl=ssl_enabled,
    )
    try:
        assert client.is_model_ready("text_classifier", headers=triton_headers), \
            "required model text_classifier is not ready"

        texts = np.array(
            [["good and stable"], ["bad and slow"], ["ordinary text"]],
            dtype=object,
        )
        input_tensor = httpclient.InferInput(
            "INPUT_TEXT", list(texts.shape), "BYTES"
        )
        input_tensor.set_data_from_numpy(texts)
        outputs = [
            httpclient.InferRequestedOutput("LABEL"),
            httpclient.InferRequestedOutput("CONFIDENCE"),
        ]

        result = client.infer(
            "text_classifier", [input_tensor], outputs, headers=triton_headers
        )
        labels = result.as_numpy("LABEL")
        confidences = result.as_numpy("CONFIDENCE")

        decoded_labels = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in labels.reshape(-1)
        ]
        assert decoded_labels == ["positive", "negative", "neutral"]
        np.testing.assert_allclose(
            confidences.reshape(-1),
            np.array([0.91, 0.89, 0.62], dtype=np.float32),
        )
    finally:
        client.close()
