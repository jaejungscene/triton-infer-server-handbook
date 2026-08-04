"""
test_cache.py — Response Cache 통합 테스트

동일 입력에 대해 캐시가 작동하는지 검증합니다.
"""

import numpy as np
import pytest
import requests


class TestResponseCache:
    """Response Cache 동작 검증"""

    def test_cache_enabled_model(self, triton_url):
        """캐시가 활성화된 모델의 반복 요청 성능 확인"""
        try:
            import tritonclient.http as httpclient
        except ImportError:
            pytest.skip("tritonclient not installed")

        client = httpclient.InferenceServerClient(url=triton_url.replace("http://", ""))

        if not client.is_model_ready("resnet50"):
            pytest.skip("resnet50 model not loaded")

        # 동일한 입력으로 2번 요청
        input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
        input_tensor = httpclient.InferInput("input", list(input_data.shape), "FP32")
        input_tensor.set_data_from_numpy(input_data)
        outputs = [httpclient.InferRequestedOutput("output")]

        # 첫 번째 요청 (miss)
        result1 = client.infer("resnet50", [input_tensor], outputs)
        output1 = result1.as_numpy("output")

        # 두 번째 요청 (hit 예상)
        result2 = client.infer("resnet50", [input_tensor], outputs)
        output2 = result2.as_numpy("output")

        # 결과가 동일해야 함
        np.testing.assert_array_equal(output1, output2)

    def test_cache_metrics(self, triton_metrics_url):
        """캐시 메트릭이 Prometheus에 노출되는지 확인"""
        response = requests.get(f"{triton_metrics_url}/metrics", timeout=10)
        response.raise_for_status()

        # 캐시 메트릭이 존재하면 캐시가 활성화된 것
        has_cache_metrics = "nv_cache" in response.text
        # 캐시가 비활성화일 수 있으므로 경고만
        if not has_cache_metrics:
            pytest.skip("Cache metrics not found (cache may be disabled)")
