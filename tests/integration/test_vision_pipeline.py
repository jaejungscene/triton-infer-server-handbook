"""
test_vision_pipeline.py — Object Detection Ensemble Pipeline 통합 테스트

전처리 → YOLOX → 후처리 전체 파이프라인이 정상 동작하는지 검증합니다.
"""

import numpy as np
import pytest


class TestVisionPipeline:
    """Object Detection Ensemble Pipeline 통합 테스트"""

    @pytest.fixture
    def sample_image(self):
        """테스트용 더미 이미지 (640x640x3)"""
        return np.random.randint(0, 255, (1, 640, 640, 3), dtype=np.uint8)

    def test_pipeline_inference(self, triton_url, triton_headers, sample_image):
        """Ensemble Pipeline 추론 E2E 테스트"""
        try:
            import tritonclient.http as httpclient
        except ImportError:
            pytest.skip("tritonclient not installed")

        client = httpclient.InferenceServerClient(
            url=triton_url.split("://", 1)[-1],
            ssl=triton_url.startswith("https://"),
        )

        if not client.is_model_ready("od_pipeline", headers=triton_headers):
            pytest.skip("od_pipeline model not loaded")

        input_tensor = httpclient.InferInput("RAW_IMAGE", list(sample_image.shape), "UINT8")
        input_tensor.set_data_from_numpy(sample_image)

        outputs = [
            httpclient.InferRequestedOutput("BBOXES"),
            httpclient.InferRequestedOutput("SCORES"),
            httpclient.InferRequestedOutput("CLASS_IDS"),
            httpclient.InferRequestedOutput("NUM_DETECTIONS"),
        ]

        result = client.infer(
            "od_pipeline", [input_tensor], outputs, headers=triton_headers
        )

        bboxes = result.as_numpy("BBOXES")
        scores = result.as_numpy("SCORES")
        class_ids = result.as_numpy("CLASS_IDS")
        counts = result.as_numpy("NUM_DETECTIONS")

        assert bboxes.shape == (1, 100, 4)
        assert scores.shape == (1, 100)
        assert class_ids.shape == (1, 100)
        assert counts.shape == (1, 1)
        assert 0 <= counts[0, 0] <= 100

    def test_preprocessor_standalone(self, triton_url, triton_headers, sample_image):
        """전처리 모델 단독 테스트"""
        try:
            import tritonclient.http as httpclient
        except ImportError:
            pytest.skip("tritonclient not installed")

        client = httpclient.InferenceServerClient(
            url=triton_url.split("://", 1)[-1],
            ssl=triton_url.startswith("https://"),
        )

        if not client.is_model_ready("od_preprocessor", headers=triton_headers):
            pytest.skip("od_preprocessor model not loaded")

        input_tensor = httpclient.InferInput("RAW_IMAGE", list(sample_image.shape), "UINT8")
        input_tensor.set_data_from_numpy(sample_image)

        outputs = [httpclient.InferRequestedOutput("PREPROCESSED_IMAGE")]

        result = client.infer(
            "od_preprocessor", [input_tensor], outputs, headers=triton_headers
        )
        preprocessed = result.as_numpy("PREPROCESSED_IMAGE")

        assert preprocessed.dtype == np.float32
        assert preprocessed.shape == (1, 3, 640, 640)
