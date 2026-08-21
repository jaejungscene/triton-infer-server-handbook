"""
test_llm_streaming.py — LLM Decoupled Streaming 통합 테스트

gRPC bi-directional streaming으로 토큰 단위 응답을 검증합니다.
"""

import pytest


class TestLLMStreaming:
    """LLM Decoupled Streaming 통합 테스트"""

    def test_streaming_inference(self, triton_grpc_url, triton_headers):
        """스트리밍 추론 E2E 테스트"""
        try:
            from client.streaming_client import TritonStreamingClient
            from client.base import TritonConfig
        except ImportError:
            pytest.skip("client module not available")

        config = TritonConfig(grpc_url=triton_grpc_url, headers=triton_headers)
        client = TritonStreamingClient(config)

        try:
            if not client._client.is_model_ready(
                "llm_vllm", headers=triton_headers
            ):
                pytest.skip("llm_vllm model not loaded")
        except Exception as exc:
            pytest.fail(f"Cannot connect to Triton gRPC: {exc}")

        tokens = []
        try:
            for token in client.stream_generate_vllm(
                "llm_vllm", "Hello", max_tokens=5
            ):
                tokens.append(token)
                assert isinstance(token, str)
        finally:
            client.close()

        assert len(tokens) > 0, "No tokens received"

    def test_decoupled_model_config(self, triton_grpc_url, triton_headers):
        """Decoupled 모델이 올바르게 설정되었는지 확인"""
        try:
            import tritonclient.grpc as grpcclient
        except ImportError:
            pytest.skip("tritonclient not installed")

        client = grpcclient.InferenceServerClient(url=triton_grpc_url)

        try:
            model_ready = client.is_model_ready(
                "llm_vllm", headers=triton_headers
            )
        except Exception as exc:
            pytest.fail(f"Cannot connect to Triton gRPC: {exc}")
        if not model_ready:
            pytest.skip("llm_vllm model not loaded")

        config = client.get_model_config("llm_vllm", headers=triton_headers)
        # Decoupled 모델은 model_transaction_policy.decoupled = true
        assert config.config.model_transaction_policy.decoupled is True
        client.close()
