"""
Streaming Client — gRPC Decoupled Streaming (LLM 토큰 스트리밍)

Decoupled 모델 (model_transaction_policy { decoupled: true }) 전용.
HTTP endpoint에서는 사용 불가 — gRPC bi-directional streaming 필수.

사용 예:
    from client.streaming_client import TritonStreamingClient, TritonConfig

    config = TritonConfig(grpc_url="localhost:8001")
    client = TritonStreamingClient(config)

    # 동기 스트리밍
    for token in client.stream_infer("llm_vllm", "Hello, how are you?", max_tokens=100):
        print(token, end="", flush=True)

    # 콜백 기반 비동기 스트리밍
    client.stream_infer_async(
        "llm_vllm", "Hello!", max_tokens=50,
        callback=lambda token, is_final: print(token, end="")
    )
"""

from queue import Queue
from typing import Callable, Iterator

import numpy as np

from .base import TritonConfig


class TritonStreamingClient:
    """gRPC decoupled streaming 클라이언트"""

    def __init__(self, config: TritonConfig | None = None):
        self.config = config or TritonConfig()
        import tritonclient.grpc as grpcclient

        self._grpcclient = grpcclient
        self._client = grpcclient.InferenceServerClient(
            url=self.config.grpc_url,
            verbose=self.config.verbose,
        )

    def stream_infer(
        self,
        model_name: str,
        prompt: str,
        max_tokens: int = 128,
        model_version: str = "",
    ) -> Iterator[str]:
        """
        동기 스트리밍 추론 — generator로 토큰 반환

        사용:
            for token in client.stream_infer("llm", "Hello"):
                print(token, end="")
        """
        result_queue: Queue = Queue()
        error_holder = [None]

        def _callback(result, error):
            if error:
                error_holder[0] = error
                result_queue.put(None)
                return

            if result:
                output = result.as_numpy("OUTPUT_TOKEN")
                if output is not None:
                    token = output[0]
                    if isinstance(token, bytes):
                        token = token.decode("utf-8")
                    result_queue.put(token)

                # Check for final response
                params = result.get_response()
                is_final = params.parameters.get("triton_final_response", {}).bool_param if hasattr(params, 'parameters') else False
                if is_final:
                    result_queue.put(None)
            else:
                result_queue.put(None)

        # Build inputs
        text_input = self._grpcclient.InferInput("INPUT_TEXT", [1, 1], "BYTES")
        text_input.set_data_from_numpy(np.array([[prompt]], dtype=object))

        max_tokens_input = self._grpcclient.InferInput("MAX_TOKENS", [1, 1], "INT32")
        max_tokens_input.set_data_from_numpy(np.array([[max_tokens]], dtype=np.int32))

        output = self._grpcclient.InferRequestedOutput("OUTPUT_TOKEN")

        self._client.start_stream(callback=_callback)
        try:
            self._client.async_stream_infer(
                model_name=model_name,
                model_version=model_version,
                inputs=[text_input, max_tokens_input],
                outputs=[output],
                enable_empty_final_response=True,
            )

            while True:
                token = result_queue.get()
                if token is None:
                    if error_holder[0]:
                        raise RuntimeError(f"Inference error: {error_holder[0]}")
                    break
                yield token
        finally:
            self._client.stop_stream()

    def stream_infer_async(
        self,
        model_name: str,
        prompt: str,
        max_tokens: int = 128,
        callback: Callable[[str, bool], None] = lambda t, f: None,
        model_version: str = "",
    ):
        """
        콜백 기반 비동기 스트리밍 추론

        callback(token: str, is_final: bool)
        """

        def _callback(result, error):
            if error:
                callback(f"ERROR: {error}", True)
                return

            if result:
                output = result.as_numpy("OUTPUT_TOKEN")
                if output is not None:
                    token = output[0]
                    if isinstance(token, bytes):
                        token = token.decode("utf-8")

                    params = result.get_response()
                    is_final = params.parameters.get("triton_final_response", {}).bool_param if hasattr(params, 'parameters') else False
                    callback(token, is_final)

        text_input = self._grpcclient.InferInput("INPUT_TEXT", [1, 1], "BYTES")
        text_input.set_data_from_numpy(np.array([[prompt]], dtype=object))

        max_tokens_input = self._grpcclient.InferInput("MAX_TOKENS", [1, 1], "INT32")
        max_tokens_input.set_data_from_numpy(np.array([[max_tokens]], dtype=np.int32))

        output = self._grpcclient.InferRequestedOutput("OUTPUT_TOKEN")

        self._client.start_stream(callback=_callback)
        self._client.async_stream_infer(
            model_name=model_name,
            model_version=model_version,
            inputs=[text_input, max_tokens_input],
            outputs=[output],
            enable_empty_final_response=True,
        )

    def close(self):
        try:
            self._client.stop_stream()
        except Exception:
            pass
        self._client.close()
