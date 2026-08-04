"""Reliable gRPC streaming clients for decoupled Triton models."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from queue import Empty, Queue

import numpy as np

from .base import TritonConfig, grpc_tls_kwargs


class TritonStreamingClient:
    """Serialize and consume one decoupled gRPC stream at a time."""

    def __init__(self, config: TritonConfig | None = None):
        self.config = config or TritonConfig()
        import tritonclient.grpc as grpcclient

        self._grpcclient = grpcclient
        self._client = grpcclient.InferenceServerClient(
            url=self.config.grpc_url,
            verbose=self.config.verbose,
            ssl=self.config.ssl,
            **grpc_tls_kwargs(self.config),
        )
        self._stream_lock = threading.Lock()
        self._closed = False

    @staticmethod
    def _is_final(result) -> bool:
        response = result.get_response()
        parameters = getattr(response, "parameters", {})
        final_parameter = parameters.get("triton_final_response")
        return bool(getattr(final_parameter, "bool_param", False))

    @staticmethod
    def _decode_token(output: np.ndarray) -> str:
        value = output.reshape(-1)[0]
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    def _stream_request(
        self,
        model_name: str,
        inputs: list,
        output_name: str,
        model_version: str,
    ) -> Iterator[str]:
        if self._closed:
            raise RuntimeError("streaming client is closed")

        events: Queue[tuple[str, object]] = Queue()

        def response_callback(result, error):
            if error is not None:
                events.put(("error", error))
                return
            if result is None:
                events.put(("final", None))
                return

            output = result.as_numpy(output_name)
            if output is not None and output.size:
                events.put(("token", self._decode_token(output)))
            if self._is_final(result):
                events.put(("final", None))

        with self._stream_lock:
            self._client.start_stream(
                callback=response_callback,
                headers=self.config.headers,
            )
            try:
                self._client.async_stream_infer(
                    model_name=model_name,
                    model_version=model_version,
                    inputs=inputs,
                    outputs=[self._grpcclient.InferRequestedOutput(output_name)],
                    enable_empty_final_response=True,
                )

                while True:
                    try:
                        event, payload = events.get(timeout=self.config.timeout)
                    except Empty as exc:
                        raise TimeoutError(
                            f"No streaming response received for {self.config.timeout}s"
                        ) from exc

                    if event == "error":
                        raise RuntimeError(f"Streaming inference failed: {payload}")
                    if event == "final":
                        break
                    yield str(payload)
            finally:
                try:
                    self._client.stop_stream(cancel_requests=True)
                except TypeError:
                    self._client.stop_stream()

    def stream_infer(
        self,
        model_name: str,
        prompt: str,
        max_tokens: int = 128,
        model_version: str = "",
    ) -> Iterator[str]:
        """Stream from the repository's decoupled_streaming Python template."""
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")

        text_input = self._grpcclient.InferInput("INPUT_TEXT", [1, 1], "BYTES")
        text_input.set_data_from_numpy(np.array([[prompt]], dtype=object))
        token_input = self._grpcclient.InferInput("MAX_TOKENS", [1, 1], "INT32")
        token_input.set_data_from_numpy(np.array([[max_tokens]], dtype=np.int32))
        return self._stream_request(
            model_name,
            [text_input, token_input],
            "OUTPUT_TOKEN",
            model_version,
        )

    def stream_generate_vllm(
        self,
        model_name: str,
        prompt: str,
        max_tokens: int = 128,
        sampling_parameters: dict | None = None,
        model_version: str = "",
    ) -> Iterator[str]:
        """Stream from the vLLM backend contract in models/serving/nlp/llm."""
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")

        parameters = dict(sampling_parameters or {})
        parameters["max_tokens"] = max_tokens

        text_input = self._grpcclient.InferInput("text_input", [1], "BYTES")
        text_input.set_data_from_numpy(np.array([prompt], dtype=object))
        stream_input = self._grpcclient.InferInput("stream", [1], "BOOL")
        stream_input.set_data_from_numpy(np.array([True], dtype=np.bool_))
        parameters_input = self._grpcclient.InferInput(
            "sampling_parameters", [1], "BYTES"
        )
        parameters_input.set_data_from_numpy(
            np.array([json.dumps(parameters)], dtype=object)
        )
        return self._stream_request(
            model_name,
            [text_input, stream_input, parameters_input],
            "text_output",
            model_version,
        )

    def stream_infer_async(
        self,
        model_name: str,
        prompt: str,
        max_tokens: int = 128,
        callback: Callable[[str, bool], None] | None = None,
        model_version: str = "",
    ) -> threading.Thread:
        """Consume the template stream on a worker thread and return its handle."""
        token_callback = callback or (lambda token, is_final: None)

        def consume():
            try:
                for token in self.stream_infer(
                    model_name, prompt, max_tokens, model_version
                ):
                    token_callback(token, False)
                token_callback("", True)
            except Exception as exc:
                token_callback(f"ERROR: {exc}", True)

        thread = threading.Thread(target=consume, daemon=True)
        thread.start()
        return thread

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._client.stop_stream(cancel_requests=True)
        except (TypeError, RuntimeError):
            try:
                self._client.stop_stream()
            except RuntimeError:
                pass
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False
