"""Reliable gRPC streaming clients for decoupled Triton models."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from queue import Empty, Queue

import numpy as np

from .base import TritonConfig, grpc_tls_kwargs

_LOGGER = logging.getLogger(__name__)


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
        self._async_state_lock = threading.Lock()
        self._async_cancel_events: set[threading.Event] = set()
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
        cancel_event: threading.Event | None = None,
    ) -> Iterator[str]:
        if self._closed:
            raise RuntimeError("streaming client is closed")

        events: Queue[tuple[str, object]] = Queue()

        def response_callback(result, error):
            try:
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
            except Exception as exc:
                events.put(("error", exc))

        while not self._stream_lock.acquire(timeout=0.1):
            if cancel_event is not None and cancel_event.is_set():
                return
        try:
            if cancel_event is not None and cancel_event.is_set():
                return
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

                idle_deadline = time.monotonic() + self.config.timeout
                while cancel_event is None or not cancel_event.is_set():
                    remaining = idle_deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            f"No streaming response received for {self.config.timeout}s"
                        )
                    try:
                        event, payload = events.get(timeout=min(0.1, remaining))
                    except Empty:
                        continue

                    idle_deadline = time.monotonic() + self.config.timeout
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
        finally:
            self._stream_lock.release()

    def stream_infer(
        self,
        model_name: str,
        prompt: str,
        max_tokens: int = 128,
        model_version: str = "",
    ) -> Iterator[str]:
        """Stream from the repository's decoupled_streaming Python template."""
        self._validate_stream_inputs(model_name, prompt, max_tokens)

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
        self._validate_stream_inputs(model_name, prompt, max_tokens)

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
        error_callback: Callable[[Exception], None] | None = None,
        model_version: str = "",
    ) -> Future[None]:
        """Consume a stream on a worker and expose completion or failure as a Future."""
        self._validate_stream_inputs(model_name, prompt, max_tokens)
        token_callback = callback or (lambda token, is_final: None)
        completion: Future[None] = Future()
        cancel_event = threading.Event()
        with self._async_state_lock:
            if self._closed:
                raise RuntimeError("streaming client is closed")
            self._async_cancel_events.add(cancel_event)

        def observe_completion(future):
            if future.cancelled():
                cancel_event.set()

        completion.add_done_callback(observe_completion)

        def consume():
            try:
                text_input = self._grpcclient.InferInput(
                    "INPUT_TEXT", [1, 1], "BYTES"
                )
                text_input.set_data_from_numpy(np.array([[prompt]], dtype=object))
                token_input = self._grpcclient.InferInput(
                    "MAX_TOKENS", [1, 1], "INT32"
                )
                token_input.set_data_from_numpy(
                    np.array([[max_tokens]], dtype=np.int32)
                )
                for token in self._stream_request(
                    model_name,
                    [text_input, token_input],
                    "OUTPUT_TOKEN",
                    model_version,
                    cancel_event,
                ):
                    if cancel_event.is_set():
                        return
                    token_callback(token, False)
                if not cancel_event.is_set() and not completion.done():
                    token_callback("", True)
                    completion.set_result(None)
            except Exception as exc:
                if cancel_event.is_set() or completion.cancelled():
                    return
                if error_callback is not None:
                    try:
                        error_callback(exc)
                    except Exception:
                        _LOGGER.exception("Streaming error callback failed")
                if not completion.done():
                    completion.set_exception(exc)
            finally:
                with self._async_state_lock:
                    self._async_cancel_events.discard(cancel_event)

        thread = threading.Thread(target=consume, daemon=True)
        thread.start()
        return completion

    @staticmethod
    def _validate_stream_inputs(model_name: str, prompt: str, max_tokens: int) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if (
            not isinstance(max_tokens, int)
            or isinstance(max_tokens, bool)
            or max_tokens <= 0
        ):
            raise ValueError("max_tokens must be a positive integer")

    def close(self):
        if self._closed:
            return
        self._closed = True
        with self._async_state_lock:
            for cancel_event in self._async_cancel_events:
                cancel_event.set()
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
