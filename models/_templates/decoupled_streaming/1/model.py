"""
Decoupled Streaming Template — Python Backend

요청 1건에 대해 토큰 단위로 여러 응답을 스트리밍하는 패턴.
LLM 서빙, ASR 실시간 전사 등에 사용.

핵심 API:
  - response_sender = request.get_response_sender()
  - response_sender.send(response)                       # 중간 응답
  - response_sender.send(response, flags=FINAL)           # 마지막 응답
"""

import json

import numpy as np
import triton_python_backend_utils as pb_utils


class RequestValidationError(ValueError):
    """Safe request-contract error that can be returned to a client."""


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        parameters = self.model_config.get("parameters", {})
        self.max_output_tokens = int(
            parameters.get("max_output_tokens", {}).get("string_value", "4096")
        )
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be greater than zero")
        # 실제 LLM 엔진 초기화 (vLLM, TGI 등)
        # self.engine = ...

    def execute(self, requests):
        for request in requests:
            response_sender = None
            try:
                response_sender = request.get_response_sender()
                self._execute_request(request, response_sender)
            except RequestValidationError as error:
                self._send_error(response_sender, str(error))
            except Exception as error:
                self._log_error(
                    f"Streaming request failed: {type(error).__name__}: {error}"
                )
                self._send_error(
                    locals().get("response_sender"),
                    "Streaming request failed; see server logs",
                )

        return None

    def _execute_request(self, request, response_sender):
        if response_sender.is_cancelled():
            self._send_final(response_sender)
            return

        input_text = self._single_value(request, "INPUT_TEXT")
        prompt = self._as_text(input_text)
        try:
            max_tokens = int(self._single_value(request, "MAX_TOKENS"))
        except (TypeError, ValueError) as error:
            raise RequestValidationError(
                "MAX_TOKENS must contain one integer"
            ) from error
        if max_tokens <= 0 or max_tokens > self.max_output_tokens:
            raise RequestValidationError(
                f"MAX_TOKENS must be between 1 and {self.max_output_tokens}"
            )

        tokens = iter(self._generate_tokens(prompt, max_tokens))
        try:
            current_token = next(tokens)
        except StopIteration:
            self._send_final(response_sender)
            return

        while True:
            if response_sender.is_cancelled():
                self._send_final(response_sender)
                return
            try:
                next_token = next(tokens)
            except StopIteration:
                self._send_token(response_sender, current_token, final=True)
                return
            self._send_token(response_sender, current_token, final=False)
            current_token = next_token

    @staticmethod
    def _single_value(request, name):
        tensor = pb_utils.get_input_tensor_by_name(request, name)
        if tensor is None:
            raise RequestValidationError(f"Missing required input: {name}")
        values = tensor.as_numpy().reshape(-1)
        if values.size != 1:
            raise RequestValidationError(f"{name} must contain exactly one value")
        return values[0]

    @staticmethod
    def _send_token(response_sender, token, final):
        output_tensor = pb_utils.Tensor(
            "OUTPUT_TOKEN", np.array([token], dtype=object)
        )
        response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
        flags = pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL if final else 0
        response_sender.send(response, flags=flags)

    @staticmethod
    def _send_final(response_sender):
        response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

    @classmethod
    def _send_error(cls, response_sender, message):
        if response_sender is None:
            cls._log_error(f"Could not return streaming error: {message}")
            return
        try:
            response_sender.send(
                pb_utils.InferenceResponse(error=pb_utils.TritonError(message)),
                flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,
            )
        except Exception as send_error:
            cls._log_error(
                f"Could not return streaming error: "
                f"{type(send_error).__name__}: {send_error}"
            )

    @staticmethod
    def _log_error(message):
        logger = getattr(pb_utils, "Logger", None)
        if logger is not None and hasattr(logger, "log_error"):
            logger.log_error(message)

    def _generate_tokens(self, prompt, max_tokens):
        """
        토큰 생성 (placeholder).
        실제 구현에서는 vLLM/HuggingFace generate() 사용.
        """
        # Placeholder: 실제 LLM 엔진으로 교체
        return (f"token_{index}" for index in range(min(max_tokens, 10)))

    @staticmethod
    def _as_text(value):
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except UnicodeDecodeError as error:
                raise RequestValidationError(
                    "INPUT_TEXT must contain valid UTF-8"
                ) from error
        return str(value)

    def finalize(self):
        # LLM 엔진 정리
        pass
