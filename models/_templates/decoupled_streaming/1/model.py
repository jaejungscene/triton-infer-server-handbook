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


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        # 실제 LLM 엔진 초기화 (vLLM, TGI 등)
        # self.engine = ...

    def execute(self, requests):
        for request in requests:
            response_sender = request.get_response_sender()

            try:
                input_text = (
                    pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT")
                    .as_numpy()[0]
                    .decode("utf-8")
                )

                max_tokens_tensor = pb_utils.get_input_tensor_by_name(
                    request, "MAX_TOKENS"
                )
                max_tokens = int(max_tokens_tensor.as_numpy()[0])

                # ---------------------------------------------------------
                # 토큰 스트리밍 루프
                # 실제 구현에서는 LLM 엔진의 generate() 이터레이터 사용
                # ---------------------------------------------------------
                generated_tokens = self._generate_tokens(input_text, max_tokens)

                if not generated_tokens:
                    response_sender.send(
                        pb_utils.InferenceResponse(output_tensors=[]),
                        flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                    )
                    continue

                for i, token in enumerate(generated_tokens):
                    is_last = i == len(generated_tokens) - 1

                    output_tensor = pb_utils.Tensor(
                        "OUTPUT_TOKEN",
                        np.array([token], dtype=object),
                    )
                    response = pb_utils.InferenceResponse(
                        output_tensors=[output_tensor]
                    )

                    if is_last:
                        # ★ 마지막 토큰: TRITONSERVER_RESPONSE_COMPLETE_FINAL 플래그
                        response_sender.send(
                            response,
                            flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                        )
                    else:
                        response_sender.send(response)

            except Exception as e:
                error_response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(str(e))
                )
                response_sender.send(
                    error_response,
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                )

        return None  # decoupled 모델은 None 반환

    def _generate_tokens(self, prompt, max_tokens):
        """
        토큰 생성 (placeholder).
        실제 구현에서는 vLLM/HuggingFace generate() 사용.
        """
        # Placeholder: 실제 LLM 엔진으로 교체
        tokens = [f"token_{i}" for i in range(min(max_tokens, 10))]
        return tokens

    def finalize(self):
        # LLM 엔진 정리
        pass
