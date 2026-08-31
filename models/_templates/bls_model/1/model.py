"""
BLS (Business Logic Scripting) Template — Python Backend

Python 코드 내에서 다른 Triton 모델을 직접 호출하여
조건 분기, 루프 등 복잡한 비즈니스 로직을 구현합니다.

주요 패턴:
  1. pb_utils.InferenceRequest 로 다른 모델 호출
  2. 중간 결과를 기반으로 분기/집계
  3. request.trace() 로 트레이싱 전파 (OpenTelemetry 연동)
"""

import json

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                responses.append(self._execute_request(request))
            except Exception as error:
                self._log_error(
                    f"BLS request failed: {type(error).__name__}: {error}"
                )
                responses.append(self._error("BLS request failed; see server logs"))
        return responses

    def _execute_request(self, request):
        input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
        if input_tensor is None:
            return self._error("Missing required input: INPUT")
        input_data = input_tensor.as_numpy()

        # -----------------------------------------------------------------
        # Step 1: 첫 번째 모델 호출 (예: preprocessor)
        # -----------------------------------------------------------------
        preprocess_input = pb_utils.Tensor("RAW_INPUT", input_data)
        preprocess_request = pb_utils.InferenceRequest(
            model_name="preprocessor",
            requested_output_names=["PREPROCESSED"],
            inputs=[preprocess_input],
            trace=request.trace(),
        )
        preprocess_response = preprocess_request.exec()

        if preprocess_response.has_error():
            return self._downstream_error("Preprocess", preprocess_response)

        preprocessed = pb_utils.get_output_tensor_by_name(
            preprocess_response, "PREPROCESSED"
        )
        if preprocessed is None:
            return self._error("Preprocessor response is missing PREPROCESSED")

        # -----------------------------------------------------------------
        # Step 2: 조건 분기 (BLS의 핵심 장점)
        # -----------------------------------------------------------------
        # 예: 입력 크기에 따라 다른 모델 호출
        # if input_data.shape[-1] > threshold:
        #     model_to_call = "large_model"
        # else:
        #     model_to_call = "small_model"

        # -----------------------------------------------------------------
        # Step 3: 두 번째 모델 호출 (예: inferencer)
        # -----------------------------------------------------------------
        infer_request = pb_utils.InferenceRequest(
            model_name="inferencer",
            requested_output_names=["RAW_OUTPUT"],
            inputs=[preprocessed],
            trace=request.trace(),
        )
        infer_response = infer_request.exec()

        if infer_response.has_error():
            return self._downstream_error("Inference", infer_response)

        output = pb_utils.get_output_tensor_by_name(infer_response, "RAW_OUTPUT")
        if output is None:
            return self._error("Inferencer response is missing RAW_OUTPUT")
        result = output.as_numpy()

        output_tensor = pb_utils.Tensor("OUTPUT", result.astype(np.float32))
        return pb_utils.InferenceResponse(output_tensors=[output_tensor])

    @classmethod
    def _downstream_error(cls, stage, response):
        detail = response.error().message()
        cls._log_error(f"{stage} model failed: {detail}")
        return cls._error(f"{stage} model failed; see server logs")

    @staticmethod
    def _log_error(message):
        logger = getattr(pb_utils, "Logger", None)
        if logger is not None and hasattr(logger, "log_error"):
            logger.log_error(message)

    @staticmethod
    def _error(message):
        return pb_utils.InferenceResponse(error=pb_utils.TritonError(message))

    def finalize(self):
        pass
