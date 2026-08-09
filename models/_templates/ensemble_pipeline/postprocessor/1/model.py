"""Explicit postprocessing template for classification-style ensemble outputs."""

import json

import numpy as np
import triton_python_backend_utils as pb_utils


def postprocess_batch(raw_output, mode):
    if raw_output.ndim < 2:
        raise ValueError(
            f"RAW_OUTPUT must include batch and feature dimensions, got {raw_output.shape}"
        )
    if not np.isfinite(raw_output).all():
        raise ValueError("RAW_OUTPUT contains NaN or infinite values")
    values = raw_output.astype(np.float32, copy=False)
    if mode == "identity":
        return values
    if mode != "softmax":
        raise ValueError(f"unsupported postprocess mode: {mode}")

    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return (exponentials / np.sum(exponentials, axis=-1, keepdims=True)).astype(
        np.float32
    )


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        parameters = model_config.get("parameters", {})
        self.mode = parameters.get("mode", {}).get("string_value", "softmax")
        if self.mode not in {"identity", "softmax"}:
            raise ValueError(f"unsupported postprocess mode: {self.mode}")

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                raw_output = pb_utils.get_input_tensor_by_name(request, "RAW_OUTPUT")
                if raw_output is None:
                    raise ValueError("missing required input RAW_OUTPUT")
                output = postprocess_batch(raw_output.as_numpy(), self.mode)
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[pb_utils.Tensor("FINAL_OUTPUT", output)]
                    )
                )
            except (TypeError, ValueError) as error:
                responses.append(
                    pb_utils.InferenceResponse(error=pb_utils.TritonError(str(error)))
                )
        return responses
