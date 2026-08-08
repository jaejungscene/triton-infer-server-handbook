"""Small deterministic text classifier for smoke tests and cache validation."""

import re

import numpy as np
import triton_python_backend_utils as pb_utils


POSITIVE_KEYWORDS = frozenset(
    {"good", "great", "love", "fast", "stable", "좋다", "빠르다"}
)
NEGATIVE_KEYWORDS = frozenset(
    {"bad", "slow", "fail", "error", "unstable", "나쁘다", "느리다"}
)


def classify_text(text):
    """Classify exact Unicode word tokens instead of ambiguous substrings."""
    tokens = set(re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE))
    positive = bool(tokens & POSITIVE_KEYWORDS)
    negative = bool(tokens & NEGATIVE_KEYWORDS)
    if positive and not negative:
        return "positive", 0.91
    if negative and not positive:
        return "negative", 0.89
    return "neutral", 0.62


class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT")
                if input_tensor is None:
                    raise ValueError("missing required input INPUT_TEXT")
                texts = input_tensor.as_numpy().reshape(-1)

                labels = []
                confidences = []
                for raw_text in texts:
                    text = (
                        raw_text.decode("utf-8", errors="strict")
                        if isinstance(raw_text, bytes)
                        else str(raw_text)
                    )
                    label, confidence = classify_text(text)
                    labels.append(label)
                    confidences.append(confidence)

                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[
                            pb_utils.Tensor(
                                "LABEL",
                                np.array(labels, dtype=object).reshape(-1, 1),
                            ),
                            pb_utils.Tensor(
                                "CONFIDENCE",
                                np.array(confidences, dtype=np.float32).reshape(-1, 1),
                            ),
                        ]
                    )
                )
            except (TypeError, UnicodeDecodeError, ValueError) as error:
                responses.append(
                    pb_utils.InferenceResponse(error=pb_utils.TritonError(str(error)))
                )
        return responses
