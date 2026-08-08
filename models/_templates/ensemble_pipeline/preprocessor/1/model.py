"""Shape-safe image preprocessing template for an ensemble pipeline."""

import json

import numpy as np
import triton_python_backend_utils as pb_utils


def _bilinear_resize(image, output_height, output_width):
    input_height, input_width = image.shape[:2]
    if (input_height, input_width) == (output_height, output_width):
        return image.astype(np.float32, copy=False)

    y = np.linspace(0, input_height - 1, output_height, dtype=np.float32)
    x = np.linspace(0, input_width - 1, output_width, dtype=np.float32)
    y0 = np.floor(y).astype(np.int32)
    x0 = np.floor(x).astype(np.int32)
    y1 = np.minimum(y0 + 1, input_height - 1)
    x1 = np.minimum(x0 + 1, input_width - 1)
    y_weight = (y - y0)[:, None, None]
    x_weight = (x - x0)[None, :, None]

    top = (
        image[y0[:, None], x0[None, :]] * (1.0 - x_weight)
        + image[y0[:, None], x1[None, :]] * x_weight
    )
    bottom = (
        image[y1[:, None], x0[None, :]] * (1.0 - x_weight)
        + image[y1[:, None], x1[None, :]] * x_weight
    )
    return top * (1.0 - y_weight) + bottom * y_weight


def preprocess_batch(images, output_height, output_width, mean, std):
    """Resize a uint8 NHWC batch, normalize it, and return contiguous NCHW FP32."""
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(
            f"RAW_INPUT must have shape [batch, height, width, 3], got {images.shape}"
        )
    if images.dtype != np.uint8:
        raise ValueError(f"RAW_INPUT must use uint8, got {images.dtype}")
    if images.shape[1] == 0 or images.shape[2] == 0:
        raise ValueError("RAW_INPUT height and width must be greater than zero")

    resized = np.empty(
        (images.shape[0], output_height, output_width, 3), dtype=np.float32
    )
    for index, image in enumerate(images):
        resized[index] = _bilinear_resize(image, output_height, output_width)

    normalized = (resized / 255.0 - mean) / std
    return np.ascontiguousarray(np.transpose(normalized, (0, 3, 1, 2)), dtype=np.float32)


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        output_config = pb_utils.get_output_config_by_name(model_config, "PREPROCESSED")
        dimensions = output_config["dims"]
        self.output_height = int(dimensions[1])
        self.output_width = int(dimensions[2])
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                raw_input = pb_utils.get_input_tensor_by_name(request, "RAW_INPUT")
                if raw_input is None:
                    raise ValueError("missing required input RAW_INPUT")
                output = preprocess_batch(
                    raw_input.as_numpy(),
                    self.output_height,
                    self.output_width,
                    self.mean,
                    self.std,
                )
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[pb_utils.Tensor("PREPROCESSED", output)]
                    )
                )
            except (TypeError, ValueError) as error:
                responses.append(
                    pb_utils.InferenceResponse(error=pb_utils.TritonError(str(error)))
                )
        return responses
