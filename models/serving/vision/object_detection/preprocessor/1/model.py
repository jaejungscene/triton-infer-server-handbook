"""YOLOX input preprocessor with batched letterbox resizing."""

import json

import numpy as np
import triton_python_backend_utils as pb_utils


def _bilinear_resize(image, output_height, output_width):
    """Resize one HWC image without adding an OpenCV runtime dependency."""
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


def letterbox_batch(images, output_height=640, output_width=640, pad_value=114.0):
    """Letterbox a uint8 NHWC batch and return NCHW data plus recovery metadata."""
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"RAW_IMAGE must have shape [batch, height, width, 3], got {images.shape}")
    if images.shape[1] == 0 or images.shape[2] == 0:
        raise ValueError("RAW_IMAGE height and width must be greater than zero")

    batch_size, input_height, input_width, _ = images.shape
    scale = min(output_height / input_height, output_width / input_width)
    resized_height = max(1, min(output_height, round(input_height * scale)))
    resized_width = max(1, min(output_width, round(input_width * scale)))
    pad_y = (output_height - resized_height) // 2
    pad_x = (output_width - resized_width) // 2

    output = np.full(
        (batch_size, output_height, output_width, 3),
        pad_value,
        dtype=np.float32,
    )
    for index, image in enumerate(images):
        resized = _bilinear_resize(image.astype(np.float32), resized_height, resized_width)
        output[index, pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized

    metadata = np.tile(
        np.array([scale, pad_x, pad_y, input_width, input_height], dtype=np.float32),
        (batch_size, 1),
    )
    normalized = np.transpose(output / 255.0, (0, 3, 1, 2))
    return normalized.astype(np.float32), metadata


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        output_config = pb_utils.get_output_config_by_name(
            model_config, "PREPROCESSED_IMAGE"
        )
        dimensions = output_config["dims"]
        self.output_height = int(dimensions[1])
        self.output_width = int(dimensions[2])

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                input_tensor = pb_utils.get_input_tensor_by_name(request, "RAW_IMAGE")
                if input_tensor is None:
                    raise ValueError("missing required input RAW_IMAGE")
                images, metadata = letterbox_batch(
                    input_tensor.as_numpy(), self.output_height, self.output_width
                )
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[
                            pb_utils.Tensor("PREPROCESSED_IMAGE", images),
                            pb_utils.Tensor("IMAGE_METADATA", metadata),
                        ]
                    )
                )
            except (TypeError, ValueError) as error:
                responses.append(
                    pb_utils.InferenceResponse(error=pb_utils.TritonError(str(error)))
                )
        return responses
