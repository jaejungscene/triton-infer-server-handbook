#!/usr/bin/env python3
"""
to_onnx.py — PyTorch 모델을 ONNX 형식으로 변환

사용법:
    python scripts/convert/to_onnx.py \
        --model-path ./weights/model.torchscript.pt \
        --output-path ./model_repository/my_model/1/model.onnx \
        --input-shape 1,3,224,224 \
        --opset 17 \
        --dynamic-axes batch_size
"""

import argparse

from common import (
    ConversionError,
    atomic_output,
    load_pytorch_module,
    parse_shape,
    validated_input,
    validated_output,
)


def convert_to_onnx(
    model_path,
    output_path,
    input_shape,
    opset,
    dynamic_axes,
    expected_sha256=None,
    allow_unsafe_pickle=False,
):
    try:
        import onnx
        import torch
    except ImportError:
        raise ConversionError("PyTorch and ONNX are required") from None

    input_path = validated_input(model_path, expected_sha256)
    destination = validated_output(output_path, input_path)
    shape = parse_shape(input_shape)
    print(f"[convert] Loading model from {input_path}")
    model = load_pytorch_module(torch, input_path, allow_unsafe_pickle)
    model.eval()
    dummy_input = torch.randn(*shape)

    # Configure dynamic axes
    dynamic_axes_dict = None
    if dynamic_axes:
        dynamic_axes_dict = {
            "input": {0: dynamic_axes},
            "output": {0: dynamic_axes},
        }

    print(f"[convert] Exporting to ONNX (opset={opset}, shape={shape})")
    with atomic_output(destination) as temporary_output:
        torch.onnx.export(
            model,
            dummy_input,
            temporary_output,
            opset_version=opset,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes_dict,
        )
        onnx.checker.check_model(str(temporary_output))

    print(f"[convert] Saved to {destination}")


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX")
    parser.add_argument("--model-path", required=True, help="Path to PyTorch model (.pt/.pth)")
    parser.add_argument("--output-path", required=True, help="Output ONNX file path")
    parser.add_argument("--input-shape", default="1,3,224,224", help="Input shape (comma-separated)")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--dynamic-axes", default="batch_size", help="Dynamic axis name (or empty)")
    parser.add_argument("--expected-sha256", help="Expected lowercase SHA-256 of the input")
    parser.add_argument(
        "--allow-unsafe-pickle",
        action="store_true",
        help="Allow torch.load for a trusted full-module checkpoint",
    )
    args = parser.parse_args()

    try:
        convert_to_onnx(
            args.model_path,
            args.output_path,
            args.input_shape,
            args.opset,
            args.dynamic_axes,
            args.expected_sha256,
            args.allow_unsafe_pickle,
        )
    except (ConversionError, OSError) as error:
        parser.exit(2, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
