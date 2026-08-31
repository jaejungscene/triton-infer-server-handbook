#!/usr/bin/env python3
"""
to_torchscript.py — PyTorch 모델을 TorchScript로 변환

사용법:
    python scripts/convert/to_torchscript.py \
        --model-path ./weights/model.pt \
        --output-path ./model_repository/my_model/1/model.pt \
        --method trace \
        --input-shape 1,3,224,224 \
        --expected-sha256 <64-lowercase-hex> \
        --allow-unsafe-pickle
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


def convert_to_torchscript(
    model_path,
    output_path,
    method,
    input_shape,
    expected_sha256=None,
    allow_unsafe_pickle=False,
):
    try:
        import torch
    except ImportError:
        raise ConversionError("PyTorch is required") from None

    input_path = validated_input(model_path, expected_sha256)
    destination = validated_output(output_path, input_path)
    shape = parse_shape(input_shape)
    print(f"[convert] Loading from {input_path}")
    model = load_pytorch_module(torch, input_path, allow_unsafe_pickle)
    model.eval()

    if method == "trace":
        dummy = torch.randn(*shape)
        scripted = torch.jit.trace(model, dummy)
    else:
        scripted = torch.jit.script(model)

    with atomic_output(destination) as temporary_output:
        scripted.save(str(temporary_output))
        torch.jit.load(str(temporary_output), map_location="cpu")
    print(f"[convert] Saved to {destination}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--method", choices=["trace", "script"], default="trace")
    parser.add_argument("--input-shape", default="1,3,224,224")
    parser.add_argument("--expected-sha256", help="Expected lowercase SHA-256 of the input")
    parser.add_argument(
        "--allow-unsafe-pickle",
        action="store_true",
        help="Allow torch.load for a trusted full-module checkpoint",
    )
    args = parser.parse_args()
    try:
        convert_to_torchscript(
            args.model_path,
            args.output_path,
            args.method,
            args.input_shape,
            args.expected_sha256,
            args.allow_unsafe_pickle,
        )
    except (ConversionError, OSError) as error:
        parser.exit(2, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
