#!/usr/bin/env python3
"""Validate TensorRT conversion paths and dynamic-shape profiles."""

from __future__ import annotations

import argparse
import re

from common import ConversionError, validated_input, validated_output


_TENSOR_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*")


def parse_profile(value: str, label: str) -> dict[str, tuple[int, ...]]:
    tensors = {}
    for item in value.split(","):
        try:
            name, shape = item.split(":", 1)
            dimensions = tuple(int(dimension) for dimension in shape.split("x"))
        except (TypeError, ValueError) as error:
            raise ConversionError(
                f"{label} must use tensor:1x3x224x224 syntax"
            ) from error
        if not _TENSOR_NAME.fullmatch(name) or not dimensions:
            raise ConversionError(f"{label} contains an invalid tensor name or shape")
        if any(dimension <= 0 for dimension in dimensions):
            raise ConversionError(f"{label} dimensions must be greater than zero")
        if name in tensors:
            raise ConversionError(f"{label} contains duplicate tensor {name!r}")
        tensors[name] = dimensions
    return tensors


def validate_profiles(minimum: str, optimum: str, maximum: str) -> None:
    profiles = {
        "min-shapes": parse_profile(minimum, "min-shapes"),
        "opt-shapes": parse_profile(optimum, "opt-shapes"),
        "max-shapes": parse_profile(maximum, "max-shapes"),
    }
    names = set(profiles["min-shapes"])
    if any(set(profile) != names for profile in profiles.values()):
        raise ConversionError("min/opt/max profiles must contain the same tensor names")

    for name in names:
        min_shape = profiles["min-shapes"][name]
        opt_shape = profiles["opt-shapes"][name]
        max_shape = profiles["max-shapes"][name]
        if len({len(min_shape), len(opt_shape), len(max_shape)}) != 1:
            raise ConversionError(f"profile rank differs for tensor {name!r}")
        if any(
            not minimum <= optimum <= maximum
            for minimum, optimum, maximum in zip(min_shape, opt_shape, max_shape)
        ):
            raise ConversionError(f"profile must satisfy min <= opt <= max for {name!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--calibration-cache")
    parser.add_argument("--min-shapes", required=True)
    parser.add_argument("--opt-shapes", required=True)
    parser.add_argument("--max-shapes", required=True)
    args = parser.parse_args()

    try:
        input_path = validated_input(args.input, args.expected_sha256)
        validated_output(args.output, input_path)
        if args.calibration_cache:
            validated_input(args.calibration_cache)
        validate_profiles(args.min_shapes, args.opt_shapes, args.max_shapes)
    except (ConversionError, OSError) as error:
        parser.exit(2, f"ERROR: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
