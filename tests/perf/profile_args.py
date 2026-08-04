#!/usr/bin/env python3
"""Render trusted perf_analyzer arguments from a versioned model profile."""

import argparse
import json
from pathlib import Path


def profile_arguments(profiles_path: Path, model: str, perf_dir: Path) -> list[str]:
    with profiles_path.open(encoding="utf-8") as profiles_file:
        document = json.load(profiles_file)

    profile = document.get("models", {}).get(model)
    if not isinstance(profile, dict):
        raise ValueError(f"No performance profile configured for model: {model}")

    input_data = profile.get("input_data")
    batch_size = profile.get("batch_size")
    if not isinstance(input_data, str) or not input_data:
        raise ValueError(f"Profile {model} must define input_data")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"Profile {model} must define a positive batch_size")

    if input_data not in {"zero", "random"}:
        input_path = (perf_dir / input_data).resolve()
        if not input_path.is_relative_to(perf_dir.resolve()) or not input_path.is_file():
            raise ValueError(f"Profile {model} input_data does not exist: {input_data}")
        input_data = str(input_path)

    arguments = ["--input-data", input_data, "--batch-size", str(batch_size)]
    shapes = profile.get("shapes", [])
    if not isinstance(shapes, list) or not all(
        isinstance(shape, str) and shape for shape in shapes
    ):
        raise ValueError(f"Profile {model} shapes must be a string list")
    for shape in shapes:
        arguments.extend(["--shape", shape])
    return arguments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--perf-dir", type=Path, required=True)
    args = parser.parse_args()

    try:
        arguments = profile_arguments(args.profiles, args.model, args.perf_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    print("\n".join(arguments))


if __name__ == "__main__":
    main()
