"""Shared validation and atomic-output helpers for model conversion scripts."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import hmac
import os
from pathlib import Path
import re
import tempfile


class ConversionError(ValueError):
    """Expected conversion input or artifact contract failure."""


def validated_input(path_value: str, expected_sha256: str | None = None) -> Path:
    path = Path(path_value).expanduser()
    if path.is_symlink() or not path.is_file():
        raise ConversionError(f"input must be a regular file, not a symlink: {path}")
    path = path.resolve()
    if expected_sha256:
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ConversionError("expected SHA-256 must be 64 lowercase hex characters")
        digest = hashlib.sha256()
        with path.open("rb") as input_file:
            for chunk in iter(lambda: input_file.read(64 * 1024), b""):
                digest.update(chunk)
        if not hmac.compare_digest(digest.hexdigest(), expected_sha256):
            raise ConversionError(f"SHA-256 mismatch for input: {path}")
    return path


def validated_output(path_value: str, input_path: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_symlink():
        raise ConversionError(f"output must not be a symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved = path.resolve(strict=False)
    if resolved == input_path:
        raise ConversionError("input and output paths must be different")
    return resolved


def parse_shape(value: str) -> list[int]:
    try:
        dimensions = [int(dimension) for dimension in value.split(",")]
    except (AttributeError, ValueError) as exc:
        raise ConversionError("input shape must be comma-separated integers") from exc
    if not dimensions or any(dimension <= 0 for dimension in dimensions):
        raise ConversionError("input shape dimensions must be greater than zero")
    return dimensions


@contextmanager
def atomic_output(output_path: Path):
    """Yield a same-filesystem temporary path and publish only a non-empty result."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}-",
        suffix=output_path.suffix or ".tmp",
        dir=output_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        yield temporary_path
        if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
            raise ConversionError("converter produced an empty output artifact")
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_pytorch_module(torch, model_path: Path, allow_unsafe_pickle: bool):
    """Load TorchScript by default and gate pickle-based eager modules explicitly."""
    try:
        return torch.jit.load(str(model_path), map_location="cpu")
    except (RuntimeError, ValueError) as jit_error:
        if not allow_unsafe_pickle:
            raise ConversionError(
                "input is not TorchScript; pass --allow-unsafe-pickle only for a "
                "trusted full-module checkpoint"
            ) from jit_error

    print("[convert] WARNING: loading a trusted pickle checkpoint")
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(model, dict) and "model" in model:
        model = model["model"]
    if not hasattr(model, "eval"):
        raise ConversionError("checkpoint does not contain an executable PyTorch module")
    return model
