import os
from pathlib import Path
import subprocess


def _fake_trtexec(tmp_path, exit_code=0):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable = bin_dir / "trtexec"
    executable.write_text(
        """#!/usr/bin/env bash
set -eu
printf '%s\n' "$@" > "${TRTEXEC_ARGS_FILE}"
for argument in "$@"; do
    case "${argument}" in
        --saveEngine=*) output="${argument#*=}" ;;
    esac
done
printf 'engine' > "${output}"
exit "${TRTEXEC_EXIT_CODE:-0}"
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return bin_dir


def _run_converter(project_root, tmp_path, *args, exit_code=0):
    arguments_file = tmp_path / "arguments.txt"
    bin_dir = _fake_trtexec(tmp_path, exit_code)
    environment = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "TRTEXEC_ARGS_FILE": str(arguments_file),
        "TRTEXEC_EXIT_CODE": str(exit_code),
    }
    result = subprocess.run(
        [
            str(Path(project_root) / "scripts" / "convert" / "to_tensorrt.sh"),
            *args,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    return result, arguments_file


def test_tensorrt_converter_uses_validated_profile_and_atomic_output(
    project_root, tmp_path
):
    source = tmp_path / "model.onnx"
    output = tmp_path / "model.plan"
    source.write_bytes(b"onnx")

    result, arguments_file = _run_converter(
        project_root,
        tmp_path,
        "--input",
        str(source),
        "--output",
        str(output),
        "--precision",
        "fp16",
        "--min-shapes",
        "images:1x3x224x224",
        "--opt-shapes",
        "images:4x3x224x224",
        "--max-shapes",
        "images:8x3x224x224",
    )

    arguments = arguments_file.read_text(encoding="utf-8")
    assert result.returncode == 0
    assert output.read_bytes() == b"engine"
    assert "--memPoolSize=workspace:4096" in arguments
    assert "--minShapes=images:1x3x224x224" in arguments
    assert "--buildOnly" in arguments


def test_tensorrt_converter_rejects_inverted_profile_before_build(
    project_root, tmp_path
):
    source = tmp_path / "model.onnx"
    output = tmp_path / "model.plan"
    source.write_bytes(b"onnx")
    output.write_bytes(b"previous")

    result, arguments_file = _run_converter(
        project_root,
        tmp_path,
        "--input",
        str(source),
        "--output",
        str(output),
        "--min-shapes",
        "input:8x3x224x224",
        "--opt-shapes",
        "input:4x3x224x224",
        "--max-shapes",
        "input:8x3x224x224",
    )

    assert result.returncode == 2
    assert "min <= opt <= max" in result.stderr
    assert output.read_bytes() == b"previous"
    assert not arguments_file.exists()


def test_tensorrt_converter_preserves_previous_engine_when_trtexec_fails(
    project_root, tmp_path
):
    source = tmp_path / "model.onnx"
    output = tmp_path / "model.plan"
    source.write_bytes(b"onnx")
    output.write_bytes(b"previous")

    result, _ = _run_converter(
        project_root,
        tmp_path,
        "--input",
        str(source),
        "--output",
        str(output),
        exit_code=1,
    )

    assert result.returncode == 1
    assert output.read_bytes() == b"previous"
    assert list(tmp_path.glob(".model.plan.*")) == []
