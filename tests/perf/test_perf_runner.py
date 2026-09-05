import os
from pathlib import Path
import subprocess


def _fake_perf_analyzer(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable = bin_dir / "perf_analyzer"
    executable.write_text(
        """#!/usr/bin/env bash
set -eu
printf '%s\n' "$@" > "${PERF_ARGS_FILE}"
while [[ $# -gt 0 ]]; do
    if [[ "$1" == "-f" ]]; then
        output="$2"
        shift 2
    else
        shift
    fi
done
if [[ "${PERF_SKIP_OUTPUT:-false}" != "true" ]]; then
    printf 'Concurrency,Inferences/Second,p95 latency\n8,100,1000\n' > "${output}"
fi
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return bin_dir


def _run(project_root, tmp_path, *args, token=None, extra_env=None):
    bin_dir = _fake_perf_analyzer(tmp_path)
    arguments_file = tmp_path / "perf-arguments.txt"
    environment = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "PERF_ARGS_FILE": str(arguments_file),
        "PERF_RESULTS_DIR": str(tmp_path / "results"),
    }
    if token is not None:
        environment["TRITON_AUTH_TOKEN"] = token
    if extra_env:
        environment.update(extra_env)
    result = subprocess.run(
        [
            str(Path(project_root) / "tests" / "perf" / "run_perf_analyzer.sh"),
            "--model",
            "text_classifier",
            *args,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    return result, arguments_file


def test_perf_runner_validates_target_and_passes_auth_header(project_root, tmp_path):
    result, arguments_file = _run(
        project_root,
        tmp_path,
        "--url",
        "http://localhost:8000",
        token="test-token",
    )

    arguments = arguments_file.read_text(encoding="utf-8")
    assert result.returncode == 0
    assert "localhost:8000" in arguments
    assert "Authorization: Bearer test-token" in arguments


def test_perf_runner_rejects_unsafe_model_before_execution(project_root, tmp_path):
    bin_dir = _fake_perf_analyzer(tmp_path)
    environment = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}

    result = subprocess.run(
        [
            str(Path(project_root) / "tests" / "perf" / "run_perf_analyzer.sh"),
            "--model",
            "../../result",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 2
    assert "model must contain only" in result.stderr


def test_perf_runner_rejects_token_with_line_break(project_root, tmp_path):
    result, arguments_file = _run(
        project_root,
        tmp_path,
        token="token\nX-Forged: true",
    )

    assert result.returncode == 2
    assert "must not contain line breaks" in result.stderr
    assert not arguments_file.exists()


def test_perf_runner_requires_a_non_empty_result_csv(project_root, tmp_path):
    result, arguments_file = _run(
        project_root,
        tmp_path,
        extra_env={"PERF_SKIP_OUTPUT": "true"},
    )

    assert arguments_file.exists()
    assert result.returncode == 1
    assert "did not create a regular non-empty CSV" in result.stderr
