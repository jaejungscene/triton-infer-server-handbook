import csv
import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).with_name("compare_baseline.py")


def _write_fixture(tmp_path, throughput, p95_latency_us):
    baseline = {
        "models": {
            "sample_model": {
                "min_throughput": 100,
                "max_p95_latency_ms": 50,
                "concurrency": 8,
            }
        }
    }
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    csv_path = tmp_path / "sample_model_perf.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["Concurrency", "Inferences/Second", "p95 latency"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "Concurrency": 8,
                "Inferences/Second": throughput,
                "p95 latency": p95_latency_us,
            }
        )
    return baseline_path


def _run_compare(tmp_path, baseline_path):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--baseline",
            str(baseline_path),
            "--results-dir",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_compare_passes_within_baseline(tmp_path):
    baseline_path = _write_fixture(tmp_path, throughput=120, p95_latency_us=45_000)

    result = _run_compare(tmp_path, baseline_path)

    assert result.returncode == 0
    assert "within baseline" in result.stdout


def test_compare_fails_for_throughput_and_latency_regression(tmp_path):
    baseline_path = _write_fixture(tmp_path, throughput=80, p95_latency_us=60_000)

    result = _run_compare(tmp_path, baseline_path)

    assert result.returncode == 1
    assert "throughput" in result.stderr
    assert "p95" in result.stderr
