import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).with_name("release_evidence.py")
SOURCE_REVISION = "a" * 40
IMAGE_DIGEST = f"sha256:{'b' * 64}"
IMAGE_REF = f"ghcr.io/example/project@{IMAGE_DIGEST}"
REPOSITORY = "example/project"
RUN_ID = "12345"
WORKFLOW_REVISION = "d" * 40


def _fixtures(tmp_path):
    baseline = tmp_path / "baseline.json"
    profiles = tmp_path / "profiles.json"
    results = tmp_path / "results"
    inventory = tmp_path / "gpu-inventory.txt"
    evidence = results / "release-performance-evidence.json"
    baseline.write_text('{"models":{"example":{}}}\n', encoding="utf-8")
    profiles.write_text('{"models":{"example":{}}}\n', encoding="utf-8")
    results.mkdir()
    (results / "example_perf.csv").write_text(
        "Concurrency,Inferences/Second,p95 latency\n8,120,45000\n",
        encoding="utf-8",
    )
    inventory.write_text("NVIDIA A100, 550.54.15\n", encoding="utf-8")
    return baseline, profiles, results, inventory, evidence


def _create_command(baseline, profiles, results, inventory, evidence):
    return [
        sys.executable,
        str(SCRIPT),
        "create",
        "--source-revision",
        SOURCE_REVISION,
        "--image-ref",
        IMAGE_REF,
        "--image-digest",
        IMAGE_DIGEST,
        "--repository",
        REPOSITORY,
        "--workflow-run-id",
        RUN_ID,
        "--workflow-run-attempt",
        "2",
        "--workflow-revision",
        WORKFLOW_REVISION,
        "--runner-name",
        "gpu-runner-1",
        "--gpu-inventory-file",
        str(inventory),
        "--baseline",
        str(baseline),
        "--profiles",
        str(profiles),
        "--results-dir",
        str(results),
        "--output",
        str(evidence),
    ]


def _verify_command(
    baseline,
    profiles,
    results,
    evidence,
    *,
    digest=IMAGE_DIGEST,
    run_attempt="2",
    workflow_revision=WORKFLOW_REVISION,
):
    return [
        sys.executable,
        str(SCRIPT),
        "verify",
        "--source-revision",
        SOURCE_REVISION,
        "--image-ref",
        f"ghcr.io/example/project@{digest}",
        "--image-digest",
        digest,
        "--repository",
        REPOSITORY,
        "--workflow-run-id",
        RUN_ID,
        "--workflow-run-attempt",
        run_attempt,
        "--workflow-revision",
        workflow_revision,
        "--baseline",
        str(baseline),
        "--profiles",
        str(profiles),
        "--results-dir",
        str(results),
        "--evidence",
        str(evidence),
    ]


def test_create_and_verify_exact_release_evidence(tmp_path):
    baseline, profiles, results, inventory, evidence = _fixtures(tmp_path)

    created = subprocess.run(
        _create_command(baseline, profiles, results, inventory, evidence),
        check=False,
        capture_output=True,
        text=True,
    )
    verified = subprocess.run(
        _verify_command(baseline, profiles, results, evidence),
        check=False,
        capture_output=True,
        text=True,
    )

    assert created.returncode == 0, created.stderr
    assert verified.returncode == 0, verified.stderr
    document = json.loads(evidence.read_text(encoding="utf-8"))
    assert document["source_revision"] == SOURCE_REVISION
    assert document["image"] == {"digest": IMAGE_DIGEST, "ref": IMAGE_REF}
    assert document["benchmark"]["scope"] == "all-ready-models"
    assert document["benchmark"]["results"][0]["model"] == "example"
    assert document["github"]["run_id"] == int(RUN_ID)
    assert document["github"]["workflow_revision"] == WORKFLOW_REVISION


def test_verify_rejects_a_different_release_digest(tmp_path):
    baseline, profiles, results, inventory, evidence = _fixtures(tmp_path)
    subprocess.run(
        _create_command(baseline, profiles, results, inventory, evidence), check=True
    )
    other_digest = f"sha256:{'c' * 64}"

    verified = subprocess.run(
        _verify_command(
            baseline, profiles, results, evidence, digest=other_digest
        ),
        check=False,
        capture_output=True,
        text=True,
    )

    assert verified.returncode == 1
    assert "image digest mismatch" in verified.stderr


def test_verify_rejects_tampered_result_bytes(tmp_path):
    baseline, profiles, results, inventory, evidence = _fixtures(tmp_path)
    subprocess.run(
        _create_command(baseline, profiles, results, inventory, evidence), check=True
    )
    with (results / "example_perf.csv").open("a", encoding="utf-8") as result_file:
        result_file.write("8,9999,1\n")

    verified = subprocess.run(
        _verify_command(baseline, profiles, results, evidence),
        check=False,
        capture_output=True,
        text=True,
    )

    assert verified.returncode == 1
    assert "result files do not match evidence" in verified.stderr


def test_verify_rejects_changed_baseline(tmp_path):
    baseline, profiles, results, inventory, evidence = _fixtures(tmp_path)
    subprocess.run(
        _create_command(baseline, profiles, results, inventory, evidence), check=True
    )
    baseline.write_text('{"models":{"example":{"changed":true}}}\n', encoding="utf-8")

    verified = subprocess.run(
        _verify_command(baseline, profiles, results, evidence),
        check=False,
        capture_output=True,
        text=True,
    )

    assert verified.returncode == 1
    assert "baseline hash mismatch" in verified.stderr


def test_verify_rejects_different_workflow_attempt(tmp_path):
    baseline, profiles, results, inventory, evidence = _fixtures(tmp_path)
    subprocess.run(
        _create_command(baseline, profiles, results, inventory, evidence), check=True
    )

    verified = subprocess.run(
        _verify_command(
            baseline, profiles, results, evidence, run_attempt="3"
        ),
        check=False,
        capture_output=True,
        text=True,
    )

    assert verified.returncode == 1
    assert "workflow run attempt mismatch" in verified.stderr


def test_verify_rejects_different_workflow_revision(tmp_path):
    baseline, profiles, results, inventory, evidence = _fixtures(tmp_path)
    subprocess.run(
        _create_command(baseline, profiles, results, inventory, evidence), check=True
    )

    verified = subprocess.run(
        _verify_command(
            baseline,
            profiles,
            results,
            evidence,
            workflow_revision="e" * 40,
        ),
        check=False,
        capture_output=True,
        text=True,
    )

    assert verified.returncode == 1
    assert "workflow revision mismatch" in verified.stderr


def test_create_requires_at_least_one_result(tmp_path):
    baseline, profiles, results, inventory, evidence = _fixtures(tmp_path)
    (results / "example_perf.csv").unlink()

    created = subprocess.run(
        _create_command(baseline, profiles, results, inventory, evidence),
        check=False,
        capture_output=True,
        text=True,
    )

    assert created.returncode == 1
    assert "at least one performance result CSV" in created.stderr
