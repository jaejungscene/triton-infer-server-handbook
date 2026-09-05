#!/usr/bin/env python3
"""Create and verify performance evidence bound to an immutable release image."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile


CHUNK_SIZE = 64 * 1024
SOURCE_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
IMAGE_REF_PATTERN = re.compile(
    r"^[a-z0-9.-]+(?::[0-9]+)?/[A-Za-z0-9._/-]+@sha256:[0-9a-f]{64}$"
)
REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
MODEL_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
RESULT_PATTERN = re.compile(r"^([A-Za-z0-9._-]+)_perf\.csv$")
WORKFLOW_FILE = ".github/workflows/perf-benchmark.yml"
FULL_SCOPE = "all-ready-models"


class EvidenceError(ValueError):
    """Invalid performance evidence or evidence input."""


def _exact_keys(document: dict, expected: set[str], field: str) -> None:
    actual = set(document)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise EvidenceError(
            f"{field} fields mismatch; missing={missing}, unknown={unknown}"
        )


def _sha256(path: Path, field: str) -> tuple[int, str]:
    if not path.is_file() or path.is_symlink():
        raise EvidenceError(f"{field} must be a regular file: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(CHUNK_SIZE):
            size += len(chunk)
            digest.update(chunk)
    if size == 0:
        raise EvidenceError(f"{field} must not be empty: {path}")
    return size, digest.hexdigest()


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise EvidenceError(f"{field} must be a positive integer")
    return value


def _safe_text(value: object, field: str, *, maximum: int = 256) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum
        or any(character in value for character in "\r\n\0")
    ):
        raise EvidenceError(f"{field} must be a non-empty single-line string")
    return value


def _validate_identity(
    source_revision: object,
    image_ref: object,
    image_digest: object,
    repository: object,
) -> tuple[str, str, str, str]:
    if (
        not isinstance(source_revision, str)
        or SOURCE_REVISION_PATTERN.fullmatch(source_revision) is None
    ):
        raise EvidenceError("source_revision must be a lowercase 40-character Git SHA")
    if not isinstance(image_digest, str) or DIGEST_PATTERN.fullmatch(image_digest) is None:
        raise EvidenceError("image_digest must be a sha256 digest")
    if not isinstance(image_ref, str) or IMAGE_REF_PATTERN.fullmatch(image_ref) is None:
        raise EvidenceError("image_ref must be an immutable registry reference")
    if not image_ref.endswith(f"@{image_digest}"):
        raise EvidenceError("image_ref does not match image_digest")
    if not isinstance(repository, str) or REPOSITORY_PATTERN.fullmatch(repository) is None:
        raise EvidenceError("repository must be an owner/name pair")
    return source_revision, image_ref, image_digest, repository


def _collect_results(results_dir: Path) -> list[dict]:
    if not results_dir.is_dir() or results_dir.is_symlink():
        raise EvidenceError(f"results directory is invalid: {results_dir}")

    entries = []
    models: set[str] = set()
    for path in sorted(results_dir.glob("*_perf.csv")):
        match = RESULT_PATTERN.fullmatch(path.name)
        if match is None or match.group(1) in {".", ".."}:
            raise EvidenceError(f"invalid performance result filename: {path.name}")
        model = match.group(1)
        if model in models:
            raise EvidenceError(f"duplicate performance result model: {model}")
        models.add(model)
        size, digest = _sha256(path, f"result {model}")
        entries.append(
            {
                "model": model,
                "file": path.name,
                "size_bytes": size,
                "sha256": digest,
            }
        )
    if not entries:
        raise EvidenceError("at least one performance result CSV is required")
    return entries


def _load_gpu_inventory(path: Path) -> list[str]:
    _sha256(path, "GPU inventory")
    inventory = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    if not inventory:
        raise EvidenceError("GPU inventory must contain at least one GPU")
    return [
        _safe_text(line, f"GPU inventory line {index}")
        for index, line in enumerate(inventory, start=1)
    ]


def create_evidence(
    *,
    source_revision: str,
    image_ref: str,
    image_digest: str,
    repository: str,
    workflow_run_id: int,
    workflow_run_attempt: int,
    workflow_revision: str,
    runner_name: str,
    gpu_inventory_file: Path,
    baseline: Path,
    profiles: Path,
    results_dir: Path,
) -> dict:
    source_revision, image_ref, image_digest, repository = _validate_identity(
        source_revision, image_ref, image_digest, repository
    )
    if (
        not isinstance(workflow_revision, str)
        or SOURCE_REVISION_PATTERN.fullmatch(workflow_revision) is None
    ):
        raise EvidenceError("workflow_revision must be a lowercase 40-character Git SHA")
    _, baseline_sha256 = _sha256(baseline, "baseline")
    _, profiles_sha256 = _sha256(profiles, "profiles")
    return {
        "schema_version": 1,
        "status": "passed",
        "source_revision": source_revision,
        "image": {
            "ref": image_ref,
            "digest": image_digest,
        },
        "benchmark": {
            "scope": FULL_SCOPE,
            "baseline_sha256": baseline_sha256,
            "profiles_sha256": profiles_sha256,
            "results": _collect_results(results_dir),
        },
        "runner": {
            "name": _safe_text(runner_name, "runner name"),
            "gpu_inventory": _load_gpu_inventory(gpu_inventory_file),
        },
        "github": {
            "repository": repository,
            "workflow_file": WORKFLOW_FILE,
            "workflow_revision": workflow_revision,
            "run_id": _positive_int(workflow_run_id, "workflow run ID"),
            "run_attempt": _positive_int(
                workflow_run_attempt, "workflow run attempt"
            ),
        },
    }


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    document = {}
    for key, value in pairs:
        if key in document:
            raise EvidenceError(f"duplicate JSON key: {key}")
        document[key] = value
    return document


def _load_evidence(path: Path) -> dict:
    if not path.is_file() or path.is_symlink():
        raise EvidenceError(f"evidence must be a regular file: {path}")
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except json.JSONDecodeError as error:
        raise EvidenceError(f"invalid evidence JSON: {error}") from error
    if not isinstance(document, dict):
        raise EvidenceError("evidence root must be an object")
    return document


def verify_evidence(
    *,
    evidence_path: Path,
    expected_source_revision: str,
    expected_image_ref: str,
    expected_image_digest: str,
    expected_repository: str,
    expected_workflow_run_id: int,
    expected_workflow_run_attempt: int,
    expected_workflow_revision: str,
    baseline: Path,
    profiles: Path,
    results_dir: Path,
) -> dict:
    _validate_identity(
        expected_source_revision,
        expected_image_ref,
        expected_image_digest,
        expected_repository,
    )
    document = _load_evidence(evidence_path)
    _exact_keys(
        document,
        {
            "schema_version",
            "status",
            "source_revision",
            "image",
            "benchmark",
            "runner",
            "github",
        },
        "evidence",
    )
    if document["schema_version"] != 1 or document["status"] != "passed":
        raise EvidenceError("evidence must use schema version 1 with passed status")

    image = document["image"]
    benchmark = document["benchmark"]
    runner = document["runner"]
    github = document["github"]
    if not all(isinstance(item, dict) for item in (image, benchmark, runner, github)):
        raise EvidenceError("image, benchmark, runner, and github must be objects")
    _exact_keys(image, {"ref", "digest"}, "image")
    _exact_keys(
        benchmark,
        {"scope", "baseline_sha256", "profiles_sha256", "results"},
        "benchmark",
    )
    _exact_keys(runner, {"name", "gpu_inventory"}, "runner")
    _exact_keys(
        github,
        {
            "repository",
            "workflow_file",
            "workflow_revision",
            "run_id",
            "run_attempt",
        },
        "github",
    )

    _validate_identity(
        document["source_revision"], image["ref"], image["digest"], github["repository"]
    )
    if document["source_revision"] != expected_source_revision:
        raise EvidenceError("performance evidence source revision mismatch")
    if image["ref"] != expected_image_ref or image["digest"] != expected_image_digest:
        raise EvidenceError("performance evidence image digest mismatch")
    if github["repository"] != expected_repository:
        raise EvidenceError("performance evidence repository mismatch")
    if github["workflow_file"] != WORKFLOW_FILE:
        raise EvidenceError("performance evidence workflow mismatch")
    if (
        not isinstance(github["workflow_revision"], str)
        or SOURCE_REVISION_PATTERN.fullmatch(github["workflow_revision"]) is None
        or github["workflow_revision"] != expected_workflow_revision
    ):
        raise EvidenceError("performance evidence workflow revision mismatch")
    if _positive_int(github["run_id"], "workflow run ID") != expected_workflow_run_id:
        raise EvidenceError("performance evidence workflow run mismatch")
    if (
        _positive_int(github["run_attempt"], "workflow run attempt")
        != expected_workflow_run_attempt
    ):
        raise EvidenceError("performance evidence workflow run attempt mismatch")

    if benchmark["scope"] != FULL_SCOPE:
        raise EvidenceError("only all-ready-models evidence can gate production")
    _, baseline_sha256 = _sha256(baseline, "baseline")
    _, profiles_sha256 = _sha256(profiles, "profiles")
    if benchmark["baseline_sha256"] != baseline_sha256:
        raise EvidenceError("performance evidence baseline hash mismatch")
    if benchmark["profiles_sha256"] != profiles_sha256:
        raise EvidenceError("performance evidence profiles hash mismatch")

    _safe_text(runner["name"], "runner name")
    gpu_inventory = runner["gpu_inventory"]
    if not isinstance(gpu_inventory, list) or not gpu_inventory:
        raise EvidenceError("runner.gpu_inventory must be a non-empty list")
    for index, line in enumerate(gpu_inventory, start=1):
        _safe_text(line, f"GPU inventory line {index}")

    expected_results = _collect_results(results_dir)
    results = benchmark["results"]
    if not isinstance(results, list) or not results:
        raise EvidenceError("benchmark.results must be a non-empty list")
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            raise EvidenceError(f"benchmark.results[{index}] must be an object")
        _exact_keys(result, {"model", "file", "size_bytes", "sha256"}, f"result {index}")
        model = result["model"]
        if (
            not isinstance(model, str)
            or model in {".", ".."}
            or MODEL_PATTERN.fullmatch(model) is None
            or result["file"] != f"{model}_perf.csv"
        ):
            raise EvidenceError(f"benchmark.results[{index}] has invalid model or file")
        _positive_int(result["size_bytes"], f"result {model} size")
        if not isinstance(result["sha256"], str) or re.fullmatch(
            r"[0-9a-f]{64}", result["sha256"]
        ) is None:
            raise EvidenceError(f"result {model} has invalid SHA-256")
    if results != expected_results:
        raise EvidenceError("performance result files do not match evidence")
    return document


def _write_evidence(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(document, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_identity(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--source-revision", required=True)
        subparser.add_argument("--image-ref", required=True)
        subparser.add_argument("--image-digest", required=True)
        subparser.add_argument("--repository", required=True)
        subparser.add_argument("--workflow-run-id", type=int, required=True)
        subparser.add_argument("--baseline", type=Path, required=True)
        subparser.add_argument("--profiles", type=Path, required=True)
        subparser.add_argument("--results-dir", type=Path, required=True)

    create_parser = subparsers.add_parser("create")
    add_identity(create_parser)
    create_parser.add_argument("--workflow-run-attempt", type=int, required=True)
    create_parser.add_argument("--workflow-revision", required=True)
    create_parser.add_argument("--runner-name", required=True)
    create_parser.add_argument("--gpu-inventory-file", type=Path, required=True)
    create_parser.add_argument("--output", type=Path, required=True)

    verify_parser = subparsers.add_parser("verify")
    add_identity(verify_parser)
    verify_parser.add_argument("--workflow-run-attempt", type=int, required=True)
    verify_parser.add_argument("--workflow-revision", required=True)
    verify_parser.add_argument("--evidence", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "create":
            document = create_evidence(
                source_revision=args.source_revision,
                image_ref=args.image_ref,
                image_digest=args.image_digest,
                repository=args.repository,
                workflow_run_id=args.workflow_run_id,
                workflow_run_attempt=args.workflow_run_attempt,
                workflow_revision=args.workflow_revision,
                runner_name=args.runner_name,
                gpu_inventory_file=args.gpu_inventory_file,
                baseline=args.baseline,
                profiles=args.profiles,
                results_dir=args.results_dir,
            )
            _write_evidence(args.output, document)
            print(f"Performance evidence created: {args.output}")
        else:
            verify_evidence(
                evidence_path=args.evidence,
                expected_source_revision=args.source_revision,
                expected_image_ref=args.image_ref,
                expected_image_digest=args.image_digest,
                expected_repository=args.repository,
                expected_workflow_run_id=args.workflow_run_id,
                expected_workflow_run_attempt=args.workflow_run_attempt,
                expected_workflow_revision=args.workflow_revision,
                baseline=args.baseline,
                profiles=args.profiles,
                results_dir=args.results_dir,
            )
            print(
                "Performance evidence verified for "
                f"{args.source_revision} at {args.image_digest}"
            )
    except (EvidenceError, OSError, UnicodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
