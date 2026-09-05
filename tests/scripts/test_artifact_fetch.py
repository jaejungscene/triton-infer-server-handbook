import hashlib
import importlib.util
import json
import subprocess
import sys

import yaml


def _artifact_manifest(uri, payload, *, checksum=None, path="1/model.bin"):
    return {
        "models": [
            {
                "source": "example",
                "target": "example",
                "enabled": True,
                "artifact": "external",
                "required_files": [path],
                "artifacts": [
                    {
                        "path": path,
                        "uri": uri,
                        "sha256": checksum or hashlib.sha256(payload).hexdigest(),
                        "size_bytes": len(payload),
                    }
                ],
            }
        ]
    }


def _run_fetch(project_root, manifest, output, *args):
    return subprocess.run(
        [
            sys.executable,
            f"{project_root}/scripts/fetch_artifacts.py",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output),
            *args,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_fetches_local_fixture_and_writes_integrity_receipt(project_root, tmp_path):
    payload = b"versioned-model-artifact"
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = source_root / "model.bin"
    source.write_bytes(payload)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(_artifact_manifest(source.as_uri(), payload)),
        encoding="utf-8",
    )
    output = tmp_path / "artifacts"

    result = _run_fetch(
        project_root,
        manifest,
        output,
        "--env",
        "prod",
        "--local-root",
        str(source_root),
    )

    assert result.returncode == 0, result.stderr
    assert (output / "example" / "1" / "model.bin").read_bytes() == payload
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["schema_version"] == 1
    assert receipt["environment"] == "prod"
    assert receipt["artifacts"] == [
        {
            "path": "1/model.bin",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
            "target": "example",
            "uri": source.as_uri(),
        }
    ]


def test_checksum_failure_preserves_previous_verified_cache(project_root, tmp_path):
    payload = b"tampered"
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = source_root / "model.bin"
    source.write_bytes(payload)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(_artifact_manifest(source.as_uri(), payload, checksum="0" * 64)),
        encoding="utf-8",
    )
    output = tmp_path / "artifacts"
    output.mkdir()
    marker = output / "previous-receipt.json"
    marker.write_text("trusted", encoding="utf-8")

    result = _run_fetch(
        project_root,
        manifest,
        output,
        "--local-root",
        str(source_root),
    )

    assert result.returncode == 1
    assert "SHA-256 mismatch" in result.stderr
    assert marker.read_text(encoding="utf-8") == "trusted"
    assert not (output / "example").exists()


def test_rejects_insecure_or_non_allowlisted_remote_uris(project_root, tmp_path):
    payload = b"artifact"
    manifest = tmp_path / "manifest.yaml"
    output = tmp_path / "artifacts"

    manifest.write_text(
        yaml.safe_dump(_artifact_manifest("http://models.example/model.bin", payload)),
        encoding="utf-8",
    )
    insecure = _run_fetch(
        project_root, manifest, output, "--allowed-hosts", "models.example"
    )
    assert insecure.returncode == 1
    assert "must be an HTTPS URL" in insecure.stderr

    manifest.write_text(
        yaml.safe_dump(_artifact_manifest("https://other.example/model.bin", payload)),
        encoding="utf-8",
    )
    disallowed = _run_fetch(
        project_root, manifest, output, "--allowed-hosts", "models.example"
    )
    assert disallowed.returncode == 1
    assert "not allowlisted" in disallowed.stderr


def test_sanitized_remote_uri_omits_query_from_receipt(project_root):
    module_name = "artifact_fetcher_under_test"
    spec = importlib.util.spec_from_file_location(
        module_name, f"{project_root}/scripts/fetch_artifacts.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)

    hostname, public_uri = module._safe_https_uri(
        "https://models.example/releases/model.bin?signature=secret",
        {"models.example"},
    )

    assert hostname == "models.example"
    assert public_uri == "https://models.example/releases/model.bin"


def test_rejects_artifact_path_escape_before_fetch(project_root, tmp_path):
    payload = b"artifact"
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            _artifact_manifest(
                "https://models.example/model.bin", payload, path="../model.bin"
            )
        ),
        encoding="utf-8",
    )

    result = _run_fetch(
        project_root,
        manifest,
        tmp_path / "artifacts",
        "--allowed-hosts",
        "models.example",
    )

    assert result.returncode == 1
    assert "normalized relative path" in result.stderr


def test_rejects_unsafe_target_before_writing_cache(project_root, tmp_path):
    payload = b"artifact"
    data = _artifact_manifest("https://models.example/model.bin", payload)
    data["models"][0]["target"] = ".."
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(yaml.safe_dump(data), encoding="utf-8")

    result = _run_fetch(
        project_root,
        manifest,
        tmp_path / "artifacts",
        "--allowed-hosts",
        "models.example",
    )

    assert result.returncode == 1
    assert "target must be a safe non-empty name" in result.stderr


def test_rejects_local_uri_query_and_duplicate_required_files(project_root, tmp_path):
    payload = b"artifact"
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = source_root / "model.bin"
    source.write_bytes(payload)
    manifest = tmp_path / "manifest.yaml"

    query_data = _artifact_manifest(f"{source.as_uri()}?token=secret", payload)
    manifest.write_text(yaml.safe_dump(query_data), encoding="utf-8")
    query_result = _run_fetch(
        project_root,
        manifest,
        tmp_path / "query-artifacts",
        "--local-root",
        str(source_root),
    )
    assert query_result.returncode == 1
    assert "without query or fragment" in query_result.stderr

    duplicate_data = _artifact_manifest(source.as_uri(), payload)
    duplicate_data["models"][0]["required_files"].append("1/model.bin")
    manifest.write_text(yaml.safe_dump(duplicate_data), encoding="utf-8")
    duplicate_result = _run_fetch(
        project_root,
        manifest,
        tmp_path / "duplicate-artifacts",
        "--local-root",
        str(source_root),
    )
    assert duplicate_result.returncode == 1
    assert "required_files must not contain duplicates" in duplicate_result.stderr


def test_external_artifacts_must_cover_required_files(project_root, tmp_path):
    payload = b"artifact"
    data = _artifact_manifest("https://models.example/model.bin", payload)
    data["models"][0]["required_files"].append("1/labels.txt")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(yaml.safe_dump(data), encoding="utf-8")

    result = _run_fetch(
        project_root,
        manifest,
        tmp_path / "artifacts",
        "--allowed-hosts",
        "models.example",
    )

    assert result.returncode == 1
    assert "must exactly match required_files" in result.stderr


def test_no_selected_external_models_still_produces_receipt(project_root, tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {
                        "source": "local",
                        "target": "local",
                        "enabled": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "artifacts"

    result = _run_fetch(project_root, manifest, output, "--env", "prod")

    assert result.returncode == 0, result.stderr
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["artifacts"] == []
