import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def _run_build(project_root, *args):
    return subprocess.run(
        [f"{project_root}/scripts/build.sh", *args],
        check=False,
        capture_output=True,
        text=True,
    )


def _minimal_project(tmp_path, build_script):
    project = tmp_path / "project"
    scripts_dir = project / "scripts"
    source_dir = project / "models" / "serving" / "example"
    version_dir = source_dir / "1"
    repository = project / "model_repository"
    scripts_dir.mkdir(parents=True)
    version_dir.mkdir(parents=True)
    repository.mkdir()
    shutil.copy2(build_script, scripts_dir / "build.sh")
    shutil.copy2(build_script.parent / "fetch_artifacts.py", scripts_dir)
    (source_dir / "config.pbtxt").write_text(
        'name: "example"\nbackend: "python"\n', encoding="utf-8"
    )
    (version_dir / "model.py").write_text("# test model\n", encoding="utf-8")
    manifest = {
        "models": [
            {
                "source": "example",
                "target": "example",
                "enabled": True,
                "required_files": ["1/model.py"],
            }
        ]
    }
    (project / "models" / "serving" / "manifest.yaml").write_text(
        yaml.safe_dump(manifest), encoding="utf-8"
    )
    return project


def test_build_creates_enabled_runtime_model(project_root):
    result = _run_build(project_root, "--env", "dev", "--clean")

    model_path = Path(project_root) / "model_repository" / "text_classifier"
    assert result.returncode == 0
    assert (model_path / "config.pbtxt").is_file()
    assert (model_path / "1" / "model.py").is_file()
    assert "fallback" not in result.stderr.lower()


def test_empty_selection_fails_without_cleaning_existing_repository(project_root):
    preserved_path = Path(project_root) / "model_repository" / "preserve-me"
    preserved_path.mkdir(exist_ok=True)
    marker = preserved_path / "marker"
    marker.write_text("preserve", encoding="utf-8")
    try:
        result = _run_build(
            project_root, "--env", "prod", "--tags", "does-not-exist", "--clean"
        )

        assert result.returncode == 1
        assert marker.read_text(encoding="utf-8") == "preserve"
        assert "no enabled models" in result.stderr
    finally:
        shutil.rmtree(preserved_path)


def test_build_rejects_unknown_environment(project_root):
    result = _run_build(project_root, "--env", "../../unsafe")

    assert result.returncode == 2
    assert "must be dev, staging, or prod" in result.stderr


def test_build_rejects_symlinked_model_payload_without_touching_repository(
    project_root, tmp_path
):
    project = _minimal_project(tmp_path, Path(project_root) / "scripts" / "build.sh")
    outside_artifact = project / "outside.py"
    outside_artifact.write_text("secret = True\n", encoding="utf-8")
    payload_link = project / "models" / "serving" / "example" / "1" / "leak.py"
    payload_link.symlink_to(outside_artifact)
    marker = project / "model_repository" / "preserved"
    marker.write_text("keep", encoding="utf-8")

    result = subprocess.run(
        [str(project / "scripts" / "build.sh"), "--env", "prod", "--clean"],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHON_BIN": sys.executable},
    )

    assert result.returncode == 1
    assert "contains a symlink: 1/leak.py" in result.stderr
    assert marker.read_text(encoding="utf-8") == "keep"
    assert not (project / "model_repository" / "example").exists()


def test_build_overlays_only_verified_external_artifact(project_root, tmp_path):
    project = _minimal_project(tmp_path, Path(project_root) / "scripts" / "build.sh")
    payload = b"verified-external-model"
    artifact_root = project / ".artifacts"
    artifact_path = artifact_root / "example" / "1" / "model.bin"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(payload)
    manifest_path = project / "models" / "serving" / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["models"][0].update(
        {
            "artifact": "external",
            "required_files": ["1/model.bin"],
            "artifacts": [
                {
                    "path": "1/model.bin",
                    "uri": "https://models.example/model.bin",
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                }
            ],
        }
    )
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    result = subprocess.run(
        [
            str(project / "scripts" / "build.sh"),
            "--env",
            "prod",
            "--artifact-root",
            str(artifact_root),
            "--clean",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHON_BIN": sys.executable},
    )

    assert result.returncode == 0, result.stderr
    assert (
        project / "model_repository" / "example" / "1" / "model.bin"
    ).read_bytes() == payload


def test_build_rechecks_cached_artifact_without_touching_repository(
    project_root, tmp_path
):
    project = _minimal_project(tmp_path, Path(project_root) / "scripts" / "build.sh")
    trusted_payload = b"trusted"
    artifact_root = project / ".artifacts"
    artifact_path = artifact_root / "example" / "1" / "model.bin"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b"changed")
    manifest_path = project / "models" / "serving" / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["models"][0].update(
        {
            "artifact": "external",
            "required_files": ["1/model.bin"],
            "artifacts": [
                {
                    "path": "1/model.bin",
                    "uri": "https://models.example/model.bin",
                    "sha256": hashlib.sha256(trusted_payload).hexdigest(),
                    "size_bytes": len(trusted_payload),
                }
            ],
        }
    )
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    marker = project / "model_repository" / "preserved"
    marker.write_text("keep", encoding="utf-8")

    result = subprocess.run(
        [
            str(project / "scripts" / "build.sh"),
            "--env",
            "prod",
            "--artifact-root",
            str(artifact_root),
            "--clean",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHON_BIN": sys.executable},
    )

    assert result.returncode == 1
    assert "cached artifact SHA-256 mismatch" in result.stderr
    assert marker.read_text(encoding="utf-8") == "keep"
    assert not (project / "model_repository" / "example").exists()
