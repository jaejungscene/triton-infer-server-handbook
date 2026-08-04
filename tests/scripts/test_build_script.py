import shutil
import subprocess
from pathlib import Path


def _run_build(project_root, *args):
    return subprocess.run(
        [f"{project_root}/scripts/build.sh", *args],
        check=False,
        capture_output=True,
        text=True,
    )


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
