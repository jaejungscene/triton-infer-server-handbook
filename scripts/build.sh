#!/usr/bin/env bash
# Build model_repository from the validated serving manifest.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MANIFEST="${PROJECT_ROOT}/models/serving/manifest.yaml"
MODEL_REPO="${PROJECT_ROOT}/model_repository"
MODELS_SRC="${PROJECT_ROOT}/models/serving"

BUILD_ENV="dev"
TAGS=""
CLEAN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            [[ $# -ge 2 ]] || { echo "--env requires a value" >&2; exit 2; }
            BUILD_ENV="$2"
            shift 2
            ;;
        --tags)
            [[ $# -ge 2 ]] || { echo "--tags requires a value" >&2; exit 2; }
            TAGS="$2"
            shift 2
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 --env {dev|staging|prod} [--tags tag1,tag2] [--clean]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
    esac
done

if [[ ! "${BUILD_ENV}" =~ ^(dev|staging|prod)$ ]]; then
    echo "[build] --env must be dev, staging, or prod" >&2
    exit 2
fi
if [[ -n "${TAGS}" && ! "${TAGS}" =~ ^[A-Za-z0-9._-]+(,[A-Za-z0-9._-]+)*$ ]]; then
    echo "[build] --tags must be a comma-separated list of safe tag names" >&2
    exit 2
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
    python_bin="${PYTHON_BIN}"
elif [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    python_bin="${PROJECT_ROOT}/.venv/bin/python"
else
    python_bin="python3"
fi

if ! command -v "${python_bin}" > /dev/null 2>&1; then
    echo "[build] Python 3 is required" >&2
    exit 1
fi

echo "[build] Validating and staging models (env=${BUILD_ENV}, tags=${TAGS:-all})"

"${python_bin}" - "${MANIFEST}" "${MODELS_SRC}" "${MODEL_REPO}" "${TAGS}" "${CLEAN}" <<'PYTHON_SCRIPT'
from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

try:
    import yaml
except ImportError:
    print(
        "[build] PyYAML is required. Run: python -m pip install -r requirements-dev.txt",
        file=sys.stderr,
    )
    raise SystemExit(1)


manifest_path = Path(sys.argv[1]).resolve()
models_src = Path(sys.argv[2]).resolve()
model_repo = Path(sys.argv[3]).resolve()
filter_tags = set(filter(None, sys.argv[4].split(",")))
clean = sys.argv[5].lower() == "true"
safe_name = re.compile(r"^[A-Za-z0-9._-]+$")


def contained_path(base: Path, relative: str, field: str) -> Path:
    candidate = (base / relative).resolve()
    if not candidate.is_relative_to(base):
        raise ValueError(f"{field} escapes its allowed directory: {relative}")
    return candidate


with manifest_path.open(encoding="utf-8") as manifest_file:
    manifest = yaml.safe_load(manifest_file)

if not isinstance(manifest, dict) or not isinstance(manifest.get("models"), list):
    raise SystemExit("[build] manifest.yaml must contain a models list")

selected_models = []
targets = set()
for index, model in enumerate(manifest["models"]):
    if not isinstance(model, dict):
        raise SystemExit(f"[build] models[{index}] must be an object")

    source = model.get("source")
    target = model.get("target")
    if not isinstance(source, str) or not source:
        raise SystemExit(f"[build] models[{index}].source is required")
    if not isinstance(target, str) or not safe_name.fullmatch(target):
        raise SystemExit(f"[build] invalid target name at models[{index}]: {target}")
    if target in targets:
        raise SystemExit(f"[build] duplicate target: {target}")
    targets.add(target)

    tags = model.get("tags", [])
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        raise SystemExit(f"[build] models[{index}].tags must be a string list")
    if not isinstance(model.get("enabled", True), bool):
        raise SystemExit(f"[build] models[{index}].enabled must be boolean")
    required_files = model.get("required_files", [])
    if not isinstance(required_files, list) or not all(
        isinstance(required_file, str) for required_file in required_files
    ):
        raise SystemExit(f"[build] models[{index}].required_files must be a string list")

    source_path = contained_path(models_src, source, "source")
    if not model.get("enabled", True):
        print(f"  SKIP (disabled): {source}")
        continue
    if filter_tags and not filter_tags.intersection(tags):
        print(f"  SKIP (tags): {source}")
        continue
    if not source_path.is_dir():
        raise SystemExit(f"[build] enabled model source not found: {source_path}")

    for required_file in required_files:
        required_path = contained_path(source_path, required_file, "required_files")
        if not required_path.is_file():
            raise SystemExit(
                f"[build] enabled model {target} is missing artifact: {required_file}"
            )

    selected_models.append((source, source_path, target))

if not selected_models:
    raise SystemExit("[build] no enabled models matched the requested tags")

model_repo.mkdir(parents=True, exist_ok=True)
staging_root = Path(tempfile.mkdtemp(prefix=".triton-build-", dir=model_repo))
ignore_files = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")

try:
    for source, source_path, target in selected_models:
        shutil.copytree(source_path, staging_root / target, ignore=ignore_files)
        print(f"  STAGED: {source} -> {target}")

    if clean:
        for child in model_repo.iterdir():
            if child.name in {".gitkeep", staging_root.name}:
                continue
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()

    for _, _, target in selected_models:
        target_path = contained_path(model_repo, target, "target")
        if target_path.exists():
            if target_path.is_dir() and not target_path.is_symlink():
                shutil.rmtree(target_path)
            else:
                target_path.unlink()
        os.replace(staging_root / target, target_path)
        print(f"  BUILT: {target}")
finally:
    shutil.rmtree(staging_root, ignore_errors=True)

print(f"\n[build] Done: {len(selected_models)} model(s) built")
PYTHON_SCRIPT

echo "[build] model_repository contents:"
ls -la "${MODEL_REPO}/"
echo "[build] Build complete (env=${BUILD_ENV})"
