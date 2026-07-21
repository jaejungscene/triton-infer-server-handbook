#!/usr/bin/env bash
# =============================================================================
# build.sh — manifest.yaml 기반 model_repository 빌드
# =============================================================================
# 사용법:
#   ./scripts/build.sh --env dev
#   ./scripts/build.sh --env prod --tags gpu,tensorrt
#   ./scripts/build.sh --env dev --clean
#
# 이 스크립트가 하는 일:
#   1. models/serving/manifest.yaml 읽기
#   2. source → target 매핑에 따라 model_repository/ 에 복사
#   3. enabled=false 모델과 tag 필터 미일치 모델 제외
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MANIFEST="${PROJECT_ROOT}/models/serving/manifest.yaml"
MODEL_REPO="${PROJECT_ROOT}/model_repository"
MODELS_SRC="${PROJECT_ROOT}/models/serving"

# ─────────────────────────────────────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────────────────────────────────────
ENV="dev"
TAGS=""
CLEAN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --env)     ENV="$2"; shift 2 ;;
        --tags)    TAGS="$2"; shift 2 ;;
        --clean)   CLEAN=true; shift ;;
        --help|-h)
            echo "Usage: $0 --env {dev|staging|prod} [--tags tag1,tag2] [--clean]"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ─────────────────────────────────────────────────────────────────────────────
# Load environment
# ─────────────────────────────────────────────────────────────────────────────
ENV_FILE="${PROJECT_ROOT}/.env.${ENV}"
if [[ -f "${ENV_FILE}" ]]; then
    echo "[build] Loading environment from ${ENV_FILE}"
    set -a
    # shellcheck source=/dev/null
    source "${ENV_FILE}"
    set +a
fi

# ─────────────────────────────────────────────────────────────────────────────
# Clean
# ─────────────────────────────────────────────────────────────────────────────
if [[ "${CLEAN}" == true ]]; then
    echo "[build] Cleaning model_repository/"
    find "${MODEL_REPO}" -mindepth 1 -not -name '.gitkeep' -exec rm -rf {} + 2>/dev/null || true
fi

# ─────────────────────────────────────────────────────────────────────────────
# Check dependencies
# ─────────────────────────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "[build] ERROR: python3 is required"
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# Parse manifest and build
# ─────────────────────────────────────────────────────────────────────────────
echo "[build] Building model_repository from manifest (env=${ENV}, tags=${TAGS:-all})"

python3 - "${MANIFEST}" "${MODELS_SRC}" "${MODEL_REPO}" "${TAGS}" <<'PYTHON_SCRIPT'
import sys
import os
import shutil

try:
    import yaml
except ImportError:
    # Fallback: simple YAML parser for manifest.yaml
    import json
    print("[build] WARNING: PyYAML not installed, using fallback parser", file=sys.stderr)
    yaml = None

manifest_path = sys.argv[1]
models_src = sys.argv[2]
model_repo = sys.argv[3]
filter_tags = set(sys.argv[4].split(",")) if sys.argv[4] else set()

def parse_manifest_simple(path):
    """Simple manifest parser without PyYAML dependency."""
    models = []
    current = {}
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("- source:"):
                if current:
                    models.append(current)
                current = {"source": stripped.split(":", 1)[1].strip()}
            elif stripped.startswith("target:") and current:
                current["target"] = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("tags:") and current:
                tags_str = stripped.split(":", 1)[1].strip().strip("[]")
                current["tags"] = [t.strip() for t in tags_str.split(",")]
            elif stripped.startswith("enabled:") and current:
                current["enabled"] = stripped.split(":", 1)[1].strip().lower() == "true"
    if current:
        models.append(current)
    return models

# Parse manifest
if yaml:
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)
    models = manifest.get("models", [])
else:
    models = parse_manifest_simple(manifest_path)

built = 0
skipped = 0

for model in models:
    # Skip disabled models
    if not model.get("enabled", True):
        print(f"  SKIP (disabled): {model['source']}")
        skipped += 1
        continue

    # Filter by tags
    model_tags = set(model.get("tags", []))
    if filter_tags and not filter_tags.intersection(model_tags):
        print(f"  SKIP (tags): {model['source']}")
        skipped += 1
        continue

    source_path = os.path.join(models_src, model["source"])
    target_path = os.path.join(model_repo, model["target"])

    if not os.path.exists(source_path):
        print(f"  WARNING: Source not found: {source_path}", file=sys.stderr)
        skipped += 1
        continue

    # Remove existing target
    if os.path.exists(target_path):
        shutil.rmtree(target_path)

    # Copy source to target
    shutil.copytree(source_path, target_path)
    print(f"  OK: {model['source']} -> {model['target']}")
    built += 1

print(f"\n[build] Done: {built} models built, {skipped} skipped")
PYTHON_SCRIPT

echo "[build] model_repository contents:"
ls -la "${MODEL_REPO}/" 2>/dev/null || echo "  (empty)"
echo "[build] Build complete (env=${ENV})"
